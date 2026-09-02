package sk.ainet.models.gemma

import sk.ainet.context.ExecutionContext
import sk.ainet.io.load
import sk.ainet.io.safetensors.ShardedSafeTensorsParametersLoader
import sk.ainet.io.safetensors.ShardedTensorInfo
import sk.ainet.io.safetensors.readTextFile
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.DTypePolicy
import sk.ainet.lang.types.FP32
import kotlin.reflect.KClass

/**
 * Loads Gemma 4 weights from HuggingFace SafeTensors format.
 *
 * Maps HuggingFace tensor names to GGUF-style names used by the runtime.
 * Supports sharded models (multiple .safetensors files with index.json).
 *
 * Key differences from Gemma 3n loader:
 * - Uses GemmaConfigParser instead of Gemma3nConfigParser
 * - No AltUp, Laurel, or activation sparsity tensors
 * - Per-layer head dim may vary (global_head_dim vs head_dim)
 *
 * Engine delegation (#375 → SKaiNET#1246): reading AND per-tensor materialization ride the
 * engine's [ShardedSafeTensorsParametersLoader]. This class owns only the family policy —
 * which HF tensors are wanted (the `tensorFilter`), the HF → GGUF slot renaming, and the PLE
 * size guard — while every dtype decision (BF16/F16 widening vs keep-native, F32 passthrough)
 * is the engine's, driven by [dtypePolicy] exactly as on the GGUF lane. No per-family dequant
 * code remains here (#346).
 *
 * @param indexPath path to `model.safetensors.index.json`.
 * @param dtypePolicy narrow-float policy forwarded to
 *   [ShardedSafeTensorsParametersLoader.withPolicy]: `Any` (default) widens BF16/F16 to FP32;
 *   `Require(BF16)` / `Require(FP16)` keep the native encoding under an FP32-typed tensor.
 */
public class GemmaSafeTensorsLoader(
    private val indexPath: String,
    private val dtypePolicy: DTypePolicy = DTypePolicy.Any,
) {

    public suspend fun <T : DType> loadToMap(
        ctx: ExecutionContext,
        dtype: KClass<T>
    ): GemmaWeights<T, Float> {
        require(dtype == FP32::class) {
            "SafeTensors loader currently only supports FP32 (got ${dtype.simpleName})"
        }

        val basePath = indexPath.substringBeforeLast("/")
        val configPath = if (basePath.isEmpty()) "config.json" else "$basePath/config.json"
        val configJson = readTextFile(configPath)
            ?: error("config.json not found at $configPath")

        val metadata = GemmaConfigParser.parseFromJson(configJson)

        // Family policy, expressed as the engine loader's filter: only the tensors this
        // family maps (vision/audio towers and anything unmapped never materialize, and are
        // exempt from the fail-fast dtype pre-scan), and the PLE table only when it fits the
        // eager ByteArray path — see the note in loadGlobalTensors.
        val wanted = wantedHfNames(metadata.blockCount)
        val loader = ShardedSafeTensorsParametersLoader.withPolicy(
            indexPath = indexPath,
            policy = dtypePolicy,
            tensorFilter = { info: ShardedTensorInfo ->
                info.name in wanted && (info.name !in PLE_TABLE_CANDIDATES || info.sizeInBytes <= MAX_BYTES_PER_TENSOR)
            },
        )
        val tensorsByHfName = linkedMapOf<String, Tensor<T, Float>>()
        loader.load<T, Float>(ctx, dtype) { name, tensor -> tensorsByHfName[name] = tensor }

        val tensorsByGgufName = linkedMapOf<String, Tensor<T, Float>>()
        loadGlobalTensors(tensorsByHfName, tensorsByGgufName)
        for (layer in 0 until metadata.blockCount) {
            loadLayerTensors(tensorsByHfName, tensorsByGgufName, layer)
        }

        return GemmaWeights(
            metadata = metadata,
            tensors = tensorsByGgufName
        )
    }

    private fun <T : DType> loadGlobalTensors(
        tensorsByHfName: Map<String, Tensor<T, Float>>,
        tensorsByGgufName: MutableMap<String, Tensor<T, Float>>
    ) {
        // Token embeddings
        val embedTensor = tensorsByHfName[HF_EMBED_TOKENS]
            ?: error("Missing tensor: $HF_EMBED_TOKENS")
        tensorsByGgufName[GemmaTensorNames.TOKEN_EMBEDDINGS] = embedTensor

        // Output norm
        tensorsByGgufName[GemmaTensorNames.OUTPUT_NORM] = tensorsByHfName[HF_OUTPUT_NORM]
            ?: error("Missing tensor: $HF_OUTPUT_NORM")

        // Output weight (weight tying - reuse embed_tokens)
        tensorsByGgufName[GemmaTensorNames.OUTPUT_WEIGHT] = embedTensor

        // Optional PLE global tensors. HF names changed in Gemma 4 vs the
        // earlier "per_layer_*" convention used by Gemma 3n / GGUF; map both
        // so synthetic GGUF-style fixtures keep working alongside real HF
        // checkpoints.
        //
        // The token-embeddings-per-layer tensor on real Gemma-4 E2B is
        // [vocab_size, num_layers, per_layer_dim] in BF16 — 4.7 GB raw, well
        // over the 2 GB JVM ByteArray limit the eager reader path is bound by.
        // The GGUF path sidesteps this by keeping the table Q6_K-packed
        // (~1.8 GB) with a dedicated row-dequant data type. On SafeTensors the
        // tensorFilter (see loadToMap) skips the table when it is too large —
        // leaving `per_layer_token_embd.weight` absent from the tensor map,
        // which auto-disables PLE in `GemmaNetworkLoader`. The model still
        // runs (sandwich norms, layer_output_scale and softcap are intact)
        // but without the per-layer side-channel signal; the JVM
        // `GemmaSafeTensorsMappedPle` path injects it file-backed afterwards.
        mapFirstExisting(tensorsByHfName, tensorsByGgufName,
            GemmaTensorNames.PER_LAYER_TOKEN_EMBD, PLE_TABLE_CANDIDATES)
        mapFirstExisting(tensorsByHfName, tensorsByGgufName,
            GemmaTensorNames.PER_LAYER_MODEL_PROJ, listOf(HF_PER_LAYER_MODEL_PROJECTION, HF_PER_LAYER_MODEL_PROJ))
        mapFirstExisting(tensorsByHfName, tensorsByGgufName,
            GemmaTensorNames.PER_LAYER_PROJ_NORM, listOf(HF_PER_LAYER_PROJECTION_NORM, HF_PER_LAYER_PROJ_NORM))
    }

    private fun <T : DType> loadLayerTensors(
        tensorsByHfName: Map<String, Tensor<T, Float>>,
        tensorsByGgufName: MutableMap<String, Tensor<T, Float>>,
        layer: Int
    ) {
        for ((hfName, ggufName) in layerSlots(layer)) {
            mapIfExists(tensorsByHfName, tensorsByGgufName, hfName, ggufName)
        }
    }

    private fun <T : DType> mapIfExists(
        tensorsByHfName: Map<String, Tensor<T, Float>>,
        tensorsByGgufName: MutableMap<String, Tensor<T, Float>>,
        hfName: String,
        ggufName: String
    ) {
        tensorsByHfName[hfName]?.let { tensorsByGgufName[ggufName] = it }
    }

    private fun <T : DType> mapFirstExisting(
        tensorsByHfName: Map<String, Tensor<T, Float>>,
        tensorsByGgufName: MutableMap<String, Tensor<T, Float>>,
        ggufName: String,
        hfCandidates: List<String>
    ) {
        for (hfName in hfCandidates) {
            val tensor = tensorsByHfName[hfName] ?: continue
            tensorsByGgufName[ggufName] = tensor
            return
        }
    }

    private companion object {
        // The engine's eager loader path returns each tensor as a single
        // ByteArray, which the JVM caps at Int.MAX_VALUE bytes. Keep a
        // little headroom for safety.
        const val MAX_BYTES_PER_TENSOR: Long = Int.MAX_VALUE.toLong() - 1024L

        /** HF names of the PLE table, new-name first; the only tensors subject to the size guard. */
        val PLE_TABLE_CANDIDATES: List<String> by lazy {
            listOf(HF_EMBED_TOKENS_PER_LAYER, HF_PER_LAYER_TOKEN_EMBD)
        }

        /** Every HF tensor this family maps, for [blockCount] layers — the engine loader's allowlist. */
        fun wantedHfNames(blockCount: Int): Set<String> = buildSet {
            add(HF_EMBED_TOKENS); add(HF_OUTPUT_NORM)
            addAll(PLE_TABLE_CANDIDATES)
            add(HF_PER_LAYER_MODEL_PROJECTION); add(HF_PER_LAYER_MODEL_PROJ)
            add(HF_PER_LAYER_PROJECTION_NORM); add(HF_PER_LAYER_PROJ_NORM)
            for (layer in 0 until blockCount) addAll(layerSlots(layer).map { it.first })
        }

        /**
         * Per-layer HF → GGUF slot pairs, in the order the GGUF lane populates them.
         *
         * Sandwich norms — Gemma 4 has FOUR norms per block:
         *  - attn_norm           ← input_layernorm
         *  - post_attention_norm ← post_attention_layernorm    (Gemma-4 only)
         *  - ffn_norm            ← pre_feedforward_layernorm   (Gemma-4 only)
         *  - post_ffw_norm       ← post_feedforward_layernorm  (Gemma-4 only)
         * The GGUF naming on `GemmaTensorNames.postAttentionLayernorm` is a legacy alias for
         * the pre-FFN norm slot (`ffn_norm`); the proper Gemma-4 source for that slot is HF
         * `pre_feedforward_layernorm`.
         *
         * PLE: the HF projection sends [B,S,perLayerDim] back into the residual at
         * hidden_size, so it goes into the `proj` slot, NOT `per_layer_input`.
         */
        fun layerSlots(layer: Int): List<Pair<String, String>> = listOf(
            hfInputLayernorm(layer) to GemmaTensorNames.inputLayernorm(layer),
            hfAttnQ(layer) to GemmaTensorNames.attnQ(layer),
            hfAttnK(layer) to GemmaTensorNames.attnK(layer),
            hfAttnV(layer) to GemmaTensorNames.attnV(layer),
            hfAttnO(layer) to GemmaTensorNames.attnOut(layer),
            hfPostAttnLayernorm(layer) to GemmaTensorNames.postAttentionNorm(layer),
            hfPreFfwLayernorm(layer) to GemmaTensorNames.postAttentionLayernorm(layer),
            hfPostFfwLayernorm(layer) to GemmaTensorNames.postFfwNorm(layer),
            hfMlpGate(layer) to GemmaTensorNames.ffnGate(layer),
            hfMlpUp(layer) to GemmaTensorNames.ffnUp(layer),
            hfMlpDown(layer) to GemmaTensorNames.ffnDown(layer),
            // Per-layer scalar (HF: `layer_scalar`, no `.weight` suffix; scalar shape).
            hfLayerScalar(layer) to GemmaTensorNames.layerOutputScale(layer),
            hfPerLayerInputGate(layer) to GemmaTensorNames.pleInpGate(layer),
            hfPerLayerProjection(layer) to GemmaTensorNames.pleProj(layer),
            hfPostPerLayerInputNorm(layer) to GemmaTensorNames.plePostNorm(layer),
            // Optional QK normalization (per-head Q/K RMSNorm).
            hfAttnQNorm(layer) to GemmaTensorNames.attnQNorm(layer),
            hfAttnKNorm(layer) to GemmaTensorNames.attnKNorm(layer),
        )

        const val HF_EMBED_TOKENS = "model.language_model.embed_tokens.weight"
        const val HF_OUTPUT_NORM = "model.language_model.norm.weight"

        // Gemma 4 (HF transformers ≥ 5.5) PLE global tensor names.
        const val HF_EMBED_TOKENS_PER_LAYER = "model.language_model.embed_tokens_per_layer.weight"
        const val HF_PER_LAYER_MODEL_PROJECTION = "model.language_model.per_layer_model_projection.weight"
        const val HF_PER_LAYER_PROJECTION_NORM = "model.language_model.per_layer_projection_norm.weight"

        // Legacy aliases (Gemma 3n / GGUF-converted checkpoints). Tried as a
        // fallback if the new HF names are absent.
        const val HF_PER_LAYER_TOKEN_EMBD = "model.language_model.per_layer_token_embd.weight"
        const val HF_PER_LAYER_MODEL_PROJ = "model.language_model.per_layer_model_proj.weight"
        const val HF_PER_LAYER_PROJ_NORM = "model.language_model.per_layer_proj_norm.weight"

        fun hfInputLayernorm(layer: Int) = "model.language_model.layers.$layer.input_layernorm.weight"
        fun hfAttnQ(layer: Int) = "model.language_model.layers.$layer.self_attn.q_proj.weight"
        fun hfAttnK(layer: Int) = "model.language_model.layers.$layer.self_attn.k_proj.weight"
        fun hfAttnV(layer: Int) = "model.language_model.layers.$layer.self_attn.v_proj.weight"
        fun hfAttnO(layer: Int) = "model.language_model.layers.$layer.self_attn.o_proj.weight"
        fun hfPostAttnLayernorm(layer: Int) = "model.language_model.layers.$layer.post_attention_layernorm.weight"
        fun hfPreFfwLayernorm(layer: Int) = "model.language_model.layers.$layer.pre_feedforward_layernorm.weight"
        fun hfPostFfwLayernorm(layer: Int) = "model.language_model.layers.$layer.post_feedforward_layernorm.weight"
        fun hfLayerScalar(layer: Int) = "model.language_model.layers.$layer.layer_scalar"
        fun hfMlpGate(layer: Int) = "model.language_model.layers.$layer.mlp.gate_proj.weight"
        fun hfMlpUp(layer: Int) = "model.language_model.layers.$layer.mlp.up_proj.weight"
        fun hfMlpDown(layer: Int) = "model.language_model.layers.$layer.mlp.down_proj.weight"
        fun hfPerLayerProjection(layer: Int) = "model.language_model.layers.$layer.per_layer_projection.weight"
        fun hfPerLayerInputGate(layer: Int) = "model.language_model.layers.$layer.per_layer_input_gate.weight"
        fun hfPostPerLayerInputNorm(layer: Int) = "model.language_model.layers.$layer.post_per_layer_input_norm.weight"
        fun hfAttnQNorm(layer: Int) = "model.language_model.layers.$layer.self_attn.q_norm.weight"
        fun hfAttnKNorm(layer: Int) = "model.language_model.layers.$layer.self_attn.k_norm.weight"
    }
}

/**
 * Load Gemma 4 runtime weights from SafeTensors format.
 */
public suspend fun <T : DType> loadGemmaRuntimeWeightsFromSafeTensors(
    ctx: ExecutionContext,
    indexPath: String,
    dtype: KClass<T>
): GemmaRuntimeWeights<T> {
    val loader = GemmaSafeTensorsLoader(indexPath)
    val loaded = loader.loadToMap<T>(ctx, dtype)
    return GemmaWeightMapper.map(loaded)
}

/** Backward-compatible overload defaulting to FP32. */
public suspend fun loadGemmaRuntimeWeightsFromSafeTensors(
    ctx: ExecutionContext,
    indexPath: String
): GemmaRuntimeWeights<FP32> = loadGemmaRuntimeWeightsFromSafeTensors(ctx, indexPath, FP32::class)
