package sk.ainet.models.gemma3n

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
 * Loads Gemma 3n weights from HuggingFace SafeTensors format.
 *
 * HuggingFace models use different tensor naming conventions than GGUF.
 * This loader maps HuggingFace names to GGUF-style names used by the runtime.
 *
 * Supports sharded models (multiple .safetensors files with index.json).
 *
 * Key differences from GGUF:
 * - Different tensor name format (model.language_model.layers.X.* vs blk.X.*)
 * - Weight tying: embed_tokens.weight is reused for output projection
 *
 * Engine delegation (SKaiNET#1246): reading AND per-tensor materialization ride the engine's
 * [ShardedSafeTensorsParametersLoader]. This class owns only the family policy — which HF
 * tensors are wanted (the `tensorFilter`), the HF → GGUF slot renaming, and the PLE-table size
 * guard — while every dtype decision (BF16/F16 widening vs keep-native, F32 passthrough) is the
 * engine's, driven by [dtypePolicy] exactly as on the GGUF lane. No per-family dequant code
 * remains here (#346).
 *
 * @param indexPath path to `model.safetensors.index.json`.
 * @param dtypePolicy narrow-float policy forwarded to
 *   [ShardedSafeTensorsParametersLoader.withPolicy]: `Any` (default) widens BF16/F16 to FP32;
 *   `Require(BF16)` / `Require(FP16)` keep the native encoding under an FP32-typed tensor.
 */
public class Gemma3nSafeTensorsWeightLoader(
    private val indexPath: String,
    private val dtypePolicy: DTypePolicy = DTypePolicy.Any,
) {

    /**
     * Load weights into a map, mapping HuggingFace names to GGUF-style names.
     *
     * @param ctx Execution context for tensor operations
     * @param dtype Target dtype (FP32)
     * @return Gemma3nWeights with mapped tensor names
     */
    public suspend fun <T : DType> loadToMap(
        ctx: ExecutionContext,
        dtype: KClass<T>
    ): Gemma3nWeights<T, Float> {
        require(dtype == FP32::class) {
            "SafeTensors loader currently only supports FP32 (got ${dtype.simpleName})"
        }

        // Read config.json from same directory
        val basePath = indexPath.substringBeforeLast("/")
        val configPath = if (basePath.isEmpty()) "config.json" else "$basePath/config.json"
        val configJson = readTextFile(configPath)
            ?: error("config.json not found at $configPath")

        val metadata = Gemma3nConfigParser.parseFromJson(configJson)

        // Family policy as the engine loader's filter: only the tensors this family maps
        // (vision/audio towers and anything unmapped never materialize, and are exempt from the
        // fail-fast dtype pre-scan), and the PLE table only when it fits the eager path — the
        // engine returns each tensor as one array, which the JVM caps at Int.MAX_VALUE bytes;
        // an oversized table is left absent, which disables PLE downstream instead of failing.
        val wanted = wantedHfNames(metadata.blockCount)
        val loader = ShardedSafeTensorsParametersLoader.withPolicy(
            indexPath = indexPath,
            policy = dtypePolicy,
            tensorFilter = { info: ShardedTensorInfo ->
                info.name in wanted && (info.name != HF_PER_LAYER_TOKEN_EMBD || info.sizeInBytes <= MAX_BYTES_PER_TENSOR)
            },
        )
        val tensorsByHfName = linkedMapOf<String, Tensor<T, Float>>()
        loader.load<T, Float>(ctx, dtype) { name, tensor -> tensorsByHfName[name] = tensor }

        val tensorsByGgufName = linkedMapOf<String, Tensor<T, Float>>()
        loadGlobalTensors(tensorsByHfName, tensorsByGgufName)
        for (layer in 0 until metadata.blockCount) {
            for ((hfName, ggufName) in layerSlots(layer)) {
                tensorsByHfName[hfName]?.let { tensorsByGgufName[ggufName] = it }
            }
        }

        return Gemma3nWeights(
            metadata = metadata,
            tensors = tensorsByGgufName,
        )
    }

    private fun <T : DType> loadGlobalTensors(
        tensorsByHfName: Map<String, Tensor<T, Float>>,
        tensorsByGgufName: MutableMap<String, Tensor<T, Float>>
    ) {
        // Token embeddings, kept as [vocab_size, embedding_dim]
        val embedTensor = tensorsByHfName[HF_EMBED_TOKENS]
            ?: error("Missing tensor: $HF_EMBED_TOKENS")
        tensorsByGgufName[Gemma3nTensorNames.TOKEN_EMBEDDINGS] = embedTensor

        // Output norm
        tensorsByGgufName[Gemma3nTensorNames.OUTPUT_NORM] = tensorsByHfName[HF_OUTPUT_NORM]
            ?: error("Missing tensor: $HF_OUTPUT_NORM")

        // Output weight (weight tying - reuse embed_tokens)
        tensorsByGgufName[Gemma3nTensorNames.OUTPUT_WEIGHT] = embedTensor

        // Optional globals: AltUp (E4B) and per-layer embedding tensors.
        for ((hfName, ggufName) in GLOBAL_OPTIONAL_SLOTS) {
            tensorsByHfName[hfName]?.let { tensorsByGgufName[ggufName] = it }
        }
    }

    // ========== HuggingFace Tensor Name Constants ==========

    private companion object {
        // The engine's eager loader path returns each tensor as a single
        // array, which the JVM caps at Int.MAX_VALUE bytes. Keep a little headroom.
        const val MAX_BYTES_PER_TENSOR: Long = Int.MAX_VALUE.toLong() - 1024L

        // Global tensors
        const val HF_EMBED_TOKENS = "model.language_model.embed_tokens.weight"
        const val HF_OUTPUT_NORM = "model.language_model.norm.weight"

        // Global AltUp tensors (E4B)
        const val HF_ALTUP_PROJ = "model.language_model.altup_proj.weight"
        const val HF_ALTUP_UNEMBD_PROJ = "model.language_model.altup_unembd_proj.weight"

        // Global per-layer embedding tensors (E4B)
        const val HF_PER_LAYER_TOKEN_EMBD = "model.language_model.per_layer_token_embd.weight"
        const val HF_PER_LAYER_MODEL_PROJ = "model.language_model.per_layer_model_proj.weight"
        const val HF_PER_LAYER_PROJ_NORM = "model.language_model.per_layer_proj_norm.weight"

        /** Optional global HF → GGUF slot pairs. */
        val GLOBAL_OPTIONAL_SLOTS: List<Pair<String, String>> by lazy {
            listOf(
                HF_ALTUP_PROJ to Gemma3nTensorNames.ALTUP_PROJ,
                HF_ALTUP_UNEMBD_PROJ to Gemma3nTensorNames.ALTUP_UNEMBD_PROJ,
                HF_PER_LAYER_TOKEN_EMBD to Gemma3nTensorNames.PER_LAYER_TOKEN_EMBD,
                HF_PER_LAYER_MODEL_PROJ to Gemma3nTensorNames.PER_LAYER_MODEL_PROJ,
                HF_PER_LAYER_PROJ_NORM to Gemma3nTensorNames.PER_LAYER_PROJ_NORM,
            )
        }

        /** Every HF tensor this family maps, for [blockCount] layers — the engine loader's allowlist. */
        fun wantedHfNames(blockCount: Int): Set<String> = buildSet {
            add(HF_EMBED_TOKENS); add(HF_OUTPUT_NORM)
            addAll(GLOBAL_OPTIONAL_SLOTS.map { it.first })
            for (layer in 0 until blockCount) addAll(layerSlots(layer).map { it.first })
        }

        /** Per-layer HF → GGUF slot pairs; all optional (E2B vs E4B differ), mapped when present. */
        fun layerSlots(layer: Int): List<Pair<String, String>> = listOf(
            hfInputLayernorm(layer) to Gemma3nTensorNames.inputLayernorm(layer),
            hfAttnQ(layer) to Gemma3nTensorNames.attnQ(layer),
            hfAttnK(layer) to Gemma3nTensorNames.attnK(layer),
            hfAttnV(layer) to Gemma3nTensorNames.attnV(layer),
            hfAttnO(layer) to Gemma3nTensorNames.attnOut(layer),
            hfPostAttnLayernorm(layer) to Gemma3nTensorNames.postAttentionLayernorm(layer),
            hfMlpGate(layer) to Gemma3nTensorNames.ffnGate(layer),
            hfMlpUp(layer) to Gemma3nTensorNames.ffnUp(layer),
            hfMlpDown(layer) to Gemma3nTensorNames.ffnDown(layer),
            hfPerLayerProjection(layer) to Gemma3nTensorNames.perLayerInput(layer),
            // E4B per-layer AltUp tensors
            hfAltupPredictCoef(layer) to Gemma3nTensorNames.altupPredictCoef(layer),
            hfAltupCorrectCoef(layer) to Gemma3nTensorNames.altupCorrectCoef(layer),
            hfAltupCorrectScale(layer) to Gemma3nTensorNames.altupCorrectScale(layer),
            hfAltupRouter(layer) to Gemma3nTensorNames.altupRouter(layer),
            hfAltupRouterNorm(layer) to Gemma3nTensorNames.altupRouterNorm(layer),
            // E4B additional norms and weights
            hfAttnQNorm(layer) to Gemma3nTensorNames.attnQNorm(layer),
            hfAttnKNorm(layer) to Gemma3nTensorNames.attnKNorm(layer),
            hfPostAttentionNorm(layer) to Gemma3nTensorNames.postAttentionNorm(layer),
            hfPostFfwNorm(layer) to Gemma3nTensorNames.postFfwNorm(layer),
            hfPostNorm(layer) to Gemma3nTensorNames.postNorm(layer),
            hfInputGate(layer) to Gemma3nTensorNames.inputGate(layer),
            hfProj(layer) to Gemma3nTensorNames.proj(layer),
            hfLaurelL(layer) to Gemma3nTensorNames.laurelL(layer),
            hfLaurelR(layer) to Gemma3nTensorNames.laurelR(layer),
            hfLaurelPostNorm(layer) to Gemma3nTensorNames.laurelPostNorm(layer),
        )

        // Layer tensor name builders
        fun hfInputLayernorm(layer: Int) = "model.language_model.layers.$layer.input_layernorm.weight"
        fun hfAttnQ(layer: Int) = "model.language_model.layers.$layer.self_attn.q_proj.weight"
        fun hfAttnK(layer: Int) = "model.language_model.layers.$layer.self_attn.k_proj.weight"
        fun hfAttnV(layer: Int) = "model.language_model.layers.$layer.self_attn.v_proj.weight"
        fun hfAttnO(layer: Int) = "model.language_model.layers.$layer.self_attn.o_proj.weight"
        fun hfPostAttnLayernorm(layer: Int) = "model.language_model.layers.$layer.post_attention_layernorm.weight"
        fun hfMlpGate(layer: Int) = "model.language_model.layers.$layer.mlp.gate_proj.weight"
        fun hfMlpUp(layer: Int) = "model.language_model.layers.$layer.mlp.up_proj.weight"
        fun hfMlpDown(layer: Int) = "model.language_model.layers.$layer.mlp.down_proj.weight"
        fun hfPerLayerProjection(layer: Int) = "model.language_model.layers.$layer.per_layer_projection.weight"

        // E4B per-layer AltUp
        fun hfAltupPredictCoef(layer: Int) = "model.language_model.layers.$layer.altup.predict_coef.weight"
        fun hfAltupCorrectCoef(layer: Int) = "model.language_model.layers.$layer.altup.correct_coef.weight"
        fun hfAltupCorrectScale(layer: Int) = "model.language_model.layers.$layer.altup.correct_scale.weight"
        fun hfAltupRouter(layer: Int) = "model.language_model.layers.$layer.altup.router.weight"
        fun hfAltupRouterNorm(layer: Int) = "model.language_model.layers.$layer.altup.router_norm.weight"

        // E4B additional per-layer tensors
        fun hfAttnQNorm(layer: Int) = "model.language_model.layers.$layer.self_attn.q_norm.weight"
        fun hfAttnKNorm(layer: Int) = "model.language_model.layers.$layer.self_attn.k_norm.weight"
        fun hfPostAttentionNorm(layer: Int) = "model.language_model.layers.$layer.post_attention_norm.weight"
        fun hfPostFfwNorm(layer: Int) = "model.language_model.layers.$layer.post_ffw_norm.weight"
        fun hfPostNorm(layer: Int) = "model.language_model.layers.$layer.post_norm.weight"
        fun hfInputGate(layer: Int) = "model.language_model.layers.$layer.inp_gate.weight"
        fun hfProj(layer: Int) = "model.language_model.layers.$layer.proj.weight"
        fun hfLaurelL(layer: Int) = "model.language_model.layers.$layer.laurel_l.weight"
        fun hfLaurelR(layer: Int) = "model.language_model.layers.$layer.laurel_r.weight"
        fun hfLaurelPostNorm(layer: Int) = "model.language_model.layers.$layer.laurel_post_norm.weight"
    }
}

/**
 * Load Gemma 3n runtime weights from SafeTensors format.
 */
public suspend fun <T : DType> loadGemma3nRuntimeWeightsFromSafeTensors(
    ctx: ExecutionContext,
    indexPath: String,
    dtype: KClass<T>
): Gemma3nRuntimeWeights<T> {
    val loader = Gemma3nSafeTensorsWeightLoader(indexPath)
    val loaded = loader.loadToMap<T>(ctx, dtype)
    return Gemma3nWeightMapper.map(loaded)
}

/** Backward-compatible overload defaulting to FP32. */
public suspend fun loadGemma3nRuntimeWeightsFromSafeTensors(
    ctx: ExecutionContext,
    indexPath: String
): Gemma3nRuntimeWeights<FP32> = loadGemma3nRuntimeWeightsFromSafeTensors(ctx, indexPath, FP32::class)
