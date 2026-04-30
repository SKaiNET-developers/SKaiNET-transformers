package sk.ainet.models.gemma

import sk.ainet.context.ExecutionContext
import sk.ainet.io.gguf.dequant.DequantOps
import sk.ainet.io.safetensors.ShardedTensorInfo
import sk.ainet.io.safetensors.StreamingShardedSafeTensorsReader
import sk.ainet.io.safetensors.readTextFile
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import kotlin.reflect.KClass

/**
 * Loads Gemma 4 weights from HuggingFace SafeTensors format.
 *
 * Maps HuggingFace tensor names to GGUF-style names used by the runtime.
 * Supports sharded models (multiple .safetensors files with index.json).
 *
 * Key differences from Gemma 3n loader:
 * - Uses Gemma4ConfigParser instead of Gemma3nConfigParser
 * - No AltUp, Laurel, or activation sparsity tensors
 * - Per-layer head dim may vary (global_head_dim vs head_dim)
 */
public class Gemma4SafeTensorsWeightLoader(
    private val indexPath: String
) {

    public suspend fun <T : DType> loadToMap(
        ctx: ExecutionContext,
        dtype: KClass<T>
    ): Gemma4Weights<T, Float> {
        require(dtype == FP32::class) {
            "SafeTensors loader currently only supports FP32 (got ${dtype.simpleName})"
        }

        val basePath = indexPath.substringBeforeLast("/")
        val configPath = if (basePath.isEmpty()) "config.json" else "$basePath/config.json"
        val configJson = readTextFile(configPath)
            ?: error("config.json not found at $configPath")

        val metadata = Gemma4ConfigParser.parseFromJson(configJson)

        val reader = StreamingShardedSafeTensorsReader.openFromIndex(indexPath)

        return reader.use {
            loadFromReader(ctx, dtype, it, metadata)
        }
    }

    private fun <T : DType> loadFromReader(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: StreamingShardedSafeTensorsReader,
        metadata: Gemma4ModelMetadata
    ): Gemma4Weights<T, Float> {
        val tensorsByGgufName = linkedMapOf<String, Tensor<T, Float>>()
        val tensorsByHfName = reader.tensors.associateBy { it.name }

        loadGlobalTensors(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName)
        for (layer in 0 until metadata.blockCount) {
            loadLayerTensors(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName, layer)
        }

        return Gemma4Weights(
            metadata = metadata,
            tensors = tensorsByGgufName,
            quantTypes = emptyMap()
        )
    }

    private fun <T : DType> loadGlobalTensors(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: StreamingShardedSafeTensorsReader,
        tensorsByHfName: Map<String, ShardedTensorInfo>,
        tensorsByGgufName: MutableMap<String, Tensor<T, Float>>
    ) {
        // Token embeddings
        val embedTokens = tensorsByHfName[HF_EMBED_TOKENS]
            ?: error("Missing tensor: $HF_EMBED_TOKENS")
        val embedTensor = loadAndConvertTensor<T>(ctx, dtype, reader, embedTokens, transpose = false)
        tensorsByGgufName[Gemma4TensorNames.TOKEN_EMBEDDINGS] = embedTensor

        // Output norm
        val norm = tensorsByHfName[HF_OUTPUT_NORM]
            ?: error("Missing tensor: $HF_OUTPUT_NORM")
        val normTensor = loadAndConvertTensor<T>(ctx, dtype, reader, norm, transpose = false)
        tensorsByGgufName[Gemma4TensorNames.OUTPUT_NORM] = normTensor

        // Output weight (weight tying - reuse embed_tokens)
        tensorsByGgufName[Gemma4TensorNames.OUTPUT_WEIGHT] = embedTensor

        // Optional PLE global tensors. HF names changed in Gemma 4 vs the
        // earlier "per_layer_*" convention used by Gemma 3n / GGUF; map both
        // so synthetic GGUF-style fixtures keep working alongside real HF
        // checkpoints.
        //
        // The token-embeddings-per-layer tensor on real Gemma-4 E2B is
        // [vocab_size, num_layers, per_layer_dim] in BF16 — 4.7 GB raw, well
        // over the 2 GB JVM ByteArray limit our reader uses. The GGUF path
        // sidesteps this by keeping the table Q6_K-packed (~1.8 GB) with a
        // dedicated row-dequant data type. We don't have an equivalent for
        // SafeTensors yet, so when the BF16 source is too large we skip the
        // load — leaving `per_layer_token_embd.weight` absent from the
        // tensor map, which auto-disables PLE in `GemmaNetworkLoader`. The
        // model still runs (sandwich norms, layer_output_scale and softcap
        // are intact) but without the per-layer side-channel signal. PLE
        // support on the SafeTensors path is tracked separately.
        loadFirstExistingIfFits(
            ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            ggufName = Gemma4TensorNames.PER_LAYER_TOKEN_EMBD,
            hfCandidates = listOf(HF_EMBED_TOKENS_PER_LAYER, HF_PER_LAYER_TOKEN_EMBD),
            transpose = false,
        )
        loadFirstExisting(
            ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            ggufName = Gemma4TensorNames.PER_LAYER_MODEL_PROJ,
            hfCandidates = listOf(HF_PER_LAYER_MODEL_PROJECTION, HF_PER_LAYER_MODEL_PROJ),
            transpose = false,
        )
        loadFirstExisting(
            ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            ggufName = Gemma4TensorNames.PER_LAYER_PROJ_NORM,
            hfCandidates = listOf(HF_PER_LAYER_PROJECTION_NORM, HF_PER_LAYER_PROJ_NORM),
            transpose = false,
        )
    }

    private fun <T : DType> loadLayerTensors(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: StreamingShardedSafeTensorsReader,
        tensorsByHfName: Map<String, ShardedTensorInfo>,
        tensorsByGgufName: MutableMap<String, Tensor<T, Float>>,
        layer: Int
    ) {
        // Pre-attention input layernorm (HF: input_layernorm; GGUF slot: attn_norm).
        loadTensorIfExists(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfInputLayernorm(layer), Gemma4TensorNames.inputLayernorm(layer), transpose = false)

        // Attention projections.
        loadTensorIfExists(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfAttnQ(layer), Gemma4TensorNames.attnQ(layer))
        loadTensorIfExists(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfAttnK(layer), Gemma4TensorNames.attnK(layer))
        loadTensorIfExists(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfAttnV(layer), Gemma4TensorNames.attnV(layer))
        loadTensorIfExists(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfAttnO(layer), Gemma4TensorNames.attnOut(layer))

        // Sandwich norms — Gemma 4 has FOUR norms per block:
        //  - attn_norm           ← input_layernorm (already loaded above)
        //  - post_attention_norm ← post_attention_layernorm    (Gemma-4 only)
        //  - ffn_norm            ← pre_feedforward_layernorm   (Gemma-4 only)
        //  - post_ffw_norm       ← post_feedforward_layernorm  (Gemma-4 only)
        // The GGUF naming on `Gemma4TensorNames.postAttentionLayernorm` is a
        // legacy alias for the pre-FFN norm slot (`ffn_norm`); the proper
        // Gemma-4 source for that slot is HF `pre_feedforward_layernorm`.
        loadTensorIfExists(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfPostAttnLayernorm(layer), Gemma4TensorNames.postAttentionNorm(layer), transpose = false)
        loadTensorIfExists(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfPreFfwLayernorm(layer), Gemma4TensorNames.postAttentionLayernorm(layer), transpose = false)
        loadTensorIfExists(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfPostFfwLayernorm(layer), Gemma4TensorNames.postFfwNorm(layer), transpose = false)

        // MLP weights.
        loadTensorIfExists(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfMlpGate(layer), Gemma4TensorNames.ffnGate(layer))
        loadTensorIfExists(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfMlpUp(layer), Gemma4TensorNames.ffnUp(layer))
        loadTensorIfExists(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfMlpDown(layer), Gemma4TensorNames.ffnDown(layer))

        // Per-layer scalar (HF: `layer_scalar`, no `.weight` suffix; scalar shape).
        loadTensorIfExists(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfLayerScalar(layer), Gemma4TensorNames.layerOutputScale(layer), transpose = false)

        // Optional Per-Layer Embedding (PLE) per-layer tensors.
        // The HF projection sends [B,S,perLayerDim] back into the residual at
        // hidden_size, so it goes into the `proj` slot, NOT `per_layer_input`.
        loadTensorIfExists(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfPerLayerInputGate(layer), Gemma4TensorNames.pleInpGate(layer))
        loadTensorIfExists(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfPerLayerProjection(layer), Gemma4TensorNames.pleProj(layer))
        loadTensorIfExists(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfPostPerLayerInputNorm(layer), Gemma4TensorNames.plePostNorm(layer), transpose = false)

        // Optional QK normalization (per-head Q/K RMSNorm).
        loadTensorIfExists(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfAttnQNorm(layer), Gemma4TensorNames.attnQNorm(layer), transpose = false)
        loadTensorIfExists(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfAttnKNorm(layer), Gemma4TensorNames.attnKNorm(layer), transpose = false)
    }

    private fun <T : DType> loadTensorIfExists(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: StreamingShardedSafeTensorsReader,
        tensorsByHfName: Map<String, ShardedTensorInfo>,
        tensorsByGgufName: MutableMap<String, Tensor<T, Float>>,
        hfName: String,
        ggufName: String,
        transpose: Boolean = false
    ) {
        val info = tensorsByHfName[hfName]
        if (info != null) {
            val tensor = loadAndConvertTensor<T>(ctx, dtype, reader, info, transpose)
            tensorsByGgufName[ggufName] = tensor
        }
    }

    private fun <T : DType> loadFirstExisting(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: StreamingShardedSafeTensorsReader,
        tensorsByHfName: Map<String, ShardedTensorInfo>,
        tensorsByGgufName: MutableMap<String, Tensor<T, Float>>,
        ggufName: String,
        hfCandidates: List<String>,
        transpose: Boolean = false
    ) {
        for (hfName in hfCandidates) {
            val info = tensorsByHfName[hfName] ?: continue
            tensorsByGgufName[ggufName] = loadAndConvertTensor<T>(ctx, dtype, reader, info, transpose)
            return
        }
    }

    /**
     * Variant of [loadFirstExisting] that silently skips the load if the
     * raw on-disk tensor exceeds [MAX_BYTES_PER_TENSOR] (the JVM ByteArray
     * limit our streaming reader is bound by). Used for PLE globals on
     * real Gemma-4 SafeTensors checkpoints, where the BF16 source is too
     * large for the eager reader path.
     */
    private fun <T : DType> loadFirstExistingIfFits(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: StreamingShardedSafeTensorsReader,
        tensorsByHfName: Map<String, ShardedTensorInfo>,
        tensorsByGgufName: MutableMap<String, Tensor<T, Float>>,
        ggufName: String,
        hfCandidates: List<String>,
        transpose: Boolean = false
    ) {
        for (hfName in hfCandidates) {
            val info = tensorsByHfName[hfName] ?: continue
            if (info.sizeInBytes > MAX_BYTES_PER_TENSOR) {
                // Leave the slot empty so PLE auto-detection turns off cleanly.
                return
            }
            tensorsByGgufName[ggufName] = loadAndConvertTensor<T>(ctx, dtype, reader, info, transpose)
            return
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun <T : DType> loadAndConvertTensor(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: StreamingShardedSafeTensorsReader,
        info: ShardedTensorInfo,
        transpose: Boolean = false
    ): Tensor<T, Float> {
        val bytes = reader.loadTensorData(info)
        val shape = Shape(*info.shape.map { it.toInt() }.toIntArray())

        val floats = when (info.dtype.uppercase()) {
            "BF16" -> DequantOps.dequantBF16FromBytes(bytes)
            "F16" -> DequantOps.dequantF16FromBytes(bytes)
            "F32" -> DequantOps.bytesToFloatArray(bytes)
            else -> error("Unsupported SafeTensors dtype: ${info.dtype}")
        }

        return if (transpose && shape.rank == 2) {
            val rows = shape[0]
            val cols = shape[1]
            val transposed = transposeRowMajor(floats, rows, cols)
            val newShape = Shape(cols, rows)
            ctx.fromFloatArray<T, Float>(newShape, dtype, transposed) as Tensor<T, Float>
        } else {
            ctx.fromFloatArray<T, Float>(shape, dtype, floats) as Tensor<T, Float>
        }
    }

    private fun transposeRowMajor(data: FloatArray, rows: Int, cols: Int): FloatArray {
        val out = FloatArray(data.size)
        for (r in 0 until rows) {
            for (c in 0 until cols) {
                out[c * rows + r] = data[r * cols + c]
            }
        }
        return out
    }

    private companion object {
        // The streaming sharded reader returns each tensor as a single
        // ByteArray, which the JVM caps at Int.MAX_VALUE bytes. Keep a
        // little headroom for safety.
        const val MAX_BYTES_PER_TENSOR: Long = Int.MAX_VALUE.toLong() - 1024L

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
public suspend fun <T : DType> loadGemma4RuntimeWeightsFromSafeTensors(
    ctx: ExecutionContext,
    indexPath: String,
    dtype: KClass<T>
): Gemma4RuntimeWeights<T> {
    val loader = Gemma4SafeTensorsWeightLoader(indexPath)
    val loaded = loader.loadToMap<T>(ctx, dtype)
    return Gemma4WeightMapper.map(loaded)
}

/** Backward-compatible overload defaulting to FP32. */
public suspend fun loadGemma4RuntimeWeightsFromSafeTensors(
    ctx: ExecutionContext,
    indexPath: String
): Gemma4RuntimeWeights<FP32> = loadGemma4RuntimeWeightsFromSafeTensors(ctx, indexPath, FP32::class)
