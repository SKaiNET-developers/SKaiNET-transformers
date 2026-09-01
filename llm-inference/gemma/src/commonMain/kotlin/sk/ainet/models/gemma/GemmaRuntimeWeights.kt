package sk.ainet.models.gemma

import kotlinx.io.Source
import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import kotlin.reflect.KClass

/**
 * Weights for a single Gemma 4 transformer layer.
 *
 * Simpler than Gemma 3n: no AltUp, no activation sparsity, no Laurel.
 * Supports per-layer varying head dimensions via global_head_dim.
 */
public data class GemmaLayerWeights<T : DType>(
    val inputLayernorm: Tensor<T, Float>,
    val wq: Tensor<T, Float>,
    val wk: Tensor<T, Float>,
    val wv: Tensor<T, Float>,
    val wo: Tensor<T, Float>,
    val postAttentionLayernorm: Tensor<T, Float>,
    val gateProj: Tensor<T, Float>,
    val upProj: Tensor<T, Float>,
    val downProj: Tensor<T, Float>,
    /** Optional per-layer input embedding (PLE) */
    val perLayerInput: Tensor<T, Float>? = null,
    /** Optional per-layer output embedding (PLE) */
    val perLayerOutput: Tensor<T, Float>? = null,
    /** Optional QK normalization weights */
    val attnQNorm: Tensor<T, Float>? = null,
    val attnKNorm: Tensor<T, Float>? = null
)

/**
 * Complete runtime weights for Gemma 4 model.
 */
public data class GemmaRuntimeWeights<T : DType>(
    val metadata: GemmaModelMetadata,
    val tokenEmbedding: Tensor<T, Float>,
    val ropeFreqReal: Tensor<T, Float>?,
    val ropeFreqImag: Tensor<T, Float>?,
    val layers: List<GemmaLayerWeights<T>>,
    val finalNorm: Tensor<T, Float>,
    val lmHead: Tensor<T, Float>,
    /** Per-layer token embedding [perLayerDim * blockCount, vocabSize] (optional PLE) */
    val perLayerTokenEmbedding: Tensor<T, Float>? = null,
    /** Per-layer model projection [hiddenSize, perLayerDim * blockCount] (optional PLE) */
    val perLayerModelProj: Tensor<T, Float>? = null,
    /** Per-layer projection norm [perLayerDim] (optional PLE) */
    val perLayerProjNorm: Tensor<T, Float>? = null
)

/**
 * Raw weights loaded from GGUF, before mapping to runtime structure.
 *
 * Every tensor carries its true logical shape — quantized weights arrive
 * from the engine loader as packed block tensor data with `[out, in]`
 * shapes, dense tensors as plain FP32.
 */
public data class GemmaWeights<T : DType, V>(
    val metadata: GemmaModelMetadata,
    val tensors: Map<String, Tensor<T, V>>
)

/**
 * Tensor name constants for Gemma 4 GGUF format.
 */
public object GemmaTensorNames {
    public const val TOKEN_EMBEDDINGS: String = "token_embd.weight"
    public const val OUTPUT_NORM: String = "output_norm.weight"
    public const val OUTPUT_WEIGHT: String = "output.weight"
    public const val ROPE_FREQS_REAL: String = "rope.freq_cis_real"
    public const val ROPE_FREQS_IMAG: String = "rope.freq_cis_imag"

    public fun inputLayernorm(layer: Int): String = "blk.$layer.attn_norm.weight"
    public fun attnQ(layer: Int): String = "blk.$layer.attn_q.weight"
    public fun attnK(layer: Int): String = "blk.$layer.attn_k.weight"
    public fun attnV(layer: Int): String = "blk.$layer.attn_v.weight"
    public fun attnOut(layer: Int): String = "blk.$layer.attn_output.weight"
    public fun postAttentionLayernorm(layer: Int): String = "blk.$layer.ffn_norm.weight"
    public fun ffnGate(layer: Int): String = "blk.$layer.ffn_gate.weight"
    public fun ffnDown(layer: Int): String = "blk.$layer.ffn_down.weight"
    public fun ffnUp(layer: Int): String = "blk.$layer.ffn_up.weight"
    public fun perLayerInput(layer: Int): String = "blk.$layer.per_layer_input.weight"
    public fun perLayerOutput(layer: Int): String = "blk.$layer.per_layer_output.weight"
    public fun attnQNorm(layer: Int): String = "blk.$layer.attn_q_norm.weight"
    public fun attnKNorm(layer: Int): String = "blk.$layer.attn_k_norm.weight"
    public fun postAttentionNorm(layer: Int): String = "blk.$layer.post_attention_norm.weight"
    public fun postFfwNorm(layer: Int): String = "blk.$layer.post_ffw_norm.weight"
    public fun layerOutputScale(layer: Int): String = "blk.$layer.layer_output_scale.weight"

    // Gemma 4 Per-Layer Embedding block-level tensors.
    public fun pleInpGate(layer: Int): String = "blk.$layer.inp_gate.weight"
    public fun plePostNorm(layer: Int): String = "blk.$layer.post_norm.weight"
    public fun pleProj(layer: Int): String = "blk.$layer.proj.weight"

    // PLE global tensors
    public const val PER_LAYER_TOKEN_EMBD: String = "per_layer_token_embd.weight"
    public const val PER_LAYER_MODEL_PROJ: String = "per_layer_model_proj.weight"
    public const val PER_LAYER_PROJ_NORM: String = "per_layer_proj_norm.weight"
}

/**
 * Maps raw weights to runtime structure with shape validation.
 */
public object GemmaWeightMapper {

    public fun <T : DType> map(weights: GemmaWeights<T, Float>): GemmaRuntimeWeights<T> {
        val metadata = weights.metadata

        fun get(name: String): Tensor<T, Float> =
            weights.tensors[name] ?: error("Missing tensor: $name")

        fun getOptional(name: String): Tensor<T, Float>? = weights.tensors[name]

        // Shape checks run for every tensor: packed engine-loader tensors
        // carry their true logical [out, in] shapes, same as dense ones.
        fun Tensor<*, *>.requireShape(expected: Shape, label: String, tensorName: String) {
            if (shape != expected) {
                error("$label expected shape $expected but was $shape")
            }
        }

        fun Tensor<*, *>.require2D(rows: Int, cols: Int, label: String, tensorName: String) =
            requireShape(Shape(rows, cols), label, tensorName)

        fun Tensor<*, *>.require1D(size: Int, label: String, tensorName: String) =
            requireShape(Shape(size), label, tensorName)

        val tokenEmbedding = get(GemmaTensorNames.TOKEN_EMBEDDINGS)
        tokenEmbedding.require2D(
            metadata.vocabSize,
            metadata.embeddingLength,
            "token embedding",
            GemmaTensorNames.TOKEN_EMBEDDINGS
        )

        val finalNorm = get(GemmaTensorNames.OUTPUT_NORM)
        finalNorm.require1D(
            metadata.embeddingLength,
            "output norm",
            GemmaTensorNames.OUTPUT_NORM
        )

        val lmHead = get(GemmaTensorNames.OUTPUT_WEIGHT)
        lmHead.require2D(
            metadata.vocabSize,
            metadata.embeddingLength,
            "lm head",
            GemmaTensorNames.OUTPUT_WEIGHT
        )

        val ropeReal = getOptional(GemmaTensorNames.ROPE_FREQS_REAL)
        val ropeImag = getOptional(GemmaTensorNames.ROPE_FREQS_IMAG)

        val layers = (0 until metadata.blockCount).map { layer ->
            val layerHeadDim = metadata.getHeadDim(layer)
            val qDim = metadata.headCount * layerHeadDim
            val kvDim = metadata.kvHeadCount * layerHeadDim
            val ffnDim = metadata.getIntermediateSize(layer)

            val inputLayernorm = get(GemmaTensorNames.inputLayernorm(layer)).apply {
                require1D(metadata.embeddingLength, "blk.$layer.attn_norm", GemmaTensorNames.inputLayernorm(layer))
            }
            val wq = get(GemmaTensorNames.attnQ(layer)).apply {
                require2D(qDim, metadata.embeddingLength, "blk.$layer.attn_q", GemmaTensorNames.attnQ(layer))
            }
            val wk = get(GemmaTensorNames.attnK(layer)).apply {
                require2D(kvDim, metadata.embeddingLength, "blk.$layer.attn_k", GemmaTensorNames.attnK(layer))
            }
            val wv = get(GemmaTensorNames.attnV(layer)).apply {
                require2D(kvDim, metadata.embeddingLength, "blk.$layer.attn_v", GemmaTensorNames.attnV(layer))
            }
            val wo = get(GemmaTensorNames.attnOut(layer)).apply {
                require2D(metadata.embeddingLength, qDim, "blk.$layer.attn_output", GemmaTensorNames.attnOut(layer))
            }
            val postAttentionLayernorm = get(GemmaTensorNames.postAttentionLayernorm(layer)).apply {
                require1D(metadata.embeddingLength, "blk.$layer.ffn_norm", GemmaTensorNames.postAttentionLayernorm(layer))
            }
            val gateProj = get(GemmaTensorNames.ffnGate(layer)).apply {
                require2D(ffnDim, metadata.embeddingLength, "blk.$layer.ffn_gate", GemmaTensorNames.ffnGate(layer))
            }
            val downProj = get(GemmaTensorNames.ffnDown(layer)).apply {
                require2D(metadata.embeddingLength, ffnDim, "blk.$layer.ffn_down", GemmaTensorNames.ffnDown(layer))
            }
            val upProj = get(GemmaTensorNames.ffnUp(layer)).apply {
                require2D(ffnDim, metadata.embeddingLength, "blk.$layer.ffn_up", GemmaTensorNames.ffnUp(layer))
            }

            GemmaLayerWeights(
                inputLayernorm = inputLayernorm,
                wq = wq,
                wk = wk,
                wv = wv,
                wo = wo,
                postAttentionLayernorm = postAttentionLayernorm,
                gateProj = gateProj,
                upProj = upProj,
                downProj = downProj,
                perLayerInput = getOptional(GemmaTensorNames.perLayerInput(layer)),
                perLayerOutput = getOptional(GemmaTensorNames.perLayerOutput(layer)),
                attnQNorm = getOptional(GemmaTensorNames.attnQNorm(layer)),
                attnKNorm = getOptional(GemmaTensorNames.attnKNorm(layer))
            )
        }

        return GemmaRuntimeWeights(
            metadata = metadata,
            tokenEmbedding = tokenEmbedding,
            ropeFreqReal = ropeReal,
            ropeFreqImag = ropeImag,
            layers = layers,
            finalNorm = finalNorm,
            lmHead = lmHead,
            perLayerTokenEmbedding = getOptional(GemmaTensorNames.PER_LAYER_TOKEN_EMBD),
            perLayerModelProj = getOptional(GemmaTensorNames.PER_LAYER_MODEL_PROJ),
            perLayerProjNorm = getOptional(GemmaTensorNames.PER_LAYER_PROJ_NORM)
        )
    }
}

/**
 * Convenience loader: reads weights from GGUF source (sequential — always
 * dequantized to dense floats), maps them into runtime structure.
 */
public suspend fun <T : DType> loadGemmaRuntimeWeights(
    ctx: ExecutionContext,
    sourceProvider: () -> Source,
    dtype: KClass<T>
): GemmaRuntimeWeights<T> {
    val loader = GemmaWeightLoader(sourceProvider = sourceProvider)
    val loaded = loader.loadToMap<T, Float>(ctx, dtype)
    return GemmaWeightMapper.map(loaded)
}

/** Backward-compatible overload defaulting to FP32. */
public suspend fun loadGemmaRuntimeWeights(
    ctx: ExecutionContext,
    sourceProvider: () -> Source
): GemmaRuntimeWeights<FP32> = loadGemmaRuntimeWeights(ctx, sourceProvider, FP32::class)

// ============== Streaming API (for large files >2GB) ==============

/**
 * Load Gemma 4 runtime weights via the engine loader. Default form keeps
 * quantized tensors packed; pass [GEMMA_DEQUANTIZE_ALL] for dense FP32.
 */
@ExperimentalMemoryApi
public suspend fun <T : DType> loadGemmaRuntimeWeightsStreaming(
    ctx: ExecutionContext,
    randomAccessProvider: () -> RandomAccessSource,
    dtype: KClass<T>,
    weightForm: WeightForm? = null
): GemmaRuntimeWeights<T> {
    val loader = GemmaWeightLoader(
        randomAccessProvider = randomAccessProvider,
        weightForm = weightForm
    )
    val loaded = loader.loadToMapStreaming<T, Float>(ctx, dtype)
    return GemmaWeightMapper.map(loaded)
}

/** Backward-compatible overload defaulting to FP32. */
@ExperimentalMemoryApi
public suspend fun loadGemmaRuntimeWeightsStreaming(
    ctx: ExecutionContext,
    randomAccessProvider: () -> RandomAccessSource,
    weightForm: WeightForm? = null
): GemmaRuntimeWeights<FP32> = loadGemmaRuntimeWeightsStreaming(ctx, randomAccessProvider, FP32::class, weightForm)
