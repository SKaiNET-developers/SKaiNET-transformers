package sk.ainet.models.gemma

import kotlinx.io.Source
import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.model.QuantPolicy
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
public data class Gemma4LayerWeights<T : DType>(
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
public data class Gemma4RuntimeWeights<T : DType>(
    val metadata: Gemma4ModelMetadata,
    val tokenEmbedding: Tensor<T, Float>,
    val ropeFreqReal: Tensor<T, Float>?,
    val ropeFreqImag: Tensor<T, Float>?,
    val layers: List<Gemma4LayerWeights<T>>,
    val finalNorm: Tensor<T, Float>,
    val lmHead: Tensor<T, Float>,
    val quantTypes: Map<String, GGMLQuantizationType> = emptyMap(),
    /** Per-layer token embedding [perLayerDim * blockCount, vocabSize] (optional PLE) */
    val perLayerTokenEmbedding: Tensor<T, Float>? = null,
    /** Per-layer model projection [hiddenSize, perLayerDim * blockCount] (optional PLE) */
    val perLayerModelProj: Tensor<T, Float>? = null,
    /** Per-layer projection norm [perLayerDim] (optional PLE) */
    val perLayerProjNorm: Tensor<T, Float>? = null
)

/**
 * Raw weights loaded from GGUF, before mapping to runtime structure.
 */
public data class Gemma4Weights<T : DType, V>(
    val metadata: Gemma4ModelMetadata,
    val tensors: Map<String, Tensor<T, V>>,
    val quantTypes: Map<String, GGMLQuantizationType> = emptyMap(),
    /**
     * Original 2-D logical shape of each tensor as reported by GGUF.
     *
     * Populated by the loader for `NATIVE_OPTIMIZED` quant policy, where
     * quantized tensors are stored with a 1-D byte shape (the packed
     * representation doesn't satisfy `byte_count == logical_volume`).
     * Downstream converters (e.g. `GemmaMemSegConverter`) need the real
     * `[rows, cols]` shape to build packed tensor-data variants, since
     * model metadata alone is unreliable for Gemma 4 — `feed_forward_length`
     * is a single scalar while real FFN widths vary across layers
     * (E2B: 6144 for blk.0–14, 12288 for blk.15+).
     */
    val logicalShapes: Map<String, sk.ainet.lang.tensor.Shape> = emptyMap()
)

/**
 * Tensor name constants for Gemma 4 GGUF format.
 */
public object Gemma4TensorNames {
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

    // PLE global tensors
    public const val PER_LAYER_TOKEN_EMBD: String = "per_layer_token_embd.weight"
    public const val PER_LAYER_MODEL_PROJ: String = "per_layer_model_proj.weight"
    public const val PER_LAYER_PROJ_NORM: String = "per_layer_proj_norm.weight"
}

/**
 * Maps raw weights to runtime structure with shape validation.
 */
public object Gemma4WeightMapper {

    public fun <T : DType> map(weights: Gemma4Weights<T, Float>): Gemma4RuntimeWeights<T> {
        val metadata = weights.metadata

        fun get(name: String): Tensor<T, Float> =
            weights.tensors[name] ?: error("Missing tensor: $name")

        fun getOptional(name: String): Tensor<T, Float>? = weights.tensors[name]

        fun isQuant(name: String): Boolean = weights.quantTypes[name] != null

        fun Tensor<*, *>.requireShape(expected: Shape, label: String, tensorName: String) {
            if (isQuant(tensorName)) return
            if (shape != expected) {
                error("$label expected shape $expected but was $shape")
            }
        }

        fun Tensor<*, *>.require2D(rows: Int, cols: Int, label: String, tensorName: String) =
            requireShape(Shape(rows, cols), label, tensorName)

        fun Tensor<*, *>.require1D(size: Int, label: String, tensorName: String) =
            requireShape(Shape(size), label, tensorName)

        val tokenEmbedding = get(Gemma4TensorNames.TOKEN_EMBEDDINGS)
        tokenEmbedding.require2D(
            metadata.vocabSize,
            metadata.embeddingLength,
            "token embedding",
            Gemma4TensorNames.TOKEN_EMBEDDINGS
        )

        val finalNorm = get(Gemma4TensorNames.OUTPUT_NORM)
        finalNorm.require1D(
            metadata.embeddingLength,
            "output norm",
            Gemma4TensorNames.OUTPUT_NORM
        )

        val lmHead = get(Gemma4TensorNames.OUTPUT_WEIGHT)
        lmHead.require2D(
            metadata.vocabSize,
            metadata.embeddingLength,
            "lm head",
            Gemma4TensorNames.OUTPUT_WEIGHT
        )

        val ropeReal = getOptional(Gemma4TensorNames.ROPE_FREQS_REAL)
        val ropeImag = getOptional(Gemma4TensorNames.ROPE_FREQS_IMAG)

        val layers = (0 until metadata.blockCount).map { layer ->
            val layerHeadDim = metadata.getHeadDim(layer)
            val qDim = metadata.headCount * layerHeadDim
            val kvDim = metadata.kvHeadCount * layerHeadDim
            val ffnDim = metadata.getIntermediateSize(layer)

            val inputLayernorm = get(Gemma4TensorNames.inputLayernorm(layer)).apply {
                require1D(metadata.embeddingLength, "blk.$layer.attn_norm", Gemma4TensorNames.inputLayernorm(layer))
            }
            val wq = get(Gemma4TensorNames.attnQ(layer)).apply {
                require2D(qDim, metadata.embeddingLength, "blk.$layer.attn_q", Gemma4TensorNames.attnQ(layer))
            }
            val wk = get(Gemma4TensorNames.attnK(layer)).apply {
                require2D(kvDim, metadata.embeddingLength, "blk.$layer.attn_k", Gemma4TensorNames.attnK(layer))
            }
            val wv = get(Gemma4TensorNames.attnV(layer)).apply {
                require2D(kvDim, metadata.embeddingLength, "blk.$layer.attn_v", Gemma4TensorNames.attnV(layer))
            }
            val wo = get(Gemma4TensorNames.attnOut(layer)).apply {
                require2D(metadata.embeddingLength, qDim, "blk.$layer.attn_output", Gemma4TensorNames.attnOut(layer))
            }
            val postAttentionLayernorm = get(Gemma4TensorNames.postAttentionLayernorm(layer)).apply {
                require1D(metadata.embeddingLength, "blk.$layer.ffn_norm", Gemma4TensorNames.postAttentionLayernorm(layer))
            }
            val gateProj = get(Gemma4TensorNames.ffnGate(layer)).apply {
                require2D(ffnDim, metadata.embeddingLength, "blk.$layer.ffn_gate", Gemma4TensorNames.ffnGate(layer))
            }
            val downProj = get(Gemma4TensorNames.ffnDown(layer)).apply {
                require2D(metadata.embeddingLength, ffnDim, "blk.$layer.ffn_down", Gemma4TensorNames.ffnDown(layer))
            }
            val upProj = get(Gemma4TensorNames.ffnUp(layer)).apply {
                require2D(ffnDim, metadata.embeddingLength, "blk.$layer.ffn_up", Gemma4TensorNames.ffnUp(layer))
            }

            Gemma4LayerWeights(
                inputLayernorm = inputLayernorm,
                wq = wq,
                wk = wk,
                wv = wv,
                wo = wo,
                postAttentionLayernorm = postAttentionLayernorm,
                gateProj = gateProj,
                upProj = upProj,
                downProj = downProj,
                perLayerInput = getOptional(Gemma4TensorNames.perLayerInput(layer)),
                perLayerOutput = getOptional(Gemma4TensorNames.perLayerOutput(layer)),
                attnQNorm = getOptional(Gemma4TensorNames.attnQNorm(layer)),
                attnKNorm = getOptional(Gemma4TensorNames.attnKNorm(layer))
            )
        }

        return Gemma4RuntimeWeights(
            metadata = metadata,
            tokenEmbedding = tokenEmbedding,
            ropeFreqReal = ropeReal,
            ropeFreqImag = ropeImag,
            layers = layers,
            finalNorm = finalNorm,
            lmHead = lmHead,
            quantTypes = weights.quantTypes,
            perLayerTokenEmbedding = getOptional(Gemma4TensorNames.PER_LAYER_TOKEN_EMBD),
            perLayerModelProj = getOptional(Gemma4TensorNames.PER_LAYER_MODEL_PROJ),
            perLayerProjNorm = getOptional(Gemma4TensorNames.PER_LAYER_PROJ_NORM)
        )
    }
}

/**
 * Convenience loader: reads weights from GGUF source, maps them into runtime structure.
 */
public suspend fun <T : DType> loadGemma4RuntimeWeights(
    ctx: ExecutionContext,
    sourceProvider: () -> Source,
    dtype: KClass<T>,
    quantPolicy: QuantPolicy = QuantPolicy.RAW_BYTES,
    allowQuantized: Boolean = false
): Gemma4RuntimeWeights<T> {
    val loader = Gemma4WeightLoader(
        sourceProvider = sourceProvider,
        quantPolicy = quantPolicy
    )
    val loaded = loader.loadToMap<T, Float>(ctx, dtype)
    if (!allowQuantized && loaded.quantTypes.isNotEmpty()) {
        error("Quantized weights detected (${loaded.quantTypes.size}). Pass allowQuantized=true to consume raw quant tensors.")
    }
    return Gemma4WeightMapper.map(loaded)
}

/** Backward-compatible overload defaulting to FP32. */
public suspend fun loadGemma4RuntimeWeights(
    ctx: ExecutionContext,
    sourceProvider: () -> Source,
    quantPolicy: QuantPolicy = QuantPolicy.RAW_BYTES,
    allowQuantized: Boolean = false
): Gemma4RuntimeWeights<FP32> = loadGemma4RuntimeWeights(ctx, sourceProvider, FP32::class, quantPolicy, allowQuantized)

/**
 * Load Gemma 4 runtime weights with dequantization.
 */
public suspend fun <T : DType> loadGemma4RuntimeWeightsDequantized(
    ctx: ExecutionContext,
    sourceProvider: () -> Source,
    dtype: KClass<T>
): Gemma4RuntimeWeights<T> {
    val loader = Gemma4WeightLoader(
        sourceProvider = sourceProvider,
        quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32
    )
    val loaded = loader.loadToMap<T, Float>(ctx, dtype)
    if (loaded.quantTypes.isNotEmpty()) {
        error("Unsupported quantized tensors remain after dequant attempt: ${loaded.quantTypes}")
    }
    return Gemma4WeightMapper.map(loaded)
}

/** Backward-compatible overload defaulting to FP32. */
public suspend fun loadGemma4RuntimeWeightsDequantized(
    ctx: ExecutionContext,
    sourceProvider: () -> Source
): Gemma4RuntimeWeights<FP32> = loadGemma4RuntimeWeightsDequantized(ctx, sourceProvider, FP32::class)

// ============== Streaming API (for large files >2GB) ==============

/**
 * Load Gemma 4 runtime weights using streaming API.
 */
public suspend fun <T : DType> loadGemma4RuntimeWeightsStreaming(
    ctx: ExecutionContext,
    randomAccessProvider: () -> RandomAccessSource,
    dtype: KClass<T>,
    quantPolicy: QuantPolicy = QuantPolicy.RAW_BYTES,
    allowQuantized: Boolean = false
): Gemma4RuntimeWeights<T> {
    val loader = Gemma4WeightLoader(
        randomAccessProvider = randomAccessProvider,
        quantPolicy = quantPolicy
    )
    val loaded = loader.loadToMapStreaming<T, Float>(ctx, dtype)
    if (!allowQuantized && loaded.quantTypes.isNotEmpty()) {
        error("Quantized weights detected (${loaded.quantTypes.size}). Pass allowQuantized=true to consume raw quant tensors.")
    }
    return Gemma4WeightMapper.map(loaded)
}

/** Backward-compatible overload defaulting to FP32. */
public suspend fun loadGemma4RuntimeWeightsStreaming(
    ctx: ExecutionContext,
    randomAccessProvider: () -> RandomAccessSource,
    quantPolicy: QuantPolicy = QuantPolicy.RAW_BYTES,
    allowQuantized: Boolean = false
): Gemma4RuntimeWeights<FP32> = loadGemma4RuntimeWeightsStreaming(ctx, randomAccessProvider, FP32::class, quantPolicy, allowQuantized)

/**
 * Load Gemma 4 runtime weights using streaming API with dequantization.
 */
public suspend fun <T : DType> loadGemma4RuntimeWeightsDequantizedStreaming(
    ctx: ExecutionContext,
    randomAccessProvider: () -> RandomAccessSource,
    dtype: KClass<T>
): Gemma4RuntimeWeights<T> {
    val loader = Gemma4WeightLoader(
        randomAccessProvider = randomAccessProvider,
        quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32
    )
    val loaded = loader.loadToMapStreaming<T, Float>(ctx, dtype)
    if (loaded.quantTypes.isNotEmpty()) {
        error("Unsupported quantized tensors remain after dequant attempt: ${loaded.quantTypes}")
    }
    return Gemma4WeightMapper.map(loaded)
}

/** Backward-compatible overload defaulting to FP32. */
public suspend fun loadGemma4RuntimeWeightsDequantizedStreaming(
    ctx: ExecutionContext,
    randomAccessProvider: () -> RandomAccessSource
): Gemma4RuntimeWeights<FP32> = loadGemma4RuntimeWeightsDequantizedStreaming(ctx, randomAccessProvider, FP32::class)
