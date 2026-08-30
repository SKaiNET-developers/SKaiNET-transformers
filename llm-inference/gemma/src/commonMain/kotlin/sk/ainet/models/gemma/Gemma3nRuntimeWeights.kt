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
 * Weights for a single Gemma 3n transformer layer.
 *
 * Key differences from LLaMA:
 * - Variable FFN dimensions per layer (MatFormer architecture)
 * - Per-layer embeddings (perLayerInput/Output) are optional
 */
public data class Gemma3nLayerWeights<T : DType>(
    val inputLayernorm: Tensor<T, Float>,
    val wq: Tensor<T, Float>,
    val wk: Tensor<T, Float>,
    val wv: Tensor<T, Float>,
    val wo: Tensor<T, Float>,
    val postAttentionLayernorm: Tensor<T, Float>,
    val gateProj: Tensor<T, Float>,
    val upProj: Tensor<T, Float>,
    val downProj: Tensor<T, Float>,
    /** Optional per-layer input embedding */
    val perLayerInput: Tensor<T, Float>?,
    /** Optional per-layer output embedding */
    val perLayerOutput: Tensor<T, Float>?,
    // ---- E4B additional per-layer weights ----
    /** Per-layer AltUp weights (E4B only) */
    val altUpLayerWeights: AltUpLayerWeights<T>? = null,
    /** QK normalization weights (E4B) */
    val attnQNorm: Tensor<T, Float>? = null,
    val attnKNorm: Tensor<T, Float>? = null,
    /** Post-attention norm (E4B: blk.N.post_attention_norm) */
    val postAttentionNorm: Tensor<T, Float>? = null,
    /** Post-FFN norm (E4B: blk.N.post_ffw_norm) */
    val postFfwNorm: Tensor<T, Float>? = null,
    /** Post norm (E4B: blk.N.post_norm) */
    val postNorm: Tensor<T, Float>? = null,
    /** Input gate for per-layer embeddings (E4B: blk.N.inp_gate) */
    val inputGate: Tensor<T, Float>? = null,
    /** Per-layer projection (E4B: blk.N.proj) */
    val proj: Tensor<T, Float>? = null,
    /** Laurel low-rank left (E4B: blk.N.laurel_l) */
    val laurelL: Tensor<T, Float>? = null,
    /** Laurel low-rank right (E4B: blk.N.laurel_r) */
    val laurelR: Tensor<T, Float>? = null,
    /** Laurel post norm (E4B: blk.N.laurel_post_norm) */
    val laurelPostNorm: Tensor<T, Float>? = null
)

/**
 * Complete runtime weights for Gemma 3n model.
 */
public data class Gemma3nRuntimeWeights<T : DType>(
    val metadata: Gemma3nModelMetadata,
    val tokenEmbedding: Tensor<T, Float>,
    val ropeFreqReal: Tensor<T, Float>?,
    val ropeFreqImag: Tensor<T, Float>?,
    val layers: List<Gemma3nLayerWeights<T>>,
    val finalNorm: Tensor<T, Float>,
    val lmHead: Tensor<T, Float>,
    /** Global AltUp weights, present only for E4B models (numAltupInputs > 1). */
    val altUpGlobalWeights: AltUpGlobalWeights<T>? = null,
    // ---- E4B global per-layer embedding tensors ----
    /** Per-layer token embedding [perLayerDim * blockCount, vocabSize] */
    val perLayerTokenEmbedding: Tensor<T, Float>? = null,
    /** Per-layer model projection [hiddenSize, perLayerDim * blockCount] */
    val perLayerModelProj: Tensor<T, Float>? = null,
    /** Per-layer projection norm [perLayerDim] */
    val perLayerProjNorm: Tensor<T, Float>? = null
)

/**
 * Raw weights loaded from GGUF, before mapping to runtime structure.
 */
public data class Gemma3nWeights<T : DType, V>(
    val metadata: Gemma3nModelMetadata,
    val tensors: Map<String, Tensor<T, V>>
)

/**
 * Tensor name constants for Gemma 3n GGUF format.
 */
public object Gemma3nTensorNames {
    public const val TOKEN_EMBEDDINGS: String = "token_embd.weight"
    public const val OUTPUT_NORM: String = "output_norm.weight"
    public const val OUTPUT_WEIGHT: String = "output.weight"
    public const val ROPE_FREQS_REAL: String = "rope.freq_cis_real"
    public const val ROPE_FREQS_IMAG: String = "rope.freq_cis_imag"

    // Standard per-layer tensors
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

    // E4B per-layer AltUp tensors
    public fun altupPredictCoef(layer: Int): String = "blk.$layer.altup_predict_coef.weight"
    public fun altupCorrectCoef(layer: Int): String = "blk.$layer.altup_correct_coef.weight"
    public fun altupCorrectScale(layer: Int): String = "blk.$layer.altup_correct_scale.weight"
    public fun altupRouter(layer: Int): String = "blk.$layer.altup_router.weight"
    public fun altupRouterNorm(layer: Int): String = "blk.$layer.altup_router_norm.weight"

    // E4B global AltUp tensors
    public const val ALTUP_PROJ: String = "altup_proj.weight"
    public const val ALTUP_UNEMBD_PROJ: String = "altup_unembd_proj.weight"

    // E4B additional per-layer tensors
    public fun attnQNorm(layer: Int): String = "blk.$layer.attn_q_norm.weight"
    public fun attnKNorm(layer: Int): String = "blk.$layer.attn_k_norm.weight"
    public fun postAttentionNorm(layer: Int): String = "blk.$layer.post_attention_norm.weight"
    public fun postFfwNorm(layer: Int): String = "blk.$layer.post_ffw_norm.weight"
    public fun postNorm(layer: Int): String = "blk.$layer.post_norm.weight"
    public fun inputGate(layer: Int): String = "blk.$layer.inp_gate.weight"
    public fun proj(layer: Int): String = "blk.$layer.proj.weight"
    public fun laurelL(layer: Int): String = "blk.$layer.laurel_l.weight"
    public fun laurelR(layer: Int): String = "blk.$layer.laurel_r.weight"
    public fun laurelPostNorm(layer: Int): String = "blk.$layer.laurel_post_norm.weight"

    // E4B global per-layer embedding tensors
    public const val PER_LAYER_TOKEN_EMBD: String = "per_layer_token_embd.weight"
    public const val PER_LAYER_MODEL_PROJ: String = "per_layer_model_proj.weight"
    public const val PER_LAYER_PROJ_NORM: String = "per_layer_proj_norm.weight"
}

/**
 * Maps raw weights to runtime structure with shape validation.
 */
public object Gemma3nWeightMapper {

    public fun <T : DType> map(weights: Gemma3nWeights<T, Float>): Gemma3nRuntimeWeights<T> {
        val metadata = weights.metadata
        val headSize = metadata.headDim

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

        val tokenEmbedding = get(Gemma3nTensorNames.TOKEN_EMBEDDINGS)
        tokenEmbedding.require2D(
            metadata.vocabSize,
            metadata.embeddingLength,
            "token embedding",
            Gemma3nTensorNames.TOKEN_EMBEDDINGS
        )

        val finalNorm = get(Gemma3nTensorNames.OUTPUT_NORM)
        finalNorm.require1D(
            metadata.embeddingLength,
            "output norm",
            Gemma3nTensorNames.OUTPUT_NORM
        )

        val lmHead = get(Gemma3nTensorNames.OUTPUT_WEIGHT)
        lmHead.require2D(
            metadata.vocabSize,
            metadata.embeddingLength,
            "lm head",
            Gemma3nTensorNames.OUTPUT_WEIGHT
        )

        val ropeReal = getOptional(Gemma3nTensorNames.ROPE_FREQS_REAL)
        val ropeImag = getOptional(Gemma3nTensorNames.ROPE_FREQS_IMAG)

        val kvDim = metadata.kvHeadCount * headSize

        val layers = (0 until metadata.blockCount).map { layer ->
            val ffnDim = metadata.getFeedForwardLength(layer)

            val inputLayernorm = get(Gemma3nTensorNames.inputLayernorm(layer)).apply {
                require1D(metadata.embeddingLength, "blk.$layer.attn_norm", Gemma3nTensorNames.inputLayernorm(layer))
            }
            val wq = get(Gemma3nTensorNames.attnQ(layer)).apply {
                require2D(
                    metadata.headCount * headSize,
                    metadata.embeddingLength,
                    "blk.$layer.attn_q",
                    Gemma3nTensorNames.attnQ(layer)
                )
            }
            val wk = get(Gemma3nTensorNames.attnK(layer)).apply {
                require2D(kvDim, metadata.embeddingLength, "blk.$layer.attn_k", Gemma3nTensorNames.attnK(layer))
            }
            val wv = get(Gemma3nTensorNames.attnV(layer)).apply {
                require2D(kvDim, metadata.embeddingLength, "blk.$layer.attn_v", Gemma3nTensorNames.attnV(layer))
            }
            val wo = get(Gemma3nTensorNames.attnOut(layer)).apply {
                require2D(
                    metadata.embeddingLength,
                    metadata.headCount * headSize,
                    "blk.$layer.attn_output",
                    Gemma3nTensorNames.attnOut(layer)
                )
            }
            val postAttentionLayernorm = get(Gemma3nTensorNames.postAttentionLayernorm(layer)).apply {
                require1D(metadata.embeddingLength, "blk.$layer.ffn_norm", Gemma3nTensorNames.postAttentionLayernorm(layer))
            }
            val gateProj = get(Gemma3nTensorNames.ffnGate(layer)).apply {
                require2D(ffnDim, metadata.embeddingLength, "blk.$layer.ffn_gate", Gemma3nTensorNames.ffnGate(layer))
            }
            val downProj = get(Gemma3nTensorNames.ffnDown(layer)).apply {
                require2D(metadata.embeddingLength, ffnDim, "blk.$layer.ffn_down", Gemma3nTensorNames.ffnDown(layer))
            }
            val upProj = get(Gemma3nTensorNames.ffnUp(layer)).apply {
                require2D(ffnDim, metadata.embeddingLength, "blk.$layer.ffn_up", Gemma3nTensorNames.ffnUp(layer))
            }

            // Optional per-layer embeddings
            val perLayerInput = getOptional(Gemma3nTensorNames.perLayerInput(layer))
            val perLayerOutput = getOptional(Gemma3nTensorNames.perLayerOutput(layer))

            // Optional per-layer AltUp weights (E4B)
            val altUpLayerWeights = run {
                val predict = getOptional(Gemma3nTensorNames.altupPredictCoef(layer))
                val correct = getOptional(Gemma3nTensorNames.altupCorrectCoef(layer))
                val scale = getOptional(Gemma3nTensorNames.altupCorrectScale(layer))
                val router = getOptional(Gemma3nTensorNames.altupRouter(layer))
                val routerNorm = getOptional(Gemma3nTensorNames.altupRouterNorm(layer))
                if (predict != null && correct != null && scale != null && router != null && routerNorm != null) {
                    AltUpLayerWeights(predict, correct, scale, router, routerNorm)
                } else null
            }

            Gemma3nLayerWeights(
                inputLayernorm = inputLayernorm,
                wq = wq,
                wk = wk,
                wv = wv,
                wo = wo,
                postAttentionLayernorm = postAttentionLayernorm,
                gateProj = gateProj,
                upProj = upProj,
                downProj = downProj,
                perLayerInput = perLayerInput,
                perLayerOutput = perLayerOutput,
                altUpLayerWeights = altUpLayerWeights,
                attnQNorm = getOptional(Gemma3nTensorNames.attnQNorm(layer)),
                attnKNorm = getOptional(Gemma3nTensorNames.attnKNorm(layer)),
                postAttentionNorm = getOptional(Gemma3nTensorNames.postAttentionNorm(layer)),
                postFfwNorm = getOptional(Gemma3nTensorNames.postFfwNorm(layer)),
                postNorm = getOptional(Gemma3nTensorNames.postNorm(layer)),
                inputGate = getOptional(Gemma3nTensorNames.inputGate(layer)),
                proj = getOptional(Gemma3nTensorNames.proj(layer)),
                laurelL = getOptional(Gemma3nTensorNames.laurelL(layer)),
                laurelR = getOptional(Gemma3nTensorNames.laurelR(layer)),
                laurelPostNorm = getOptional(Gemma3nTensorNames.laurelPostNorm(layer))
            )
        }

        // Global AltUp weights (optional, present in E4B)
        val altUpGlobalWeights = run {
            val proj = getOptional(Gemma3nTensorNames.ALTUP_PROJ)
            val unembdProj = getOptional(Gemma3nTensorNames.ALTUP_UNEMBD_PROJ)
            if (proj != null && unembdProj != null) {
                AltUpGlobalWeights(projWeight = proj, unembdProjWeight = unembdProj)
            } else null
        }

        return Gemma3nRuntimeWeights(
            metadata = metadata,
            tokenEmbedding = tokenEmbedding,
            ropeFreqReal = ropeReal,
            ropeFreqImag = ropeImag,
            layers = layers,
            finalNorm = finalNorm,
            lmHead = lmHead,
            altUpGlobalWeights = altUpGlobalWeights,
            perLayerTokenEmbedding = getOptional(Gemma3nTensorNames.PER_LAYER_TOKEN_EMBD),
            perLayerModelProj = getOptional(Gemma3nTensorNames.PER_LAYER_MODEL_PROJ),
            perLayerProjNorm = getOptional(Gemma3nTensorNames.PER_LAYER_PROJ_NORM)
        )
    }
}

/**
 * Convenience loader: reads weights from GGUF source (sequential — always
 * dequantized to dense floats), maps them into runtime structure.
 */
public suspend fun <T : DType> loadGemma3nRuntimeWeights(
    ctx: ExecutionContext,
    sourceProvider: () -> Source,
    dtype: KClass<T>
): Gemma3nRuntimeWeights<T> {
    val loader = Gemma3nWeightLoader(sourceProvider = sourceProvider)
    val loaded = loader.loadToMap<T, Float>(ctx, dtype)
    return Gemma3nWeightMapper.map(loaded)
}

/** Backward-compatible overload defaulting to FP32. */
public suspend fun loadGemma3nRuntimeWeights(
    ctx: ExecutionContext,
    sourceProvider: () -> Source
): Gemma3nRuntimeWeights<FP32> = loadGemma3nRuntimeWeights(ctx, sourceProvider, FP32::class)

// ============== Streaming API (for large files >2GB) ==============

/**
 * Load Gemma 3n runtime weights via the engine loader. Default form keeps
 * quantized tensors packed; pass [GEMMA_DEQUANTIZE_ALL] for dense FP32.
 */
@ExperimentalMemoryApi
public suspend fun <T : DType> loadGemma3nRuntimeWeightsStreaming(
    ctx: ExecutionContext,
    randomAccessProvider: () -> RandomAccessSource,
    dtype: KClass<T>,
    weightForm: WeightForm? = null
): Gemma3nRuntimeWeights<T> {
    val loader = Gemma3nWeightLoader(
        randomAccessProvider = randomAccessProvider,
        weightForm = weightForm
    )
    val loaded = loader.loadToMapStreaming<T, Float>(ctx, dtype)
    return Gemma3nWeightMapper.map(loaded)
}

/** Backward-compatible overload defaulting to FP32. */
@ExperimentalMemoryApi
public suspend fun loadGemma3nRuntimeWeightsStreaming(
    ctx: ExecutionContext,
    randomAccessProvider: () -> RandomAccessSource,
    weightForm: WeightForm? = null
): Gemma3nRuntimeWeights<FP32> = loadGemma3nRuntimeWeightsStreaming(ctx, randomAccessProvider, FP32::class, weightForm)
