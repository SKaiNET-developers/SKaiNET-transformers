package sk.ainet.models.llama

import sk.ainet.lang.nn.dsl.decoder.DecoderGgufWeightLoader
import sk.ainet.lang.nn.dsl.decoder.DecoderGgufWeights
import sk.ainet.lang.nn.dsl.decoder.DecoderTensorNames
import sk.ainet.lang.nn.dsl.decoder.GgufDecoderMetadata

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import sk.ainet.context.ExecutionContext
import kotlinx.io.Source
import sk.ainet.io.RandomAccessSource
import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.plan.EncodingRequest
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.plan.WeightShapeOrientation
import kotlin.reflect.KClass

public data class LlamaLayerWeights<T : DType>(
    val attnNorm: Tensor<T, Float>,
    val wq: Tensor<T, Float>,
    val wk: Tensor<T, Float>,
    val wv: Tensor<T, Float>,
    val wo: Tensor<T, Float>,
    val ffnNorm: Tensor<T, Float>,
    val ffnGate: Tensor<T, Float>,
    val ffnDown: Tensor<T, Float>,
    val ffnUp: Tensor<T, Float>,
    val qNorm: Tensor<T, Float>? = null,
    val kNorm: Tensor<T, Float>? = null
)

public data class LlamaRuntimeWeights<T : DType>(
    val metadata: GgufDecoderMetadata,
    val tokenEmbedding: Tensor<T, Float>,
    val ropeFreqReal: Tensor<T, Float>?,
    val ropeFreqImag: Tensor<T, Float>?,
    val layers: List<LlamaLayerWeights<T>>,
    val outputNorm: Tensor<T, Float>,
    val outputWeight: Tensor<T, Float>
)

/**
    Converts loader-emitted tensors to a typed structure ready for runtime/module wiring.
    Enforces basic shape sanity against the metadata to fail early before graph construction.
 */
public object LlamaWeightMapper {

    public fun <T : DType> map(weights: DecoderGgufWeights<T, Float>): LlamaRuntimeWeights<T> {
        val metadata = weights.metadata
        val headSize = metadata.embeddingLength / metadata.headCount
        require(headSize * metadata.headCount == metadata.embeddingLength) {
            "headSize is not divisible: dim=${metadata.embeddingLength} heads=${metadata.headCount}"
        }

        fun get(name: String): Tensor<T, Float> =
            weights.tensors[name] ?: error("Missing tensor: $name")

        fun Tensor<*, *>.requireShape(expected: Shape, label: String, tensorName: String) {
            if (shape != expected) {
                error("$label expected shape $expected but was $shape")
            }
        }

        fun Tensor<*, *>.require2D(rows: Int, cols: Int, label: String, tensorName: String) =
            requireShape(Shape(rows, cols), label, tensorName)

        fun Tensor<*, *>.require1D(size: Int, label: String, tensorName: String) =
            requireShape(Shape(size), label, tensorName)

        val tokenEmbedding = get(DecoderTensorNames.TOKEN_EMBEDDINGS)
        // After transpose: [vocab_size, embedding_dim] (row-major, standard for embedding lookup)
        tokenEmbedding.require2D(metadata.vocabSize, metadata.embeddingLength, "token embedding", DecoderTensorNames.TOKEN_EMBEDDINGS)

        val outputNorm = get(DecoderTensorNames.OUTPUT_NORM)
        outputNorm.require1D(metadata.embeddingLength, "output norm", DecoderTensorNames.OUTPUT_NORM)

        val outputWeight = get(DecoderTensorNames.OUTPUT_WEIGHT)
        // After transpose: [vocab_size, embedding_dim] (row-major)
        outputWeight.require2D(metadata.vocabSize, metadata.embeddingLength, "output weight", DecoderTensorNames.OUTPUT_WEIGHT)

        val ropeReal = weights.tensors[DecoderTensorNames.ROPE_FREQS_REAL]?.also {
            it.requireShape(Shape(metadata.contextLength, headSize / 2), "rope.freq_cis_real", DecoderTensorNames.ROPE_FREQS_REAL)
        }
        val ropeImag = weights.tensors[DecoderTensorNames.ROPE_FREQS_IMAG]?.also {
            it.requireShape(Shape(metadata.contextLength, headSize / 2), "rope.freq_cis_imag", DecoderTensorNames.ROPE_FREQS_IMAG)
        }

        // For GQA: K/V projections have shape [kv_dim, dim] where kv_dim = kv_heads * head_size
        val kvDim = metadata.kvHeadCount * headSize

        val layers = (0 until metadata.blockCount).map { layer ->
            val attnNorm = get(DecoderTensorNames.attnNorm(layer)).apply {
                require1D(metadata.embeddingLength, "blk.$layer.attn_norm.weight", DecoderTensorNames.attnNorm(layer))
            }
            val wq = get(DecoderTensorNames.attnQ(layer)).apply {
                // After transpose: [dim, dim] (symmetric, so unchanged)
                require2D(metadata.embeddingLength, metadata.embeddingLength, "blk.$layer.attn_q.weight", DecoderTensorNames.attnQ(layer))
            }
            val wk = get(DecoderTensorNames.attnK(layer)).apply {
                // After transpose: [kv_dim, dim] (was [dim, kv_dim] in GGUF)
                require2D(kvDim, metadata.embeddingLength, "blk.$layer.attn_k.weight", DecoderTensorNames.attnK(layer))
            }
            val wv = get(DecoderTensorNames.attnV(layer)).apply {
                // After transpose: [kv_dim, dim] (was [dim, kv_dim] in GGUF)
                require2D(kvDim, metadata.embeddingLength, "blk.$layer.attn_v.weight", DecoderTensorNames.attnV(layer))
            }
            val wo = get(DecoderTensorNames.attnOut(layer)).apply {
                // After transpose: [dim, dim] (symmetric, so unchanged)
                require2D(metadata.embeddingLength, metadata.embeddingLength, "blk.$layer.attn_output.weight", DecoderTensorNames.attnOut(layer))
            }
            val ffnNorm = get(DecoderTensorNames.ffnNorm(layer)).apply {
                require1D(metadata.embeddingLength, "blk.$layer.ffn_norm.weight", DecoderTensorNames.ffnNorm(layer))
            }
            val ffnGate = get(DecoderTensorNames.ffnGate(layer)).apply {
                // After transpose: [ff_dim, dim] (was [dim, ff_dim] in GGUF)
                require2D(metadata.feedForwardLength, metadata.embeddingLength, "blk.$layer.ffn_gate.weight", DecoderTensorNames.ffnGate(layer))
            }
            val ffnDown = get(DecoderTensorNames.ffnDown(layer)).apply {
                // After transpose: [dim, ff_dim] (was [ff_dim, dim] in GGUF)
                require2D(metadata.embeddingLength, metadata.feedForwardLength, "blk.$layer.ffn_down.weight", DecoderTensorNames.ffnDown(layer))
            }
            val ffnUp = get(DecoderTensorNames.ffnUp(layer)).apply {
                // After transpose: [ff_dim, dim] (was [dim, ff_dim] in GGUF)
                require2D(metadata.feedForwardLength, metadata.embeddingLength, "blk.$layer.ffn_up.weight", DecoderTensorNames.ffnUp(layer))
            }
            val qNorm = weights.tensors[DecoderTensorNames.attnQNorm(layer)]
            val kNorm = weights.tensors[DecoderTensorNames.attnKNorm(layer)]

            LlamaLayerWeights(
                attnNorm = attnNorm,
                wq = wq,
                wk = wk,
                wv = wv,
                wo = wo,
                ffnNorm = ffnNorm,
                ffnGate = ffnGate,
                ffnDown = ffnDown,
                ffnUp = ffnUp,
                qNorm = qNorm,
                kNorm = kNorm
            )
        }

        return LlamaRuntimeWeights(
            metadata = metadata,
            tokenEmbedding = tokenEmbedding,
            ropeFreqReal = ropeReal,
            ropeFreqImag = ropeImag,
            layers = layers,
            outputNorm = outputNorm,
            outputWeight = outputWeight
        )
    }
}

/**
 * Convenience loader for the legacy eager runtime: reads weights from a sequential GGUF
 * source (dense, fully dequantized) and maps them into the runtime structure.
 */
public suspend fun <T : DType> loadLlamaRuntimeWeights(
    ctx: ExecutionContext,
    sourceProvider: () -> Source,
    dtype: KClass<T>
): LlamaRuntimeWeights<T> {
    val loader = DecoderGgufWeightLoader(sourceProvider = sourceProvider)
    val loaded = loader.loadToMap<T, Float>(ctx, dtype)
    return LlamaWeightMapper.map(loaded)
}

/** Backward-compatible overload defaulting to FP32. */
public suspend fun loadLlamaRuntimeWeights(
    ctx: ExecutionContext,
    sourceProvider: () -> Source
): LlamaRuntimeWeights<FP32> = loadLlamaRuntimeWeights(ctx, sourceProvider, FP32::class)

// ============== Streaming API (for large files >2GB) ==============

/**
 * Load LLaMA runtime weights for the legacy eager runtime using the streaming API.
 * Parses metadata only, then streams tensors through the engine loader with every
 * tensor dequantized to dense FP32 — the eager runtime's shape checks and kernels
 * assume dense data.
 */
@OptIn(ExperimentalMemoryApi::class)
public suspend fun <T : DType> loadLlamaRuntimeWeightsStreaming(
    ctx: ExecutionContext,
    randomAccessProvider: () -> RandomAccessSource,
    dtype: KClass<T>,
    acceptedArchitectures: Set<String> = setOf("llama")
): LlamaRuntimeWeights<T> {
    val loader = DecoderGgufWeightLoader(
        randomAccessProvider = randomAccessProvider,
        acceptedArchitectures = acceptedArchitectures,
        weightForm = WeightForm(
            encoding = EncodingRequest.DequantizeTo(FP32),
            shape = WeightShapeOrientation.OUT_IN
        )
    )
    val loaded = loader.loadToMapStreaming<T, Float>(ctx, dtype)
    return LlamaWeightMapper.map(loaded)
}

/** Backward-compatible overload defaulting to FP32. */
public suspend fun loadLlamaRuntimeWeightsStreaming(
    ctx: ExecutionContext,
    randomAccessProvider: () -> RandomAccessSource
): LlamaRuntimeWeights<FP32> = loadLlamaRuntimeWeightsStreaming(ctx, randomAccessProvider, FP32::class)
