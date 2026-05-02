package sk.ainet.models.apertus

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32

/**
 * Model metadata for Apertus architecture.
 */
public data class ApertusModelMetadata(
    val architecture: String,
    val embeddingLength: Int,
    val contextLength: Int,
    val blockCount: Int,
    val headCount: Int,
    val kvHeadCount: Int,
    val feedForwardLength: Int,
    val ropeDimensionCount: Int?,
    val vocabSize: Int,
    val ropeTheta: Float = 12000000f,
    val rmsNormEps: Float = 1e-5f,
    val qkNorm: Boolean = true,
    val hiddenAct: String = "xielu",
    val tiedEmbeddings: Boolean = false,
    val bosTokenId: Int = 1,
    val eosTokenId: Int = 2
)

/**
 * Per-layer xIELU activation parameters.
 *
 * All four are learned scalar parameters stored as BF16 in the model weights.
 */
public data class ApertusXIELUParams(
    val alphaP: Float,
    val alphaN: Float,
    val beta: Float,
    val eps: Float
)

/**
 * Per-layer weights for Apertus.
 *
 * Key differences from LLaMA:
 * - No `ffnGate` (ungated MLP)
 * - Has `qNorm` and `kNorm` for QK-norm
 * - Has `xieluParams` for learned activation parameters
 */
public data class ApertusLayerWeights<T : DType>(
    val attnNorm: Tensor<T, Float>,
    val wq: Tensor<T, Float>,
    val wk: Tensor<T, Float>,
    val wv: Tensor<T, Float>,
    val wo: Tensor<T, Float>,
    val qNorm: Tensor<T, Float>,
    val kNorm: Tensor<T, Float>,
    val ffnNorm: Tensor<T, Float>,
    val ffnDown: Tensor<T, Float>,
    val ffnUp: Tensor<T, Float>,
    val xieluParams: ApertusXIELUParams
)

/**
 * Complete model weights for Apertus runtime.
 */
public data class ApertusRuntimeWeights<T : DType>(
    val metadata: ApertusModelMetadata,
    val tokenEmbedding: Tensor<T, Float>,
    val layers: List<ApertusLayerWeights<T>>,
    val outputNorm: Tensor<T, Float>,
    val outputWeight: Tensor<T, Float>,
    val ropeFreqs: Tensor<T, Float>? = null,
    val preTransposed: Boolean = false
)

/**
 * Canonical tensor name constants for Apertus (GGUF-style naming).
 */
public object ApertusTensorNames {
    public const val TOKEN_EMBEDDINGS: String = "token_embd.weight"
    public const val OUTPUT_NORM: String = "output_norm.weight"
    public const val OUTPUT_WEIGHT: String = "output.weight"
    public const val ROPE_FREQS: String = "rope_freqs.weight"

    public fun attnNorm(layer: Int): String = "blk.$layer.attn_norm.weight"
    public fun attnQ(layer: Int): String = "blk.$layer.attn_q.weight"
    public fun attnK(layer: Int): String = "blk.$layer.attn_k.weight"
    public fun attnV(layer: Int): String = "blk.$layer.attn_v.weight"
    public fun attnOut(layer: Int): String = "blk.$layer.attn_output.weight"
    public fun attnQNorm(layer: Int): String = "blk.$layer.attn_q_norm.weight"
    public fun attnKNorm(layer: Int): String = "blk.$layer.attn_k_norm.weight"
    public fun ffnNorm(layer: Int): String = "blk.$layer.ffn_norm.weight"
    public fun ffnDown(layer: Int): String = "blk.$layer.ffn_down.weight"
    public fun ffnUp(layer: Int): String = "blk.$layer.ffn_up.weight"
}

// ============== Quantized (lazy-dequant) weight structures ==============

/**
 * Per-layer weights with large projection matrices stored in quantized form.
 *
 * Small tensors (norms) are kept as FP32 since they're negligible in size.
 * Large 2D weight matrices stay quantized until execution time, saving 4-8x memory.
 */
public data class ApertusQuantizedLayerWeights(
    // Small tensors — always FP32
    val attnNorm: Tensor<FP32, Float>,
    val qNorm: Tensor<FP32, Float>,
    val kNorm: Tensor<FP32, Float>,
    val ffnNorm: Tensor<FP32, Float>,
    val xieluParams: ApertusXIELUParams,
    // Large projection matrices — quantized, dequantized at execution time
    val wq: QuantizedTensor,
    val wk: QuantizedTensor,
    val wv: QuantizedTensor,
    val wo: QuantizedTensor,
    val ffnUp: QuantizedTensor,
    val ffnDown: QuantizedTensor
)

/**
 * Complete model weights for Apertus with lazy dequantization.
 *
 * Token embedding is kept FP32 (needed for integer-indexed lookup).
 * Output weight and all per-layer projection matrices remain quantized.
 */
public data class ApertusQuantizedRuntimeWeights(
    val metadata: ApertusModelMetadata,
    val tokenEmbedding: Tensor<FP32, Float>,
    val layers: List<ApertusQuantizedLayerWeights>,
    val outputNorm: Tensor<FP32, Float>,
    val outputWeight: QuantizedTensor,
    val ropeFreqs: Tensor<FP32, Float>? = null
)

/**
 * Intermediate weight container used during loading.
 *
 * @property quantTypes For each tensor that was quantized in the source GGUF,
 *   the original [sk.ainet.io.gguf.GGMLQuantizationType]. Empty when the
 *   loader fully dequantized everything (e.g. `QuantPolicy.DEQUANTIZE_TO_FP32`).
 *   Populated under `NATIVE_OPTIMIZED` so the JVM-side
 *   `ApertusMemSegConverter` can wrap each tensor in the right
 *   block-major TensorData (Q4_KBlockTensorData / Q6_KBlockTensorData / …)
 *   before the model runs.
 * @property logicalShapes Logical `[out, in]` shapes of every tensor, indexed
 *   by tensor name. Under `NATIVE_OPTIMIZED` the actual stored
 *   `tensors[name].shape` is byte-level rank-1, so the converter needs the
 *   logical shape from this side-map to compute the relayout target shape.
 * @property quantBytes Raw quantized payload bytes keyed by tensor name.
 *   The loader stashes these alongside the byte-shape tensors in `tensors`
 *   so the converter can re-wrap them in the right `Q*_KBlockTensorData`
 *   without having to dig the bytes back out of the loader's intermediate
 *   `Int8 [byteCount]` tensor.
 */
public data class ApertusWeights<T : DType, V>(
    val metadata: ApertusModelMetadata,
    val tensors: Map<String, Tensor<T, V>>,
    val xieluParams: Map<Int, ApertusXIELUParams> = emptyMap(),
    val preTransposed: Boolean = false,
    val quantTypes: Map<String, sk.ainet.io.gguf.GGMLQuantizationType> = emptyMap(),
    val logicalShapes: Map<String, sk.ainet.lang.tensor.Shape> = emptyMap(),
    val quantBytes: Map<String, ByteArray> = emptyMap()
)

/**
 * Maps loaded tensors to [ApertusRuntimeWeights], validating shapes.
 */
public object ApertusWeightMapper {

    public fun <T : DType> map(weights: ApertusWeights<T, Float>): ApertusRuntimeWeights<T> {
        val metadata = weights.metadata
        val headSize = metadata.embeddingLength / metadata.headCount
        require(headSize * metadata.headCount == metadata.embeddingLength) {
            "headSize is not divisible: dim=${metadata.embeddingLength} heads=${metadata.headCount}"
        }

        fun get(name: String): Tensor<T, Float> =
            weights.tensors[name] ?: error("Missing tensor: $name")

        val tokenEmbedding = get(ApertusTensorNames.TOKEN_EMBEDDINGS)
        val outputNorm = get(ApertusTensorNames.OUTPUT_NORM)
        val outputWeight = get(ApertusTensorNames.OUTPUT_WEIGHT)

        val kvDim = metadata.kvHeadCount * headSize

        val layers = (0 until metadata.blockCount).map { layer ->
            val xieluParams = weights.xieluParams[layer]
                ?: error("Missing xIELU params for layer $layer")

            ApertusLayerWeights(
                attnNorm = get(ApertusTensorNames.attnNorm(layer)),
                wq = get(ApertusTensorNames.attnQ(layer)),
                wk = get(ApertusTensorNames.attnK(layer)),
                wv = get(ApertusTensorNames.attnV(layer)),
                wo = get(ApertusTensorNames.attnOut(layer)),
                qNorm = get(ApertusTensorNames.attnQNorm(layer)),
                kNorm = get(ApertusTensorNames.attnKNorm(layer)),
                ffnNorm = get(ApertusTensorNames.ffnNorm(layer)),
                ffnDown = get(ApertusTensorNames.ffnDown(layer)),
                ffnUp = get(ApertusTensorNames.ffnUp(layer)),
                xieluParams = xieluParams
            )
        }

        val ropeFreqs = weights.tensors[ApertusTensorNames.ROPE_FREQS]

        return ApertusRuntimeWeights(
            metadata = metadata,
            tokenEmbedding = tokenEmbedding,
            layers = layers,
            outputNorm = outputNorm,
            outputWeight = outputWeight,
            ropeFreqs = ropeFreqs,
            preTransposed = weights.preTransposed
        )
    }
}

/**
 * Maps loaded quantized weight containers to [ApertusQuantizedRuntimeWeights].
 */
public object ApertusQuantizedWeightMapper {

    public fun map(weights: ApertusQuantizedWeights): ApertusQuantizedRuntimeWeights {
        val metadata = weights.metadata

        fun getTensor(name: String): Tensor<FP32, Float> =
            weights.fp32Tensors[name] ?: error("Missing FP32 tensor: $name")

        fun getQuantized(name: String): QuantizedTensor =
            weights.quantizedTensors[name] ?: error("Missing quantized tensor: $name")

        val layers = (0 until metadata.blockCount).map { layer ->
            val xieluParams = weights.xieluParams[layer]
                ?: error("Missing xIELU params for layer $layer")

            ApertusQuantizedLayerWeights(
                attnNorm = getTensor(ApertusTensorNames.attnNorm(layer)),
                qNorm = getTensor(ApertusTensorNames.attnQNorm(layer)),
                kNorm = getTensor(ApertusTensorNames.attnKNorm(layer)),
                ffnNorm = getTensor(ApertusTensorNames.ffnNorm(layer)),
                xieluParams = xieluParams,
                wq = getQuantized(ApertusTensorNames.attnQ(layer)),
                wk = getQuantized(ApertusTensorNames.attnK(layer)),
                wv = getQuantized(ApertusTensorNames.attnV(layer)),
                wo = getQuantized(ApertusTensorNames.attnOut(layer)),
                ffnUp = getQuantized(ApertusTensorNames.ffnUp(layer)),
                ffnDown = getQuantized(ApertusTensorNames.ffnDown(layer))
            )
        }

        return ApertusQuantizedRuntimeWeights(
            metadata = metadata,
            tokenEmbedding = getTensor(ApertusTensorNames.TOKEN_EMBEDDINGS),
            layers = layers,
            outputNorm = getTensor(ApertusTensorNames.OUTPUT_NORM),
            outputWeight = getQuantized(ApertusTensorNames.OUTPUT_WEIGHT),
            ropeFreqs = weights.fp32Tensors[ApertusTensorNames.ROPE_FREQS]
        )
    }
}

/**
 * Intermediate container for quantized weight loading.
 *
 * Small tensors (norms, embeddings) go into [fp32Tensors].
 * Large weight matrices go into [quantizedTensors].
 */
public data class ApertusQuantizedWeights(
    val metadata: ApertusModelMetadata,
    val fp32Tensors: Map<String, Tensor<FP32, Float>>,
    val quantizedTensors: Map<String, QuantizedTensor>,
    val xieluParams: Map<Int, ApertusXIELUParams> = emptyMap()
)
