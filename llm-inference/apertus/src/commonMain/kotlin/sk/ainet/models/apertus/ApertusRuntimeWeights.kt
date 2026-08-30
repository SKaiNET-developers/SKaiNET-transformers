package sk.ainet.models.apertus

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

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
    val ropeFreqs: Tensor<T, Float>? = null
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

/**
 * Intermediate weight container used during loading.
 *
 * Tensors are keyed by GGUF name with logical `[out, in]` shapes. On the
 * engine-backed streaming path, quantized projection matrices keep their
 * stored block encoding as packed
 * [sk.ainet.lang.tensor.storage.PackedBlockStorage] data; the sequential
 * path delivers dense floats.
 */
public data class ApertusWeights<T : DType, V>(
    val metadata: ApertusModelMetadata,
    val tensors: Map<String, Tensor<T, V>>,
    val xieluParams: Map<Int, ApertusXIELUParams> = emptyMap()
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
            ropeFreqs = ropeFreqs
        )
    }
}
