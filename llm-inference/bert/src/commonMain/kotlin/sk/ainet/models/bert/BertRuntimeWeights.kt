package sk.ainet.models.bert

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

// BertModelConfig and MDBR_LEAF_IR_CONFIG live in BertConfig.kt (same package).

/**
 * Per-layer weights for a BERT encoder layer.
 */
public data class BertLayerWeights<T : DType>(
    // Self-attention
    val queryWeight: Tensor<T, Float>,
    val queryBias: Tensor<T, Float>,
    val keyWeight: Tensor<T, Float>,
    val keyBias: Tensor<T, Float>,
    val valueWeight: Tensor<T, Float>,
    val valueBias: Tensor<T, Float>,
    // Attention output
    val attnOutputWeight: Tensor<T, Float>,
    val attnOutputBias: Tensor<T, Float>,
    val attnLayerNormWeight: Tensor<T, Float>,
    val attnLayerNormBias: Tensor<T, Float>,
    // FFN (intermediate + output)
    val intermediateWeight: Tensor<T, Float>,
    val intermediateBias: Tensor<T, Float>,
    val outputWeight: Tensor<T, Float>,
    val outputBias: Tensor<T, Float>,
    val outputLayerNormWeight: Tensor<T, Float>,
    val outputLayerNormBias: Tensor<T, Float>
)

/**
 * Complete BERT runtime weights: embeddings + encoder layers + optional pooler/projection.
 */
public data class BertRuntimeWeights<T : DType>(
    val config: BertModelConfig,
    // Embeddings
    val wordEmbeddings: Tensor<T, Float>,
    val positionEmbeddings: Tensor<T, Float>,
    val tokenTypeEmbeddings: Tensor<T, Float>,
    val embeddingLayerNormWeight: Tensor<T, Float>,
    val embeddingLayerNormBias: Tensor<T, Float>,
    // Encoder layers
    val layers: List<BertLayerWeights<T>>,
    // Optional pooler dense (cls token projection)
    val poolerDenseWeight: Tensor<T, Float>? = null,
    val poolerDenseBias: Tensor<T, Float>? = null,
    // Optional sentence-transformers dense projection
    val projectionWeight: Tensor<T, Float>? = null,
    val projectionBias: Tensor<T, Float>? = null
)
