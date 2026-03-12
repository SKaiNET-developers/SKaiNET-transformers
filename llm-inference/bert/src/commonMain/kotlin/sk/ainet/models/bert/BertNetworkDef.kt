package sk.ainet.models.bert

import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.dsl.NeuralNetworkDsl
import sk.ainet.lang.nn.dsl.sequential
import sk.ainet.lang.types.DType

/**
 * BERT architecture defined via the network DSL.
 *
 * Replaces the hand-coded [BertRuntime] with a declarative definition.
 *
 * Architecture: Embedding → LayerNorm →
 *               N × (MHA(bidirectional, bias) → Residual → LayerNorm →
 *               Dense → GeLU → Dense → Residual → LayerNorm)
 *
 * Note: BERT has 3 embedding types (word, position, token_type) but only
 * word embeddings are defined here. Position and token_type are handled
 * by the runtime's forward logic as additive lookups.
 */
public inline fun <reified T : DType, V> bertNetwork(
    config: BertModelConfig
): Module<T, V> {
    val dim = config.hiddenSize
    val nHeads = config.numAttentionHeads
    val nLayers = config.numHiddenLayers
    val ffnDim = config.intermediateSize
    val vocabSize = config.vocabSize
    val eps = config.layerNormEps

    return sequential<T, V> {
        // Embedding stage: word embeddings + LayerNorm
        stage("embeddings") {
            embedding(vocabSize, dim, id = "word_embeddings")
            layerNorm(intArrayOf(dim), eps, id = "LayerNorm")
        }

        // Encoder layers
        for (layer in 0 until nLayers) {
            stage("encoder.layer.$layer") {
                // Self-attention (bidirectional, no causal mask, with bias)
                multiHeadAttention(
                    dim = dim,
                    nHeads = nHeads,
                    causal = false,
                    bias = true,
                    id = "attention"
                )
                residual()
                layerNorm(intArrayOf(dim), eps, id = "attn_ln")

                // GeLU FFN
                dense(ffnDim, id = "intermediate")
                gelu()
                dense(dim, id = "output")
                residual()
                layerNorm(intArrayOf(dim), eps, id = "output_ln")
            }
        }
    }
}
