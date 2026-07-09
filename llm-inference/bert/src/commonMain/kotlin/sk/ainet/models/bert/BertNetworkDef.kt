package sk.ainet.models.bert

import sk.ainet.apps.llm.HybridTransformerBlock
import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.dsl.NeuralNetworkDslImpl
import sk.ainet.lang.nn.dsl.StageImpl
import sk.ainet.lang.nn.dsl.multiHeadAttention
import sk.ainet.lang.nn.dsl.residual
import sk.ainet.lang.nn.dsl.sequential
import sk.ainet.lang.tensor.gelu
import sk.ainet.lang.types.DType

/**
 * BERT architecture defined via the network DSL.
 *
 * Architecture: BertEmbeddings (word + position + token_type + LayerNorm) →
 *               N × (MHA(bidirectional, bias) → Residual → LayerNorm →
 *               Dense → GeLU → Dense → Residual → LayerNorm)
 *
 * The returned module is a complete `tokens → hidden-states` encoder: feed it
 * a `[L]`-shaped token-id tensor and it produces `[L, hiddenSize]` hidden
 * states. Pooling and the optional sentence-transformers projection live in
 * [BertEncoderRuntime], keeping the traced graph a pure encoder.
 */
public inline fun <reified T : DType, V> bertNetwork(
    config: BertModelConfig
): Module<T, V> {
    val dim = config.hiddenSize
    val nHeads = config.numAttentionHeads
    val nLayers = config.numHiddenLayers
    val ffnDim = config.intermediateSize
    val eps = config.layerNormEps

    return sequential<T, V> {
        val dslImpl = this as NeuralNetworkDslImpl<T, V>
        val nnCtx = DefaultNeuralNetworkExecutionContext()

        // Complete embeddings block: word + position + token_type, LayerNorm
        dslImpl.modules += BertEmbeddings(config, T::class)

        // Encoder layers — use TransformerBlock for residual connections
        for (layer in 0 until nLayers) {
            val stage = StageImpl<T, V>(nnCtx, "encoder.layer.$layer", T::class)
            // Self-attention (bidirectional, no causal mask, with bias)
            stage.multiHeadAttention(
                dim = dim,
                nHeads = nHeads,
                causal = false,
                bias = true,
                id = "attention"
            )
            stage.residual()
            stage.layerNorm(intArrayOf(dim), eps, id = "attn_ln")

            // GeLU FFN
            stage.dense(ffnDim, id = "intermediate")
            stage.activation { it.gelu() }
            stage.dense(dim, id = "output")
            stage.residual()
            stage.layerNorm(intArrayOf(dim), eps, id = "output_ln")

            dslImpl.modules += HybridTransformerBlock(stage.modules.toList(), name = "encoder.layer.$layer")
        }
    }
}
