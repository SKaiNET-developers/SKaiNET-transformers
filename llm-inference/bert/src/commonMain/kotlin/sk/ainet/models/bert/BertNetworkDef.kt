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

        // Encoder layers. BERT is POST-norm: each residual adds the value the
        // sub-block started from, and LayerNorm sits AFTER the add:
        //   h1 = LN(x + MHA(x));  h2 = LN(h1 + FFN(h1))
        // The transformer blocks wire each ResidualAdd to the value at the
        // start of its residual segment (the input of the module right after
        // the previous ResidualAdd) — correct for pre-norm decoder stacks, but
        // for BERT the FFN residual must start AT the post-attention LayerNorm
        // output. Splitting the layer into two blocks puts each residual
        // boundary exactly there: within a block, the first segment starts at
        // the block input.
        for (layer in 0 until nLayers) {
            val attnStage = StageImpl<T, V>(nnCtx, "encoder.layer.$layer.attn", T::class)
            // Self-attention (bidirectional, no causal mask, with bias)
            attnStage.multiHeadAttention(
                dim = dim,
                nHeads = nHeads,
                causal = false,
                bias = true,
                id = "attention"
            )
            attnStage.residual()
            attnStage.layerNorm(intArrayOf(dim), eps, id = "attn_ln")
            dslImpl.modules += HybridTransformerBlock(attnStage.modules.toList(), name = "encoder.layer.$layer.attn")

            // GeLU FFN; residual adds this block's input = post-attention LN output
            val ffnStage = StageImpl<T, V>(nnCtx, "encoder.layer.$layer.ffn", T::class)
            ffnStage.dense(ffnDim, id = "intermediate")
            ffnStage.activation { it.gelu() }
            ffnStage.dense(dim, id = "output")
            ffnStage.residual()
            ffnStage.layerNorm(intArrayOf(dim), eps, id = "output_ln")
            dslImpl.modules += HybridTransformerBlock(ffnStage.modules.toList(), name = "encoder.layer.$layer.ffn")
        }
    }
}
