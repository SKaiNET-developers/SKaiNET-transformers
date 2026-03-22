package sk.ainet.models.llama

import sk.ainet.apps.llm.TransformerBlock
import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.dsl.NeuralNetworkDslImpl
import sk.ainet.lang.nn.dsl.StageImpl
import sk.ainet.lang.nn.dsl.sequential
import sk.ainet.lang.types.DType

/**
 * Llama architecture defined via the network DSL.
 *
 * Replaces the hand-coded [LlamaRuntime] with a declarative definition that:
 * - Builds a Module<T,V> tree (for direct execution and weight loading)
 * - Can be traced into a GraphProgram (DAG) for optimization
 *
 * Architecture: Embedding → N × (RMSNorm → MHA(RoPE, KVCache) → Residual →
 *               RMSNorm → SwiGLU FFN → Residual) → RMSNorm → Dense
 *
 * Each transformer layer uses [TransformerBlock] (not the generic MLP) so that
 * [ResidualAdd] modules receive the correct skip-connection input.
 */
public inline fun <reified T : DType, V> llamaNetwork(
    metadata: LlamaModelMetadata
): Module<T, V> {
    val dim = metadata.embeddingLength
    val nHeads = metadata.headCount
    val nKVHeads = metadata.kvHeadCount
    val nLayers = metadata.blockCount
    val ffnDim = metadata.feedForwardLength
    val seqLen = metadata.contextLength
    val vocabSize = metadata.vocabSize
    val headDim = metadata.ropeDimensionCount ?: (dim / nHeads)
    val eps = 1e-5f

    return sequential<T, V> {
        embedding(vocabSize, dim, id = "token_embd")

        // Build each transformer layer via the DSL helpers, but wrap in
        // TransformerBlock instead of MLP so residual connections work.
        val dslImpl = this as NeuralNetworkDslImpl<T, V>
        val nnCtx = DefaultNeuralNetworkExecutionContext()
        for (layer in 0 until nLayers) {
            val stage = StageImpl<T, V>(nnCtx, "blk.$layer", T::class)
            stage.rmsNorm(dim, eps, id = "attn_norm")
            stage.multiHeadAttention(
                dim = dim,
                nHeads = nHeads,
                nKVHeads = nKVHeads,
                causal = true,
                id = "attn"
            ) {
                rope(headDim, seqLen)
                kvCache(seqLen, nKVHeads, headDim)
            }
            stage.residual()

            stage.rmsNorm(dim, eps, id = "ffn_norm")
            stage.swiGluFFN(dim, ffnDim, id = "ffn")
            stage.residual()

            dslImpl.modules += TransformerBlock(stage.modules.toList(), name = "blk.$layer")
        }

        rmsNorm(dim, eps, id = "output_norm")
        dense(vocabSize, id = "output")
    }
}
