package sk.ainet.models.llama

import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.dsl.NeuralNetworkDsl
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

        for (layer in 0 until nLayers) {
            stage("blk.$layer") {
                rmsNorm(dim, eps, id = "attn_norm")
                multiHeadAttention(
                    dim = dim,
                    nHeads = nHeads,
                    nKVHeads = nKVHeads,
                    causal = true,
                    id = "attn"
                ) {
                    rope(headDim, seqLen)
                    kvCache(seqLen, nKVHeads, headDim)
                }
                residual()

                rmsNorm(dim, eps, id = "ffn_norm")
                swiGluFFN(dim, ffnDim, id = "ffn")
                residual()
            }
        }

        rmsNorm(dim, eps, id = "output_norm")
        dense(vocabSize, id = "output")
    }
}
