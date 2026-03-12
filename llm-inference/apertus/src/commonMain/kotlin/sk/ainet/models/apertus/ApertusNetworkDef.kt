package sk.ainet.models.apertus

import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.dsl.NeuralNetworkDsl
import sk.ainet.lang.nn.dsl.sequential
import sk.ainet.lang.types.DType

/**
 * Apertus architecture defined via the network DSL.
 *
 * Replaces the hand-coded [ApertusRuntime] with a declarative definition.
 *
 * Key differences from Llama:
 * - QK-Norm: per-head RMSNorm on Q and K before RoPE
 * - xIELU activation with per-layer learned parameters (replaces SiLU)
 * - Ungated FFN: only up_proj + down_proj (no gate_proj)
 *
 * Architecture: Embedding → N × (RMSNorm → MHA(QKNorm, RoPE, KVCache) → Residual →
 *               RMSNorm → Dense → xIELU → Dense → Residual) → RMSNorm → Dense
 */
public inline fun <reified T : DType, V> apertusNetwork(
    metadata: ApertusModelMetadata
): Module<T, V> {
    val dim = metadata.embeddingLength
    val nHeads = metadata.headCount
    val nKVHeads = metadata.kvHeadCount
    val nLayers = metadata.blockCount
    val ffnDim = metadata.feedForwardLength
    val seqLen = metadata.contextLength
    val vocabSize = metadata.vocabSize
    val headDim = metadata.ropeDimensionCount ?: (dim / nHeads)
    val eps = metadata.rmsNormEps

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
                    qkNorm = true,
                    id = "attn"
                ) {
                    rope(headDim, seqLen)
                    kvCache(seqLen, nKVHeads, headDim)
                }
                residual()

                // Ungated FFN with xIELU (no SwiGLU)
                rmsNorm(dim, eps, id = "ffn_norm")
                dense(ffnDim, id = "ffn_up")
                xielu(id = "act_fn")
                dense(dim, id = "ffn_down")
                residual()
            }
        }

        rmsNorm(dim, eps, id = "output_norm")
        dense(vocabSize, id = "output")
    }
}
