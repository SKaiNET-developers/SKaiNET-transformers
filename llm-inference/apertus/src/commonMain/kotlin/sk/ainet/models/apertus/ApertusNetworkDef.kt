package sk.ainet.models.apertus

import sk.ainet.apps.llm.HybridTransformerBlock
import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.dsl.NeuralNetworkDslImpl
import sk.ainet.lang.nn.dsl.StageImpl
import sk.ainet.lang.nn.dsl.embedding
import sk.ainet.lang.nn.dsl.multiHeadAttention
import sk.ainet.lang.nn.dsl.residual
import sk.ainet.lang.nn.dsl.rmsNorm
import sk.ainet.lang.nn.dsl.sequential
import sk.ainet.lang.nn.dsl.xielu
import sk.ainet.lang.nn.transformer.RoPEMode
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
        val dslImpl = this as NeuralNetworkDslImpl<T, V>
        dslImpl.embedding(vocabSize, dim, id = "token_embd")
        val nnCtx = DefaultNeuralNetworkExecutionContext()
        for (layer in 0 until nLayers) {
            val stage = StageImpl<T, V>(nnCtx, "blk.$layer", T::class)
            stage.rmsNorm(dim, eps, id = "attn_norm")
            stage.multiHeadAttention(
                dim = dim,
                nHeads = nHeads,
                nKVHeads = nKVHeads,
                causal = true,
                qkNorm = true,
                id = "attn"
            ) {
                // Apertus is an HF rotate-half model: llama.cpp runs it as
                // LLAMA_ROPE_TYPE_NEOX with rope_theta from the GGUF (12M on
                // Apertus-8B). The DSL defaults (INTERLEAVED, base 10_000)
                // silently produced wrong rotations on every position.
                rope(headDim, seqLen, mode = RoPEMode.SPLIT_HALF, base = metadata.ropeTheta)
                kvCache(seqLen, nKVHeads, headDim)
            }
            stage.residual()

            // Ungated FFN with xIELU (no SwiGLU)
            stage.rmsNorm(dim, eps, id = "ffn_norm")
            stage.dense(ffnDim, id = "ffn_up")
            stage.xielu(id = "act_fn")
            stage.dense(dim, id = "ffn_down")
            stage.residual()

            dslImpl.modules += HybridTransformerBlock(stage.modules.toList(), name = "blk.$layer")
        }

        dslImpl.rmsNorm(dim, eps, id = "output_norm")
        dense(vocabSize, id = "output")
    }
}
