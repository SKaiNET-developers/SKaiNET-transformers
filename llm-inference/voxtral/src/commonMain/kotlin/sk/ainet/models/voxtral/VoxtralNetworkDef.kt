package sk.ainet.models.voxtral

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
import sk.ainet.lang.nn.dsl.decoder.decoderTransformerNetwork
import sk.ainet.lang.nn.dsl.swiGluFFN
import sk.ainet.lang.types.DType
import sk.ainet.models.llama.LlamaModelMetadata

/**
 * Voxtral TTS text backbone defined via the network DSL.
 *
 * The backbone is a 26-layer Ministral-3B transformer (LLaMA architecture:
 * GQA + SwiGLU FFN + RoPE + RMSNorm, no attention biases, tied embeddings).
 * It generates semantic audio tokens autoregressively from text input.
 *
 * Architecture: `Embedding → 26 × (RMSNorm → MHA(RoPE, KVCache) → Residual →
 * RMSNorm → SwiGLU FFN → Residual) → RMSNorm → Dense`.
 *
 * Default config: dim=3072, heads=32, kv_heads=8, ffn=9216, vocab=131072,
 *                 rope_theta=1M, head_dim=128.
 */
public inline fun <reified T : DType, V> voxtralBackboneNetwork(
    metadata: LlamaModelMetadata,
): Module<T, V> = decoderTransformerNetwork<T, V>(
    metadata = metadata,
    qkNorm = false,
)

/**
 * Voxtral acoustic transformer defined via the network DSL.
 *
 * A 3-layer transformer that takes the backbone's hidden states and generates
 * 36 acoustic codebooks via flow matching (not autoregressive). Uses a different
 * RoPE theta (10k vs 1M) than the backbone.
 *
 * Architecture: 3 × (RMSNorm → MHA(RoPE, KVCache) → Residual →
 *               RMSNorm → SwiGLU FFN → Residual) → RMSNorm
 *
 * Note: This network does NOT include input/output projections for the flow-matching
 * pipeline (embedding of backbone hidden states, acoustic codebook projection).
 * Those are handled separately since flow matching requires noise scheduling
 * and iterative denoising at inference time.
 *
 * Default config: dim=3072, heads=32, kv_heads=8, ffn=9216, layers=3,
 *                 rope_theta=10k, head_dim=128
 *
 * @param ropeBase RoPE base frequency (default: 10_000 for acoustic model)
 */
public inline fun <reified T : DType, V> voxtralAcousticNetwork(
    metadata: LlamaModelMetadata,
    ropeBase: Float = 10_000f
): Module<T, V> {
    val dim = metadata.embeddingLength
    val nHeads = metadata.headCount
    val nKVHeads = metadata.kvHeadCount
    val nLayers = metadata.blockCount
    val ffnDim = metadata.feedForwardLength
    val seqLen = metadata.contextLength
    val headDim = metadata.ropeDimensionCount ?: (dim / nHeads)
    val eps = 1e-5f

    return sequential<T, V> {
        val dslImpl = this as NeuralNetworkDslImpl<T, V>

        val nnCtx = DefaultNeuralNetworkExecutionContext()
        for (layer in 0 until nLayers) {
            val stage = StageImpl<T, V>(nnCtx, "acoustic.blk.$layer", T::class)
            stage.rmsNorm(dim, eps, id = "attn_norm")
            stage.multiHeadAttention(
                dim = dim,
                nHeads = nHeads,
                nKVHeads = nKVHeads,
                causal = true,
                id = "attn"
            ) {
                rope(headDim, seqLen, base = ropeBase)
                kvCache(seqLen, nKVHeads, headDim)
            }
            stage.residual()

            stage.rmsNorm(dim, eps, id = "ffn_norm")
            stage.swiGluFFN(dim, ffnDim, id = "ffn")
            stage.residual()

            dslImpl.modules += HybridTransformerBlock(stage.modules.toList(), name = "acoustic.blk.$layer")
        }

        dslImpl.rmsNorm(dim, eps, id = "acoustic.output_norm")
    }
}
