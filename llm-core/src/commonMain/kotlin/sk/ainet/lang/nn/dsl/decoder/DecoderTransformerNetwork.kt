package sk.ainet.lang.nn.dsl.decoder

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
import sk.ainet.lang.nn.dsl.swiGluFFN
import sk.ainet.lang.nn.transformer.RoPEMode
import sk.ainet.lang.nn.transformer.VoidDense
import sk.ainet.lang.types.DType

/**
 * Architecture-neutral decoder-only transformer body builder.
 *
 * Architecture: `Embedding → N × (RMSNorm → MHA(RoPE, KVCache, [QK-norm]) →
 * Residual → RMSNorm → SwiGLU FFN → Residual) → RMSNorm → output Dense`.
 *
 * Each model's `xNetwork(metadata)` function should call this with that
 * model's specific knobs. Per-model `*NetworkDef.kt` files are expected to
 * be thin (≤ 10 lines) — all composition happens here.
 *
 * Currently FFN is hardcoded to SwiGLU (Llama / Qwen / Mistral / Gemma all
 * use it). When a model with a different FFN variant lands, add an
 * `ffnKind` parameter rather than copying this builder.
 *
 * @param metadata model shape (dim, heads, layers, vocab, …)
 * @param ropeBase RoPE rotary frequency base. Llama: 10_000, Qwen3: 1_000_000.
 * @param eps RMSNorm epsilon. Typically 1e-5 (Llama) or 1e-6 (Qwen3).
 * @param qkNorm whether to apply per-head RMSNorm to Q and K post-projection.
 *   Qwen3: true. Llama / Mistral: false.
 * @param qkNormUnitOffset whether the QK-norm uses unit-offset weights
 *   (Gemma-style). Defaults to false.
 * @param ropeMode pairing convention for RoPE rotation. [RoPEMode.INTERLEAVED]
 *   (`(buf[2i], buf[2i+1])`, llama.cpp NORM mode 0) is the LLaMA / Mistral /
 *   Gemma default. [RoPEMode.SPLIT_HALF] (`(buf[i], buf[i+ropeDim/2])`,
 *   llama.cpp NEOX mode 2) is what Qwen 2/3, Phi, Falcon, StableLM,
 *   Starcoder2 store in GGUF. Mirrors `RopeType.forArchitecture()` in
 *   [sk.ainet.apps.llm.RopeUtils] — picking the wrong mode produces
 *   subtly-wrong logits per token (correct-by-accident at very small
 *   contexts; compounds across positions).
 * @param maxInferenceLen sequence length used to size the KV cache and RoPE
 *   tables. Capped at min(metadata.contextLength, 4096) by default.
 */
public inline fun <reified T : DType, V> decoderTransformerNetwork(
    metadata: DecoderModelMetadata,
    ropeBase: Float = metadata.ropeFreqBase,
    eps: Float = metadata.rmsNormEps,
    qkNorm: Boolean = false,
    qkNormUnitOffset: Boolean = false,
    ropeMode: RoPEMode = RoPEMode.INTERLEAVED,
    maxInferenceLen: Int = minOf(metadata.contextLength, 4096),
): Module<T, V> {
    val dim = metadata.embeddingLength
    val nHeads = metadata.headCount
    val nKVHeads = metadata.kvHeadCount
    val nLayers = metadata.blockCount
    val ffnDim = metadata.feedForwardLength
    val seqLen = maxInferenceLen
    val vocabSize = metadata.vocabSize
    val headDim = metadata.ropeDimensionCount ?: (dim / nHeads)

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
                qkNorm = qkNorm,
                qkNormUnitOffset = qkNormUnitOffset,
                qkNormEps = eps,
                id = "attn",
            ) {
                rope(headDim, seqLen, mode = ropeMode, base = ropeBase)
                kvCache(seqLen, nKVHeads, headDim)
            }
            stage.residual()

            stage.rmsNorm(dim, eps, id = "ffn_norm")
            stage.swiGluFFN(dim, ffnDim, id = "ffn")
            stage.residual()

            dslImpl.modules += HybridTransformerBlock(stage.modules.toList(), name = "blk.$layer")
        }

        dslImpl.rmsNorm(dim, eps, id = "output_norm")
        // VoidDense placeholder — avoids allocating a [vocabSize, dim] zero
        // tensor before WeightMapper binds the real `output.weight`.
        dslImpl.modules += VoidDense<T, V>("output", vocabSize, dim)
    }
}
