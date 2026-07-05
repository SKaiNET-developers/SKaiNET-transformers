package sk.ainet.models.moonshine

import sk.ainet.apps.llm.HybridTransformerBlock
import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.activations.GELU
import sk.ainet.lang.nn.dsl.NeuralNetworkDslImpl
import sk.ainet.lang.nn.dsl.StageImpl
import sk.ainet.lang.nn.dsl.multiHeadAttention
import sk.ainet.lang.nn.dsl.residual
import sk.ainet.lang.nn.transformer.RoPEMode
import sk.ainet.lang.nn.transformer.VoidDense
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Moonshine-tiny ENCODER transformer stack, authored in the SKaiNET NN DSL.
 *
 * Input is the conv-frontend output `[batch, frames, dim]` (the audio frontend is
 * built separately, see [MoonshineConfig]); output is the encoder memory
 * `[batch, frames, dim]` for the decoder's cross-attention.
 *
 * Each layer is pre-norm:  x + Attn(LN(x));  x + MLP(LN(x))  with
 * non-causal RoPE self-attention (no bias) and a plain GELU MLP (up→gelu→down).
 *
 * The network is parameterized on the element type [T]; build it with `BF16` so
 * the DSL→StableHLO export keeps bf16 weights at the matmul — the Torq NPU
 * requirement proven in the demo's `docs/torq-npu-weight-crash.md` (fp32 weights
 * crash the torq compiler's `getWeightMemoryFormat`).
 */
public fun <T : DType, V> moonshineEncoder(
    cfg: MoonshineConfig,
    dtype: KClass<T>,
): Module<T, V> {
    val dim = cfg.dim
    val eps = cfg.layerNormEps
    val nnCtx = DefaultNeuralNetworkExecutionContext()
    val dsl = NeuralNetworkDslImpl<T, V>(nnCtx, dtype)

    for (layer in 0 until cfg.encoderLayers) {
        val stage = StageImpl<T, V>(nnCtx, "enc.$layer", dtype)

        // x + Attn(LN(x))
        stage.layerNorm(intArrayOf(dim), eps.toDouble(), id = "attn_norm")
        stage.multiHeadAttention(
            dim = dim,
            nHeads = cfg.nHeads,
            nKVHeads = cfg.nHeads,
            causal = false, // encoder = bidirectional self-attention
            bias = false,
            id = "attn",
        ) {
            rope(headDim = cfg.headDim, maxSeqLen = cfg.maxFrames, mode = RoPEMode.INTERLEAVED, base = cfg.ropeBase)
        }
        stage.residual()

        // x + MLP(LN(x)) — plain GELU MLP (not gated). VoidDense carries explicit
        // in/out dims (no eager placeholder alloc, and no reliance on DSL
        // dimension-tracking through the attention/residual sub-blocks).
        stage.layerNorm(intArrayOf(dim), eps.toDouble(), id = "ffn_norm")
        stage.modules += VoidDense<T, V>("enc.$layer.ffn_up", cfg.ffnDim, dim, dtype)
        stage.modules += GELU<T, V>(name = "enc.$layer.ffn_gelu")
        stage.modules += VoidDense<T, V>("enc.$layer.ffn_down", dim, cfg.ffnDim, dtype)
        stage.residual()

        dsl.modules += HybridTransformerBlock(stage.modules.toList(), name = "enc.$layer")
    }

    dsl.layerNorm(intArrayOf(dim), eps.toDouble(), id = "enc_out_norm")
    return dsl.create()
}
