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
 * The network is **dtype-portable**: it is parameterized on the element type [T], so the SAME
 * graph lowers to any IREE target. Choose the dtype to match the target, not the model:
 *   - `FP32` — portable host/GPU builds (llvm-cpu AVX, CUDA/Vulkan). The recommended default for reuse.
 *   - `BF16` — the Torq NPU, where the weights must stay bf16 AT THE matmul (fp32 weights crash the
 *     torq compiler's `getWeightMemoryFormat`, proven in the demo's `docs/torq-npu-weight-crash.md`).
 * So bf16 is a **target choice**, not a property of this model — see this module's `README.md` for the
 * standalone-reuse + multi-target story.
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

        // x + Attn(LN(x)). Layer-qualify every id ("enc.$layer.*") so the attention /
        // LayerNorm parameter names are UNIQUE across layers — matching the FFN naming below.
        // Without the prefix, `attn.q_proj.weight` / `attn_norm.weight` repeat every layer and
        // by-name weight loading can't tell the layers apart.
        stage.layerNorm(intArrayOf(dim), eps.toDouble(), id = "enc.$layer.attn_norm")
        stage.multiHeadAttention(
            dim = dim,
            nHeads = cfg.nHeads,
            nKVHeads = cfg.nHeads,
            causal = false, // encoder = bidirectional self-attention
            bias = false,
            id = "enc.$layer.attn",
        ) {
            rope(
                headDim = cfg.headDim,
                maxSeqLen = cfg.maxFrames,
                mode = RoPEMode.INTERLEAVED, // Moonshine rope = adjacent-pair (verified bit-exact vs ONNX Mul_4)
                base = cfg.ropeBase,
                partialRotaryFactor = cfg.partialRotaryFactor,
                freqDenomRotaryDim = true, // Moonshine inv_freq uses rotaryDim (32), not headDim (36)
            )
        }
        stage.residual()

        // x + MLP(LN(x)) — plain GELU MLP (not gated). VoidDense carries explicit
        // in/out dims (no eager placeholder alloc, and no reliance on DSL
        // dimension-tracking through the attention/residual sub-blocks).
        stage.layerNorm(intArrayOf(dim), eps.toDouble(), id = "enc.$layer.ffn_norm")
        // Moonshine MLP is a biased fc1 -> GELU -> biased fc2 (the reference model
        // carries `mlp.fc1.bias`/`mlp.fc2.bias`). addBias=true keeps the trace faithful.
        stage.modules += VoidDense<T, V>("enc.$layer.ffn_up", cfg.ffnDim, dim, dtype, addBias = true)
        stage.modules += GELU<T, V>(name = "enc.$layer.ffn_gelu")
        stage.modules += VoidDense<T, V>("enc.$layer.ffn_down", dim, cfg.ffnDim, dtype, addBias = true)
        stage.residual()

        dsl.modules += HybridTransformerBlock(stage.modules.toList(), name = "enc.$layer")
    }

    dsl.layerNorm(intArrayOf(dim), eps.toDouble(), id = "enc_out_norm")
    return dsl.create()
}
