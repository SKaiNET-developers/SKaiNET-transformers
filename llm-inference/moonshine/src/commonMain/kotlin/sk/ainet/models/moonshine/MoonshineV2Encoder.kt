package sk.ainet.models.moonshine

import sk.ainet.lang.nn.transformer.TransformerBlock
import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.activations.GELU
import sk.ainet.lang.nn.dsl.NeuralNetworkDslImpl
import sk.ainet.lang.nn.dsl.StageImpl
import sk.ainet.lang.nn.dsl.multiHeadAttention
import sk.ainet.lang.nn.dsl.residual
import sk.ainet.lang.nn.transformer.VoidDense
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Moonshine **v2** streaming encoder parameters (see the v2 paper,
 * huggingface.co/papers/2602.12241). The v2 encoder differs from v1 in two ways that make it
 * both **streamable** and (potentially) more NPU-tileable:
 *  - **position-free**: no RoPE / absolute position — the layers are translation-invariant, so the
 *    encoder can run incrementally over a rolling frame buffer.
 *  - **sliding-window local attention with bounded lookahead**: each frame attends to a fixed band
 *    `[pos - window + 1, pos + rightContext]` instead of the full O(T²) sequence. Layers with
 *    `rightContext = 0` are strictly causal (latency-bounding "(16,0)" layers); layers with
 *    `rightContext > 0` add a small future context ("(16,4)" layers).
 *
 * NOTE: the exact v2-tiny hyperparameters (dim/layers/heads, per-layer window & lookahead) and the
 * adapter layer are not yet confirmed against a released v2 checkpoint — the defaults below follow the
 * paper's described (16,4)/(16,0) scheme and reuse v1-tiny's width. Verify + bake once weights exist.
 */
public data class MoonshineV2Config(
    val dim: Int = 288,
    val encoderLayers: Int = 6,
    val nHeads: Int = 8,
    val headDim: Int = 36,          // 8 * 36 = 288
    val ffnDim: Int = 1152,         // 4 * dim
    val layerNormEps: Float = 1e-5f,
    /** Left context: frames back each query attends to (inclusive). The paper's "16". */
    val slidingWindow: Int = 16,
    /** Right context (bounded lookahead) for the non-causal layers. The paper's "4". */
    val lookahead: Int = 4,
    /**
     * Number of trailing layers that are strictly causal ("(16,0)") to bound the finalized-state
     * latency. The earlier layers use `lookahead` right-context ("(16,4)"). Confirm the exact pattern
     * against the v2 checkpoint.
     */
    val causalTailLayers: Int = 1,
) {
    /** Right context for [layer]: 0 (causal) for the trailing [causalTailLayers], else [lookahead]. */
    public fun rightContextForLayer(layer: Int): Int =
        if (layer >= encoderLayers - causalTailLayers) 0 else lookahead
}

/**
 * Moonshine **v2** streaming ENCODER stack, authored in the SKaiNET NN DSL.
 *
 * Same pre-norm shape as v1 (`x + Attn(LN(x)); x + MLP(LN(x))`, plain GELU MLP), but the attention is
 * **position-free** (no `rope` block) and uses a **bounded-lookahead sliding window** instead of full
 * bidirectional attention. Per layer, `rightContext = 0` ⇒ causal-left window; `rightContext > 0` ⇒
 * non-causal local band (requires the transformer-core `rightContext` knob added for v2).
 *
 * dtype-portable like the v1 models: `FP32` for host/GPU, `BF16` for the Torq NPU (see the module README).
 *
 * Not included here: the **adapter layer** that re-injects position for the position-aware decoder — it
 * sits between this encoder and `MoonshineDecoder`, and is a separate follow-up (needs the v2 checkpoint).
 */
public fun <T : DType, V> moonshineV2Encoder(
    cfg: MoonshineV2Config,
    dtype: KClass<T>,
): Module<T, V> {
    val dim = cfg.dim
    val eps = cfg.layerNormEps
    val nnCtx = DefaultNeuralNetworkExecutionContext()
    val dsl = NeuralNetworkDslImpl<T, V>(nnCtx, dtype)

    for (layer in 0 until cfg.encoderLayers) {
        val stage = StageImpl<T, V>(nnCtx, "enc.$layer", dtype)
        val rc = cfg.rightContextForLayer(layer)

        // x + Attn(LN(x)) — position-free (NO rope block) sliding-window local attention.
        stage.layerNorm(intArrayOf(dim), eps.toDouble(), id = "enc.$layer.attn_norm")
        stage.multiHeadAttention(
            dim = dim,
            nHeads = cfg.nHeads,
            nKVHeads = cfg.nHeads,
            causal = rc == 0,               // (16,0) layers are causal-left; (16,w) layers are non-causal
            bias = false,
            id = "enc.$layer.attn",
            slidingWindow = cfg.slidingWindow,
            rightContext = rc,
        )
        stage.residual()

        // x + MLP(LN(x)) — plain biased GELU MLP (unchanged from v1).
        stage.layerNorm(intArrayOf(dim), eps.toDouble(), id = "enc.$layer.ffn_norm")
        stage.modules += VoidDense<T, V>("enc.$layer.ffn_up", cfg.ffnDim, dim, dtype, addBias = true)
        stage.modules += GELU<T, V>(name = "enc.$layer.ffn_gelu")
        stage.modules += VoidDense<T, V>("enc.$layer.ffn_down", dim, cfg.ffnDim, dtype, addBias = true)
        stage.residual()

        dsl.modules += TransformerBlock(stage.modules.toList(), name = "enc.$layer")
    }

    dsl.layerNorm(intArrayOf(dim), eps.toDouble(), id = "enc_out_norm")
    return dsl.create()
}
