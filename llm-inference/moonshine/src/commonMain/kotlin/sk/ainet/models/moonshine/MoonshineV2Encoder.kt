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
 * CONFIRMED (2026-07-24) against the real model — `moonshine-voice` `medium-streaming`
 * `streaming_config.json` (obtain via `uv run moonshine-voice download --stt --language en`): the pipeline is
 * frontend → encoder → **adapter** → cross_kv + decoder_kv (exactly as authored). Real medium dims:
 * `encoder_dim=768, depth=14, nheads=10, head_dim=64, decoder_dim=640, vocab_size=32768`. The (16,4)
 * lookahead layers are the **first + last two** encoder layers ((16,0) intermediate) — fixed in
 * [rightContextForLayer] below (was a trailing-layers guess). The numeric defaults here stay a small
 * placeholder pending the **tiny-streaming** `streaming_config.json` (the SL2610-scale variant) and ONNX
 * confirmation of the encoder head geometry (note `encoder_dim` 768 ≠ `nheads×head_dim` 640 — the encoder
 * uses a projection or a distinct head config). `vocab_size` is confirmed below.
 */
public data class MoonshineV2Config(
    val dim: Int = 288,
    val encoderLayers: Int = 6,
    val nHeads: Int = 8,
    val headDim: Int = 36,          // 8 * 36 = 288
    val ffnDim: Int = 1152,         // 4 * dim (v2 FFN width not in streaming_config.json — confirm via ONNX)
    val vocabSize: Int = 32768,     // confirmed: real v2 vocab_size
    val layerNormEps: Float = 1e-5f,
    /** Left context: frames back each query attends to (inclusive). The paper's "16". */
    val slidingWindow: Int = 16,
    /** Right context (bounded lookahead) for the edge layers. The paper's "4" (the (16,4) layers). */
    val lookahead: Int = 4,
    /**
     * Number of layers **at each end** that use bounded lookahead ("(16,4)"); the intermediate layers are
     * strictly causal ("(16,0)"). The v2 paper + real model: the **first two AND last two** encoder layers
     * are (16,4). (Was previously a wrong "trailing layers" guess.)
     */
    val lookaheadEdgeLayers: Int = 2,
) {
    /**
     * Right context for [layer]: [lookahead] for the first / last [lookaheadEdgeLayers] layers, else 0 —
     * matching the v2 paper ((16,4) on the first + last two encoder layers; (16,0) intermediate).
     */
    public fun rightContextForLayer(layer: Int): Int =
        if (layer < lookaheadEdgeLayers || layer >= encoderLayers - lookaheadEdgeLayers) lookahead else 0
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
