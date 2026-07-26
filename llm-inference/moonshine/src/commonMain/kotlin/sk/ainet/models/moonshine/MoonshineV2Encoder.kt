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
 * CONFIRMED (2026-07-24) against the real **tiny-streaming** model (the SL2610-scale variant), obtained via
 * `uv run moonshine-voice download` + the float ONNX graphs at
 * `download.moonshine.ai/model/tiny-streaming-en/{float,quantized}/`. The pipeline is
 * frontend → encoder → **adapter** → cross_kv + decoder_kv (exactly as authored). The defaults below are the
 * real tiny numbers from `streaming_config.json` + the encoder ONNX:
 * `encoder_dim=320, depth=6, nheads=8, head_dim=40 (8×40=320), vocab_size=32768`, and the FFN width read
 * from the graph (`blocks.N.ff.project_in` = 1280 = 4×dim). Attention is **bias-free and position-free**
 * (the graph has no RoPE — only a scalar 1/√d scale), and the encoder LayerNorms are **scale-only (no bias)**.
 * The (16,4) lookahead layers are the **first + last two** encoder layers ((16,0) intermediate) — see
 * [rightContextForLayer]. (medium-streaming is larger — `encoder_dim=768, depth=14, nheads=10, head_dim=64`,
 * where `encoder_dim 768 ≠ nheads×head_dim 640`; tiny is clean/consistent, so it's the demo target.)
 */
public data class MoonshineV2Config(
    val dim: Int = 320,             // tiny-streaming encoder_dim
    val encoderLayers: Int = 6,     // tiny-streaming depth
    val nHeads: Int = 8,            // tiny-streaming nheads
    val headDim: Int = 40,          // 8 * 40 = 320 (tiny-streaming head_dim)
    val ffnDim: Int = 1280,         // 4 * dim — confirmed from the encoder ONNX (blocks.N.ff.project_in = 1280)
    val vocabSize: Int = 32768,     // confirmed: real v2 vocab_size
    val layerNormEps: Float = 1e-5f,
    /**
     * Left context window. CORRECTED to 17 (2026-07-26) after a direct DSL-vmfb-vs-onnxruntime comparison:
     * the real encoder's attention band is `-4 ≤ (i−j) ≤ 16` i.e. `j ∈ [i−16, i+4]` (extracted from the ONNX
     * mask: `GreaterOrEqual/LessOrEqual` on `i−j`). The DSL's `buildSlidingWindowMask` band is `[q−w+1, q+r]`,
     * so reaching `i−16` needs `w = 17` (16 gave `j ≥ i−15`, an off-by-one → cos 0.991 not 1.0). With `w=17`
     * the numpy reimplementation of this encoder matches onnxruntime **cos = 1.000000** (bit-exact).
     */
    val slidingWindow: Int = 17,
    /** Right context (bounded lookahead) for the edge layers. The paper's "4" (the (16,4) layers). */
    val lookahead: Int = 4,
    /**
     * Number of layers **at each end** that use bounded lookahead ("(16,4)"); the intermediate layers are
     * strictly causal ("(16,0)"). The v2 paper + real model: the **first two AND last two** encoder layers
     * are (16,4). (Was previously a wrong "trailing layers" guess.)
     */
    val lookaheadEdgeLayers: Int = 2,
    // --- decoder (see [moonshineV2Decoder]) — confirmed against tiny-streaming decoder_kv.onnx ---
    /** Decoder depth (`decoder_kv` has 6 layers; = encoder depth here). */
    val decoderLayers: Int = 6,
    /** RoPE base for the decoder self-attention. */
    val ropeBase: Float = 10000.0f,
    /**
     * Partial-rotary factor: `rotaryDim = headDim * factor`. The real decoder's `rotary.inv_freq` has 16
     * entries ⇒ rotaryDim=32; `32 / headDim(40) = 0.8` (the trailing 8 head dims pass through, as in v1).
     */
    val partialRotaryFactor: Float = 0.8f,
    /** Decoder RoPE table size (max decode positions). */
    val maxDecodeTokens: Int = 512,
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
