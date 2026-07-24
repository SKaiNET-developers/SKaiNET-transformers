package sk.ainet.models.moonshine

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Map/bake the Moonshine **v2** tiny-streaming ENCODER (+ adapter) weights into [moonshineV2Encoder] /
 * [MoonshineV2Adapter]. Parallel to [encoderHfNameFor] (v1), but the source is the **v2 float ONNX**
 * baked by `scripts/convert_moonshine_v2_weights.py` (per-tensor little-endian f32 `.bin` + manifest),
 * so it reuses the same infra ([DecWeightSource], [DecDirBinWeightSource], [DecMap], [bakeMoonshineWeights]).
 *
 * Two v2-specific differences from the v1 HF-safetensors mapping:
 *  - **Orientation**: the baker keeps the raw ONNX linear orientation `[in, out]`, while the DSL
 *    `linearProject` wants `[out, in]` → every linear weight maps with **transpose=true** (v1's HF
 *    safetensors were already `[out, in]`, transpose=false). Norm scales / biases are 1-D (no transpose).
 *  - **Norms are scale-only**: the v2 encoder LayerNorms have no bias in the graph (the learned affine is
 *    a single `[dim]` scale), so `*_norm.bias` maps to `null` → zeros, exactly as v1's bias-free norms.
 *
 * The baker names its `.bin` files with the model-native scheme (`enc.$l.attn.q.weight`,
 * `enc.$l.ffn_up.{weight,bias}`, `enc_out_norm.weight`, `v2_adapter.pos_embed.weight`); this mapper turns
 * each **DSL param name** (`p.name`, e.g. `enc.$l.attn.q_proj.weight`) into that source name. A single
 * mapper covers both the encoder (`enc.*`, `enc_out_norm.*`) and the adapter (`v2_adapter.*`) namespaces,
 * so it bakes either module. Unmapped params return `null` → [bakeMoonshineWeights] fail-fasts.
 */
private val V2_ENC_LAYER = Regex("""^enc\.(\d+)\.(.+)$""")

internal fun v2EncoderSrcNameFor(dslName: String): DecMap? {
    when (dslName) {
        "enc_out_norm.weight" -> return DecMap("enc_out_norm.weight", false)
        "enc_out_norm.bias" -> return DecMap(null, false)            // scale-only norm
        // adapter: learned absolute positional embedding [maxFrames, dim] — Gather, no transpose.
        "v2_adapter.pos_embed.weight" -> return DecMap("v2_adapter.pos_embed.weight", false)
    }
    val m = V2_ENC_LAYER.matchEntire(dslName) ?: return null
    val l = m.groupValues[1]
    return when (m.groupValues[2]) {
        // pre-attention / pre-FFN norms: learned [dim] scale, no bias (→ zeros).
        "attn_norm.weight" -> DecMap("enc.$l.attn_norm.weight", false)
        "attn_norm.bias" -> DecMap(null, false)
        "ffn_norm.weight" -> DecMap("enc.$l.ffn_norm.weight", false)
        "ffn_norm.bias" -> DecMap(null, false)
        // bias-free, position-free attention projections — ONNX [in,out] → transpose to [out,in].
        "attn.q_proj.weight" -> DecMap("enc.$l.attn.q.weight", true)
        "attn.k_proj.weight" -> DecMap("enc.$l.attn.k.weight", true)
        "attn.v_proj.weight" -> DecMap("enc.$l.attn.v.weight", true)
        "attn.o_proj.weight" -> DecMap("enc.$l.attn.o.weight", true)
        // GELU MLP — weights transpose; biases are 1-D (kept in the graph).
        "ffn_up.weight" -> DecMap("enc.$l.ffn_up.weight", true)
        "ffn_up.bias" -> DecMap("enc.$l.ffn_up.bias", false)
        "ffn_down.weight" -> DecMap("enc.$l.ffn_down.weight", true)
        "ffn_down.bias" -> DecMap("enc.$l.ffn_down.bias", false)
        else -> null
    }
}

/** Overwrite every v2 encoder/adapter parameter with real weights from [src]. Fail-fast on any gap. */
internal fun <T : DType, V> bakeV2EncoderWeights(
    model: Module<T, V>,
    src: DecWeightSource,
    dtypeClass: KClass<T>,
    ctx: ExecutionContext,
): Int = bakeMoonshineWeights(model, src, ::v2EncoderSrcNameFor, dtypeClass, ctx)
