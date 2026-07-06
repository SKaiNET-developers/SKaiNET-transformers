package sk.ainet.models.moonshine

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.DType
import java.io.File
import kotlin.reflect.KClass

/**
 * Phase B — map/bake the Moonshine-tiny DECODER weights into [MoonshineDecoderModel].
 *
 * Source of truth is the HF safetensors checkpoint (`UsefulSensors/moonshine-tiny`), converted
 * to per-tensor little-endian f32 `.bin` files by `convert_moonshine_weights.py` (raw HF layout,
 * `model.` prefix stripped). The DSL projects through
 * [sk.ainet.lang.nn.transformer.linearProject] = `x @ Wᵀ` with `W` in `[out, in]` — identical to
 * HF `nn.Linear` — so raw HF `[out, in]` weights map **without transpose** (this differs from the
 * encoder's ONNX-sourced mapping, whose initializers are `[in, out]` and need transpose=true).
 *
 * Verified against the checkpoint header (6 layers, dim 288, ffn 1152, vocab 32768):
 *   self_attn / encoder_attn q,k,v,o .weight   [288,288]   (attention_bias=false → no bias)
 *   mlp.fc1.weight [2304,288] + .bias [2304]    (fused value|gate)
 *   mlp.fc2.weight [288,1152] + .bias [288]
 *   input_layernorm / post_attention_layernorm / final_layernorm .weight [288]  (bias=false)
 *   decoder.norm.weight [288];  decoder.embed_tokens.weight [32768,288]  (proj_out is TIED to it)
 *
 * The DSL LayerNorm carries a bias param; Moonshine LayerNorm has none, so bias maps to zeros
 * (returned by [zerosFor]). `lm_head.weight` is fed `embed_tokens` (the tie).
 */
internal interface DecWeightSource {
    /** Flat row-major f32 values for [hfName], or null if absent. */
    fun weight(hfName: String): FloatArray?
}

/** Reads `$dir/<hfName>.bin` as little-endian f32 (matches `convert_moonshine_weights.py`). */
internal class DecDirBinWeightSource(private val dir: String) : DecWeightSource {
    override fun weight(hfName: String): FloatArray? {
        val f = File(dir, "$hfName.bin")
        if (!f.exists()) return null
        val bytes = f.readBytes()
        return FloatArray(bytes.size / 4) { i ->
            var bits = 0
            for (b in 0 until 4) bits = bits or ((bytes[i * 4 + b].toInt() and 0xFF) shl (8 * b))
            Float.fromBits(bits)
        }
    }
}

private val DEC_LAYER = Regex("""^dec\.(\d+)\.(.+)$""")

/**
 * Map a DSL decoder param name → (HF tensor name, transpose?). Raw HF `[out,in]` weights need no
 * transpose (see file header). `null` HF name means "synthesize zeros of the param's shape" (the
 * absent LayerNorm biases). Returns `null` for a genuinely unmapped param (fail-fast upstream).
 */
internal fun decoderHfNameFor(dslName: String): DecMap? {
    when (dslName) {
        "dec_out_norm.weight" -> return DecMap("decoder.norm.weight", false)
        "dec_out_norm.bias" -> return DecMap(null, false) // Moonshine final norm has no bias
        // proj_out is tied to embed_tokens — feed the embedding table as the lm_head weight.
        "lm_head.weight" -> return DecMap("decoder.embed_tokens.weight", false)
        "lm_head.bias" -> return DecMap(null, false) // proj_out has bias=false
    }
    val m = DEC_LAYER.matchEntire(dslName) ?: return null
    val l = m.groupValues[1]
    return when (m.groupValues[2]) {
        // norms (weight only; bias → zeros)
        "self_attn_norm.weight" -> DecMap("decoder.layers.$l.input_layernorm.weight", false)
        "self_attn_norm.bias" -> DecMap(null, false)
        "cross_attn_norm.weight" -> DecMap("decoder.layers.$l.post_attention_layernorm.weight", false)
        "cross_attn_norm.bias" -> DecMap(null, false)
        "mlp_norm.weight" -> DecMap("decoder.layers.$l.final_layernorm.weight", false)
        "mlp_norm.bias" -> DecMap(null, false)
        // causal self-attention projections (no bias)
        "self_attn.q_proj.weight" -> DecMap("decoder.layers.$l.self_attn.q_proj.weight", false)
        "self_attn.k_proj.weight" -> DecMap("decoder.layers.$l.self_attn.k_proj.weight", false)
        "self_attn.v_proj.weight" -> DecMap("decoder.layers.$l.self_attn.v_proj.weight", false)
        "self_attn.o_proj.weight" -> DecMap("decoder.layers.$l.self_attn.o_proj.weight", false)
        // cross-attention → encoder memory (HF `encoder_attn`)
        "cross_attn.q_proj.weight" -> DecMap("decoder.layers.$l.encoder_attn.q_proj.weight", false)
        "cross_attn.k_proj.weight" -> DecMap("decoder.layers.$l.encoder_attn.k_proj.weight", false)
        "cross_attn.v_proj.weight" -> DecMap("decoder.layers.$l.encoder_attn.v_proj.weight", false)
        "cross_attn.o_proj.weight" -> DecMap("decoder.layers.$l.encoder_attn.o_proj.weight", false)
        // gated SiLU MLP (fused fc1 emits value|gate; both fc1/fc2 biased)
        "mlp_fc1.weight" -> DecMap("decoder.layers.$l.mlp.fc1.weight", false)
        "mlp_fc1.bias" -> DecMap("decoder.layers.$l.mlp.fc1.bias", false)
        "mlp_fc2.weight" -> DecMap("decoder.layers.$l.mlp.fc2.weight", false)
        "mlp_fc2.bias" -> DecMap("decoder.layers.$l.mlp.fc2.bias", false)
        else -> null
    }
}

internal data class DecMap(val hfName: String?, val transpose: Boolean)

private val ENC_LAYER = Regex("""^enc\.(\d+)\.(.+)$""")

/**
 * Encoder HF-safetensors mapping (raw `[out,in]` → transpose=false, same reasoning as the decoder).
 * NOTE this differs from the demo's `MoonshineWeights.hfNameFor`, which targets the ONNX-extracted
 * source (`[in,out]`, transpose=true) and uses `self_attn_layer_norm`/`fc1` names. The HF safetensors
 * names are `input_layernorm`/`post_attention_layernorm`/`mlp.fc1` (weight-only LNs, no attn bias).
 */
internal fun encoderHfNameFor(dslName: String): DecMap? {
    when (dslName) {
        "enc_out_norm.weight" -> return DecMap("encoder.layer_norm.weight", false)
        "enc_out_norm.bias" -> return DecMap(null, false)
    }
    val m = ENC_LAYER.matchEntire(dslName) ?: return null
    val l = m.groupValues[1]
    return when (m.groupValues[2]) {
        "attn_norm.weight" -> DecMap("encoder.layers.$l.input_layernorm.weight", false)
        "attn_norm.bias" -> DecMap(null, false)
        "attn.q_proj.weight" -> DecMap("encoder.layers.$l.self_attn.q_proj.weight", false)
        "attn.k_proj.weight" -> DecMap("encoder.layers.$l.self_attn.k_proj.weight", false)
        "attn.v_proj.weight" -> DecMap("encoder.layers.$l.self_attn.v_proj.weight", false)
        "attn.o_proj.weight" -> DecMap("encoder.layers.$l.self_attn.o_proj.weight", false)
        "ffn_norm.weight" -> DecMap("encoder.layers.$l.post_attention_layernorm.weight", false)
        "ffn_norm.bias" -> DecMap(null, false)
        "ffn_up.weight" -> DecMap("encoder.layers.$l.mlp.fc1.weight", false)
        "ffn_up.bias" -> DecMap("encoder.layers.$l.mlp.fc1.bias", false)
        "ffn_down.weight" -> DecMap("encoder.layers.$l.mlp.fc2.weight", false)
        "ffn_down.bias" -> DecMap("encoder.layers.$l.mlp.fc2.bias", false)
        else -> null
    }
}

private fun <T : DType, V> walkParams(
    m: Module<T, V>,
    out: MutableList<ModuleParameter<*, *>> = mutableListOf(),
): List<ModuleParameter<*, *>> {
    out.addAll(m.params)
    for (child in m.modules) walkParams(child, out)
    return out
}

/** Overwrite every decoder parameter with real weights from [src]. Fail-fast on any gap. */
internal fun <T : DType, V> bakeDecoderWeights(
    model: Module<T, V>,
    src: DecWeightSource,
    dtypeClass: KClass<T>,
    ctx: ExecutionContext,
): Int = bakeMoonshineWeights(model, src, ::decoderHfNameFor, dtypeClass, ctx)

/** Overwrite every parameter of [model] using [mapper] (encoder or decoder). Fail-fast on any gap. */
internal fun <T : DType, V> bakeMoonshineWeights(
    model: Module<T, V>,
    src: DecWeightSource,
    mapper: (String) -> DecMap?,
    dtypeClass: KClass<T>,
    ctx: ExecutionContext,
): Int {
    var baked = 0
    for (p in walkParams(model)) {
        val map = mapper(p.name)
            ?: error("no HF mapping for DSL param '${p.name}'")
        val shape = p.value.shape.dimensions
        val nElems = shape.fold(1) { a, b -> a * b }

        var data = if (map.hfName == null) {
            FloatArray(nElems) // synthesized zeros (absent LayerNorm biases)
        } else {
            src.weight(map.hfName) ?: error("checkpoint missing tensor '${map.hfName}' for '${p.name}'")
        }
        if (map.transpose && shape.size == 2) data = transpose2d(data, rows = shape[1], cols = shape[0])
        require(data.size == nElems) {
            "tensor '${map.hfName}' size ${data.size} != DSL param '${p.name}' shape ${shape.toList()}"
        }

        @Suppress("UNCHECKED_CAST")
        val tensor = ctx.fromData(
            DenseFloatArrayTensorData<T>(Shape(*shape), data) as TensorData<T, V>,
            dtypeClass,
        )
        @Suppress("UNCHECKED_CAST")
        (p as ModuleParameter<T, V>).value = tensor
        baked++
    }
    return baked
}

private fun transpose2d(src: FloatArray, rows: Int, cols: Int): FloatArray {
    val dst = FloatArray(src.size)
    for (r in 0 until rows) for (c in 0 until cols) dst[c * rows + r] = src[r * cols + c]
    return dst
}
