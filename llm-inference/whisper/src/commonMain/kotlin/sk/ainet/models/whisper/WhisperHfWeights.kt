package sk.ainet.models.whisper

import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.safetensors.StreamingSafeTensorsReader
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.DType
import kotlin.math.exp
import kotlin.math.ln
import kotlin.reflect.KClass

/**
 * Bake HF whisper safetensors weights (e.g. `primeline/whisper-tiny-german-1224`)
 * into the DSL encoder/decoder. All-Kotlin — no python/ONNX anywhere.
 *
 * The DSL projects via `linearProject` = `x @ Wᵀ` with `W [out, in]`, identical to
 * HF `nn.Linear`, so weights map WITHOUT transpose. `k_proj` carries no bias in the
 * checkpoint and the DSL k-projection is built bias-free — the mapping simply has
 * no entry. The token embedding is TIED: `embed_tokens.weight` feeds both the
 * gather table and the output projection (one tensor end to end).
 *
 * `enc_pos.weight` is the checkpoint's `model.encoder.embed_positions.weight`
 * sliced to `[0:audioCtx]` — Whisper encoder positions are a deterministic sinusoid,
 * so the slice IS the short-context table. [sinusoids] recomputes them for the
 * diagnostic parity warning (a finetune that touched them would surface here).
 */
public interface WhisperWeightSource {
    /** Flat row-major f32 values for [hfName], or null if absent. */
    public fun weight(hfName: String): FloatArray?
}

/** Streams tensors by name straight from an HF `model.safetensors`. */
public class SafeTensorsWeightSource(
    sourceProvider: () -> RandomAccessSource,
) : WhisperWeightSource, AutoCloseable {
    private val reader = StreamingSafeTensorsReader.open(sourceProvider())

    override fun weight(hfName: String): FloatArray? {
        val info = reader.tensors.firstOrNull { it.name == hfName } ?: return null
        val bytes = try {
            reader.loadTensorData(hfName)
        } catch (_: Exception) {
            return null
        }
        return when (info.dtype.uppercase()) {
            "F32" -> FloatArray(bytes.size / 4) { i ->
                var bits = 0
                for (b in 0 until 4) bits = bits or ((bytes[i * 4 + b].toInt() and 0xFF) shl (8 * b))
                Float.fromBits(bits)
            }
            "F16" -> FloatArray(bytes.size / 2) { i ->
                f16ToF32((bytes[i * 2].toInt() and 0xFF) or ((bytes[i * 2 + 1].toInt() and 0xFF) shl 8))
            }
            "BF16" -> FloatArray(bytes.size / 2) { i ->
                val hi = (bytes[i * 2].toInt() and 0xFF) or ((bytes[i * 2 + 1].toInt() and 0xFF) shl 8)
                Float.fromBits(hi shl 16)
            }
            else -> error("unsupported safetensors dtype '${info.dtype}' for '$hfName'")
        }
    }

    private fun f16ToF32(h: Int): Float {
        val sign = (h ushr 15) and 1
        val exp = (h ushr 10) and 0x1F
        val mant = h and 0x3FF
        val bits = when {
            exp == 0 -> if (mant == 0) sign shl 31 else {
                // subnormal: normalize
                var m = mant
                var e = -1
                while (m and 0x400 == 0) { m = m shl 1; e++ }
                ((sign shl 31) or ((127 - 15 - e) shl 23) or ((m and 0x3FF) shl 13))
            }
            exp == 0x1F -> (sign shl 31) or 0x7F800000 or (mant shl 13)
            else -> (sign shl 31) or ((exp - 15 + 127) shl 23) or (mant shl 13)
        }
        return Float.fromBits(bits)
    }

    override fun close(): Unit = reader.close()
}

private val DEC_LAYER = Regex("""^dec\.(\d+)\.(.+)$""")
private val ENC_LAYER = Regex("""^enc\.(\d+)\.(.+)$""")

/** DSL param name → HF tensor name (never transposed; see file header). */
public fun whisperHfNameFor(dslName: String): String? {
    when (dslName) {
        "embed_tokens.weight" -> return "model.decoder.embed_tokens.weight"
        "pos_embed.weight" -> return "model.decoder.embed_positions.weight"
        "dec_ln.weight" -> return "model.decoder.layer_norm.weight"
        "dec_ln.bias" -> return "model.decoder.layer_norm.bias"
        "enc_conv1.weight" -> return "model.encoder.conv1.weight"
        "enc_conv1.bias" -> return "model.encoder.conv1.bias"
        "enc_conv2.weight" -> return "model.encoder.conv2.weight"
        "enc_conv2.bias" -> return "model.encoder.conv2.bias"
        "enc_pos.weight" -> return "model.encoder.embed_positions.weight" // sliced to audioCtx rows
        "enc_ln_post.weight" -> return "model.encoder.layer_norm.weight"
        "enc_ln_post.bias" -> return "model.encoder.layer_norm.bias"
    }
    DEC_LAYER.matchEntire(dslName)?.let { m ->
        val l = m.groupValues[1]
        val hf = "model.decoder.layers.$l"
        return when (m.groupValues[2]) {
            "self_attn_norm.weight" -> "$hf.self_attn_layer_norm.weight"
            "self_attn_norm.bias" -> "$hf.self_attn_layer_norm.bias"
            "cross_attn_norm.weight" -> "$hf.encoder_attn_layer_norm.weight"
            "cross_attn_norm.bias" -> "$hf.encoder_attn_layer_norm.bias"
            "mlp_norm.weight" -> "$hf.final_layer_norm.weight"
            "mlp_norm.bias" -> "$hf.final_layer_norm.bias"
            "self_attn.q_proj.weight" -> "$hf.self_attn.q_proj.weight"
            "self_attn.q_proj.bias" -> "$hf.self_attn.q_proj.bias"
            "self_attn.k_proj.weight" -> "$hf.self_attn.k_proj.weight"
            "self_attn.v_proj.weight" -> "$hf.self_attn.v_proj.weight"
            "self_attn.v_proj.bias" -> "$hf.self_attn.v_proj.bias"
            "self_attn.o_proj.weight" -> "$hf.self_attn.out_proj.weight"
            "self_attn.o_proj.bias" -> "$hf.self_attn.out_proj.bias"
            "cross_attn.q_proj.weight" -> "$hf.encoder_attn.q_proj.weight"
            "cross_attn.q_proj.bias" -> "$hf.encoder_attn.q_proj.bias"
            "cross_attn.k_proj.weight" -> "$hf.encoder_attn.k_proj.weight"
            "cross_attn.v_proj.weight" -> "$hf.encoder_attn.v_proj.weight"
            "cross_attn.v_proj.bias" -> "$hf.encoder_attn.v_proj.bias"
            "cross_attn.o_proj.weight" -> "$hf.encoder_attn.out_proj.weight"
            "cross_attn.o_proj.bias" -> "$hf.encoder_attn.out_proj.bias"
            "fc1.weight" -> "$hf.fc1.weight"
            "fc1.bias" -> "$hf.fc1.bias"
            "fc2.weight" -> "$hf.fc2.weight"
            "fc2.bias" -> "$hf.fc2.bias"
            else -> null
        }
    }
    ENC_LAYER.matchEntire(dslName)?.let { m ->
        val l = m.groupValues[1]
        val hf = "model.encoder.layers.$l"
        return when (m.groupValues[2]) {
            "attn_norm.weight" -> "$hf.self_attn_layer_norm.weight"
            "attn_norm.bias" -> "$hf.self_attn_layer_norm.bias"
            "mlp_norm.weight" -> "$hf.final_layer_norm.weight"
            "mlp_norm.bias" -> "$hf.final_layer_norm.bias"
            "attn.q_proj.weight" -> "$hf.self_attn.q_proj.weight"
            "attn.q_proj.bias" -> "$hf.self_attn.q_proj.bias"
            "attn.k_proj.weight" -> "$hf.self_attn.k_proj.weight"
            "attn.v_proj.weight" -> "$hf.self_attn.v_proj.weight"
            "attn.v_proj.bias" -> "$hf.self_attn.v_proj.bias"
            "attn.o_proj.weight" -> "$hf.self_attn.out_proj.weight"
            "attn.o_proj.bias" -> "$hf.self_attn.out_proj.bias"
            "fc1.weight" -> "$hf.fc1.weight"
            "fc1.bias" -> "$hf.fc1.bias"
            "fc2.weight" -> "$hf.fc2.weight"
            "fc2.bias" -> "$hf.fc2.bias"
            else -> null
        }
    }
    return null
}

private fun <T : DType, V> walkParams(
    m: Module<T, V>,
    out: MutableList<ModuleParameter<*, *>> = mutableListOf(),
): List<ModuleParameter<*, *>> {
    out.addAll(m.params)
    for (child in m.modules) walkParams(child, out)
    return out
}

/** Overwrite every parameter of [model] from [src]. Fail-fast on any gap. Returns count. */
public fun <T : DType, V> bakeWhisperWeights(
    model: Module<T, V>,
    src: WhisperWeightSource,
    cfg: WhisperConfig,
    dtypeClass: KClass<T>,
    ctx: ExecutionContext,
): Int {
    var baked = 0
    for (p in walkParams(model)) {
        val shape = p.value.shape.dimensions
        val nElems = shape.fold(1) { a, b -> a * b }
        // k_proj carries no bias in the checkpoint; a zero bias is numerically identical.
        val zeroSynth = p.name.endsWith("k_proj.bias")
        val hfName = if (zeroSynth) null else whisperHfNameFor(p.name)
            ?: error("no HF mapping for DSL param '${p.name}'")
        var data = if (hfName == null) FloatArray(nElems)
        else src.weight(hfName) ?: error("checkpoint missing tensor '$hfName' for '${p.name}'")
        if (p.name == "enc_pos.weight" && data.size > nElems) {
            // slice [0:audioCtx] rows of the sinusoid table; warn if the checkpoint
            // drifted from the analytic sinusoids (diagnostic only).
            data = data.copyOfRange(0, nElems)
            val analytic = sinusoids(cfg.audioCtx, cfg.dim)
            var maxDiff = 0f
            for (i in data.indices) {
                val d = kotlin.math.abs(data[i] - analytic[i])
                if (d > maxDiff) maxDiff = d
            }
            if (maxDiff > 1e-4f) {
                println("WARN: enc_pos deviates from analytic sinusoids (max |Δ| = $maxDiff) — finetuned positions?")
            }
        }
        require(data.size == nElems) {
            "tensor '$hfName' size ${data.size} != DSL param '${p.name}' shape ${shape.toList()}"
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

/** OpenAI whisper `sinusoids(length, channels)` — row-major `[length, channels]`. */
public fun sinusoids(length: Int, channels: Int, maxTimescale: Float = 10_000f): FloatArray {
    require(channels % 2 == 0)
    val half = channels / 2
    val logInc = ln(maxTimescale) / (half - 1)
    val out = FloatArray(length * channels)
    for (t in 0 until length) {
        for (i in 0 until half) {
            val scaled = t * exp(-logInc * i)
            out[t * channels + i] = kotlin.math.sin(scaled)
            out[t * channels + half + i] = kotlin.math.cos(scaled)
        }
    }
    return out
}
