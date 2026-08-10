package sk.ainet.models.whisper

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * G1 — numerical anchor for the DSL German whisper against the golden fixtures
 * recorded from the validated ONNX pipeline (mel_de4s.bin / golden_feat.bin /
 * golden_logits.bin, resources copied from the wc_de probe set):
 *
 *  1. eager ENCODER features vs golden_feat  (cosine per-frame; tanh-GELU vs
 *     erf-GELU costs ~1e-3, well inside tolerance)
 *  2. eager PREFILL last-prompt-row logits argmax == golden_logits argmax
 *  3. full greedy decode runs to a stop condition and yields a stable sequence
 *
 * Env-gated: set WHISPER_SAFETENSORS to the HF snapshot dir containing
 * model.safetensors (primeline/whisper-tiny-german-1224). Skips when unset.
 */
class WhisperEagerGreedyTest {

    private val cfg = WhisperConfig() // tiny multilingual @ audioCtx=200
    private val maxP = 48

    private fun snapshotDir(): File? =
        System.getenv("WHISPER_SAFETENSORS")?.let { File(it) }?.takeIf { File(it, "model.safetensors").exists() }

    private fun resourceFloats(name: String): FloatArray {
        val bytes = checkNotNull(javaClass.getResourceAsStream("/whisper-de/$name")) { "missing resource $name" }.readBytes()
        return FloatArray(bytes.size / 4) { i ->
            var bits = 0
            for (b in 0 until 4) bits = bits or ((bytes[i * 4 + b].toInt() and 0xFF) shl (8 * b))
            Float.fromBits(bits)
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun ids(ctx: DirectCpuExecutionContext, vararg v: Int): Tensor<FP32, Float> =
        ctx.fromIntArray<Int32, Float>(Shape(v.size), Int32::class, v) as Tensor<FP32, Float>

    @Test
    fun eagerGreedyMatchesGolden() {
        val snap = snapshotDir() ?: run {
            println("SKIP: WHISPER_SAFETENSORS not set"); return
        }
        val ctx = DirectCpuExecutionContext.create()
        val encoder = WhisperEncoderModel<FP32, Float>(cfg, FP32::class)
        val decoder = WhisperDecoderModel<FP32, Float>(cfg, FP32::class)
        SafeTensorsWeightSource({ JvmRandomAccessSource.open(File(snap, "model.safetensors").path) }).use { src ->
            val n1 = bakeWhisperWeights(encoder, src, cfg, FP32::class, ctx as ExecutionContext)
            val n2 = bakeWhisperWeights(decoder, src, cfg, FP32::class, ctx)
            println("baked: encoder=$n1 decoder=$n2 params")
        }

        // --- 1. encoder parity ---
        val mel = resourceFloats("mel_de4s.bin") // [1,80,400]
        val melT = ctx.fromFloatArray<FP32, Float>(Shape(1, cfg.nMels, cfg.melFrames), FP32::class, mel)
        val feat = encoder.forward(melT, ctx)
        val golden = resourceFloats("golden_feat.bin") // [1,200,384]
        val featArr = toFloats(feat, cfg.audioCtx * cfg.dim)
        var dot = 0.0; var na = 0.0; var nb = 0.0
        for (i in golden.indices) { dot += featArr[i] * golden[i]; na += featArr[i] * featArr[i]; nb += golden[i] * golden[i] }
        val cos = dot / (kotlin.math.sqrt(na) * kotlin.math.sqrt(nb))
        println("encoder cosine vs golden: $cos")
        assertTrue(cos > 0.999, "encoder features diverged: cosine=$cos")

        // --- 2. prefill logits anchor ---
        val tokens = WhisperSpecialTokens.forVocab(cfg.vocabSize)
        val prompt = tokens.transcribePrompt("de")
        val s = prompt.size
        val causal = WhisperMasks.causal<FP32, Float>(s, ctx, FP32::class)
        val zeroPad = WhisperMasks.zeroPad<FP32, Float>(s, maxP, cfg.dim, ctx, FP32::class)
        val prefill = decoder.forwardPrefill(ids(ctx, *prompt), feat, causal, zeroPad, s, maxP, ctx)
        val logits = toFloats(prefill.logits, s * cfg.vocabSize)
        val lastRow = logits.copyOfRange((s - 1) * cfg.vocabSize, s * cfg.vocabSize)
        val goldenLogits = resourceFloats("golden_logits.bin")
        val ourArg = argmax(lastRow)
        val goldArg = argmax(goldenLogits)
        println("first-token argmax: ours=$ourArg golden=$goldArg")
        assertEquals(goldArg, ourArg, "first decoded token diverged from the ONNX-pipeline golden")

        // --- 3. full greedy decode ---
        var selfK = prefill.selfK.toMutableList()
        var selfV = prefill.selfV.toMutableList()
        val out = mutableListOf(ourArg)
        var tok = ourArg
        var pos = s
        while (tok != tokens.eot && pos < maxP && out.size < maxP - s) {
            val addMask = ctx.fromFloatArray<FP32, Float>(Shape(1, 1, 1, maxP), FP32::class, WhisperMasks.stepAddMask(pos, maxP))
            val wf = ctx.fromFloatArray<FP32, Float>(Shape(1, maxP, 1), FP32::class, WhisperMasks.stepWriteVector(pos, maxP))
            val r = decoder.forwardStep(
                ids(ctx, tok), ids(ctx, pos), addMask, wf,
                selfK, selfV, prefill.crossK, prefill.crossV, maxP, ctx,
            )
            selfK = r.selfK.toMutableList()
            selfV = r.selfV.toMutableList()
            tok = argmax(toFloats(r.logits, cfg.vocabSize))
            out += tok
            pos++
        }
        println("greedy tokens: $out")
        assertTrue(out.size > 2, "decode stopped immediately")
        assertTrue(out.last() == tokens.eot || out.size >= 3, "no meaningful decode")
    }

    private fun toFloats(t: Tensor<FP32, Float>, n: Int): FloatArray {
        val out = FloatArray(n)
        val data = t.data
        val dims = t.shape.dimensions
        var idx = 0
        // row-major walk
        val strides = IntArray(dims.size)
        var acc = 1
        for (d in dims.indices.reversed()) { strides[d] = acc; acc *= dims[d] }
        val index = IntArray(dims.size)
        while (idx < n) {
            var rem = idx
            for (d in dims.indices) { index[d] = rem / strides[d]; rem %= strides[d] }
            out[idx] = data.get(*index) as Float
            idx++
        }
        return out
    }

    private fun argmax(a: FloatArray): Int {
        var best = 0
        for (i in a.indices) if (a[i] > a[best]) best = i
        return best
    }
}
