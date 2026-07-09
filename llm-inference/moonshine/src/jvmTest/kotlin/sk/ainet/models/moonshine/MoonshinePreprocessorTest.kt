package sk.ainet.models.moonshine

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import java.io.File
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Validate the DSL audio frontend [moonshinePreprocessor] against the reference `enc_frontend.onnx`:
 * raw audio `[1, 64000]` → features `[1, 165, 288]`, eager on CPU with real weights.
 *
 * Gated on env: PP_CHECKPOINT (HF `.bin` weights dir), PP_INPUT (`[1,64000]` f32), PP_REF (`[1,165,288]` f32).
 */
class MoonshinePreprocessorTest {
    @Test
    fun preprocessorMatchesOnnxFrontend() {
        val ckpt = System.getenv("PP_CHECKPOINT") ?: return skip("PP_CHECKPOINT")
        val inPath = System.getenv("PP_INPUT") ?: return skip("PP_INPUT")
        val refPath = System.getenv("PP_REF") ?: return skip("PP_REF")

        val cfg = MoonshineConfig()
        val ctx = DirectCpuExecutionContext.create()
        val model = moonshinePreprocessor<FP32, Float>(cfg, FP32::class)
        val baked = bakeMoonshineWeights(model, DecDirBinWeightSource(ckpt), ::preprocessorHfNameFor, FP32::class, ctx as ExecutionContext)
        println("baked $baked preprocessor params")

        val samples = readF32(File(inPath))
        val input = ctx.fromFloatArray<FP32, Float>(Shape(1, samples.size), FP32::class, samples)
        val out = model.forward(input, ctx)
        val ours = out.data.copyToFloatArray()
        val ref = readF32(File(refPath))
        val cos = cosine(ours, ref)
        println("preprocessor output shape=${out.shape.dimensions.toList()} cos vs onnx frontend = $cos")
        assertTrue(cos > 0.99, "DSL frontend must match the ONNX frontend (cos=$cos)")
    }

    private fun skip(m: String) = println("SKIP preprocessorMatchesOnnxFrontend: set $m")

    private fun cosine(a: FloatArray, b: FloatArray): Double {
        val n = minOf(a.size, b.size)
        var d = 0.0; var na = 0.0; var nb = 0.0
        for (i in 0 until n) { d += a[i] * b[i]; na += a[i].toDouble() * a[i]; nb += b[i].toDouble() * b[i] }
        return d / (sqrt(na) * sqrt(nb))
    }

    private fun readF32(f: File): FloatArray {
        val b = f.readBytes()
        return FloatArray(b.size / 4) { i ->
            var bits = 0
            for (k in 0 until 4) bits = bits or ((b[i * 4 + k].toInt() and 0xFF) shl (8 * k))
            Float.fromBits(bits)
        }
    }
}
