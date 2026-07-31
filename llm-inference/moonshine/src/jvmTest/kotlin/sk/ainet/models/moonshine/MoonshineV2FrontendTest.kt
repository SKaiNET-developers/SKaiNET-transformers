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
 * Validate the DSL v2 audio frontend [moonshineV2Frontend] against the reference `frontend.onnx` (eager CPU,
 * real weights): raw audio `[1, samples]` → features `[1, frames, 320]`, cos-sim vs onnxruntime.
 *
 * Gated on `FE_DIR` — the dir produced by `bake_moonshine_v2_frontend.py` (fe_* .bin) plus `input.bin`
 * (`[samples]` f32, multiple of 80) and `ref.bin` (`[frames,320]` f32 = frontend.onnx on that input, zero state).
 */
class MoonshineV2FrontendTest {

    /** fe_* params are baked under their own names (weight-norm resolved), so the source map is identity. */
    private fun srcName(dsl: String): DecMap = DecMap(dsl, false)

    @Test
    fun v2FrontendMatchesOnnx() {
        val dir = System.getenv("FE_DIR") ?: run { println("SKIP v2FrontendMatchesOnnx: set FE_DIR"); return }

        val ctx = DirectCpuExecutionContext.create()
        val model = moonshineV2Frontend<FP32, Float>(FP32::class)
        val baked = bakeMoonshineWeights(model, DecDirBinWeightSource(dir), ::srcName, FP32::class, ctx as ExecutionContext)
        println("baked $baked v2 frontend params")

        val samples = readF32(File("$dir/input.bin"))
        val input = ctx.fromFloatArray<FP32, Float>(Shape(1, samples.size), FP32::class, samples)
        val out = model.forward(input, ctx)
        val ours = out.data.copyToFloatArray()
        val ref = readF32(File("$dir/ref.bin"))
        val cos = cosine(ours, ref)
        println("v2 frontend out shape=${out.shape.dimensions.toList()} (ref ${ref.size / 320}x320) cos=$cos")
        assertTrue(cos > 0.999, "DSL v2 frontend must match frontend.onnx (cos=$cos)")
    }

    private fun cosine(a: FloatArray, b: FloatArray): Double {
        val n = minOf(a.size, b.size)
        var d = 0.0; var na = 0.0; var nb = 0.0
        for (i in 0 until n) { d += a[i].toDouble() * b[i]; na += a[i].toDouble() * a[i]; nb += b[i].toDouble() * b[i] }
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
