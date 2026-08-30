package sk.ainet.lang.nn.transformer

import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

/**
 * Pins [BitNetFFN]'s math (transformers#336) against a hand-written reference of the BitNet
 * b1.58 FFN — `down(subNorm(relu(gate(x))² * up(x)))` — the structure verified against NeoGPU's
 * reference driver, and pins the [MultiHeadAttention] `attnSubNorm` placement (RMSNorm on the
 * merged attention output BEFORE o_proj).
 */
class BitNetFFNTest {

    private val ctx = DirectCpuExecutionContext()
    private val eps = 1e-5f

    private fun tensor(rows: Int, cols: Int, seed: Int): Tensor<FP32, Float> =
        ctx.fromFloatArray(
            Shape(rows, cols), FP32::class,
            FloatArray(rows * cols) { (kotlin.math.sin((seed * 1000 + it).toFloat()) * 0.5f) },
        )

    private fun vector(n: Int, seed: Int): FloatArray =
        FloatArray(n) { (kotlin.math.cos((seed * 500 + it).toFloat()) * 0.5f + 0.75f) }

    private fun matVec(w: FloatArray, rows: Int, cols: Int, x: FloatArray): FloatArray =
        FloatArray(rows) { r ->
            var acc = 0f
            for (c in 0 until cols) acc += w[r * cols + c] * x[c]
            acc
        }

    private fun rmsNorm(x: FloatArray, w: FloatArray, eps: Float): FloatArray {
        var meanSq = 0f
        for (v in x) meanSq += v * v
        val rms = sqrt(meanSq / x.size + eps)
        return FloatArray(x.size) { x[it] / rms * w[it] }
    }

    @Test
    fun forwardMatchesTheHandComputedBitNetFfn() {
        val dim = 4; val hidden = 6
        val ffn = BitNetFFN<FP32, Float>(dim, hidden, subNormEps = eps.toDouble(), name = "ffn", dtype = FP32::class)

        val gateW = FloatArray(hidden * dim) { (kotlin.math.sin((1000 + it).toFloat()) * 0.5f) }
        val upW = FloatArray(hidden * dim) { (kotlin.math.sin((2000 + it).toFloat()) * 0.5f) }
        val downW = FloatArray(dim * hidden) { (kotlin.math.sin((3000 + it).toFloat()) * 0.5f) }
        val subW = vector(hidden, seed = 4)
        ffn.params[0].value = ctx.fromFloatArray(Shape(hidden, dim), FP32::class, gateW)
        ffn.params[1].value = ctx.fromFloatArray(Shape(hidden, dim), FP32::class, upW)
        ffn.params[2].value = ctx.fromFloatArray(Shape(dim, hidden), FP32::class, downW)
        ffn.subNorm.params[0].value = ctx.fromFloatArray(Shape(hidden), FP32::class, subW)

        val x = floatArrayOf(0.5f, -1.0f, 0.25f, 0.75f)
        val out = ffn.forward(ctx.fromFloatArray(Shape(1, dim), FP32::class, x), ctx)

        // reference: down(subNorm(relu(gate x)² * (up x)))
        val gate = matVec(gateW, hidden, dim, x)
        val up = matVec(upW, hidden, dim, x)
        val gated = FloatArray(hidden) {
            val r = if (gate[it] > 0f) gate[it] else 0f
            r * r * up[it]
        }
        val normed = rmsNorm(gated, subW, eps)
        val expected = matVec(downW, dim, hidden, normed)

        for (i in 0 until dim) {
            val got = out.data.get(0, i)
            assertTrue(
                abs(got - expected[i]) <= 1e-4f * maxOf(1f, abs(expected[i])),
                "[$i]: ffn=$got reference=${expected[i]}",
            )
        }
    }

    @Test
    fun attnSubNormIsAppliedToTheMergedOutputBeforeOProj() {
        // seqLen 1, causal self-attention: softmax over the single key is 1, so the merged
        // attention output IS v = W_v·x — making the whole path hand-computable:
        //   expected = W_o · rmsNorm(W_v · x, subW)
        val dim = 4
        val mha = MultiHeadAttention<FP32, Float>(
            dim = dim, nHeads = 2, nKVHeads = 2, causal = true,
            bias = false, qkNorm = false,
            attnSubNorm = true, attnSubNormEps = eps.toDouble(),
            name = "attn",
        )
        val qW = FloatArray(dim * dim) { (kotlin.math.sin((100 + it).toFloat()) * 0.5f) }
        val kW = FloatArray(dim * dim) { (kotlin.math.sin((200 + it).toFloat()) * 0.5f) }
        val vW = FloatArray(dim * dim) { (kotlin.math.sin((300 + it).toFloat()) * 0.5f) }
        val oW = FloatArray(dim * dim) { (kotlin.math.sin((400 + it).toFloat()) * 0.5f) }
        val subW = vector(dim, seed = 7)
        mha.params[0].value = ctx.fromFloatArray(Shape(dim, dim), FP32::class, qW)
        mha.params[1].value = ctx.fromFloatArray(Shape(dim, dim), FP32::class, kW)
        mha.params[2].value = ctx.fromFloatArray(Shape(dim, dim), FP32::class, vW)
        mha.params[3].value = ctx.fromFloatArray(Shape(dim, dim), FP32::class, oW)
        mha.subNorm!!.params[0].value = ctx.fromFloatArray(Shape(dim), FP32::class, subW)

        val x = floatArrayOf(0.5f, -0.25f, 1.0f, -0.5f)
        val out = mha.forward(ctx.fromFloatArray(Shape(1, dim), FP32::class, x), ctx)

        val v = matVec(vW, dim, dim, x)
        val expected = matVec(oW, dim, dim, rmsNorm(v, subW, eps))
        for (i in 0 until dim) {
            val got = out.data.get(0, i)
            assertTrue(
                abs(got - expected[i]) <= 1e-4f * maxOf(1f, abs(expected[i])),
                "[$i]: mha=$got reference=${expected[i]}",
            )
        }
    }
}
