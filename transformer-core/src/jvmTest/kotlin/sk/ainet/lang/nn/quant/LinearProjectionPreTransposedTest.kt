package sk.ainet.lang.nn.quant

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.nn.transformer.linearProject
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32

/**
 * Proves the #184-hoist-3 pre-transpose marker end to end on a real CPU
 * backend: a Q5_1 weight packed with [BlockQuantPacking.packPreTransposed]
 * (logical `[in, out]`, [PreTransposedWeight]-marked) must produce, through
 * [linearProject]'s transpose-skipping branch, bit-identical output to the
 * same bytes packed with [BlockQuantPacking.pack] and run through the classic
 * `ops.matmul(x, ops.transpose(W))` path — and both must match an FP32
 * reference built from the analytic Q5_1 dequant formula.
 *
 * jvmTest because it needs actual packed-matmul kernel dispatch
 * (ServiceLoader-discovered scalar/Panama providers); the layout/marker
 * mechanics are covered platform-neutrally in [BlockQuantPackingTest].
 */
class LinearProjectionPreTransposedTest {

    /** IEEE-754 float32 → float16 bits (round-to-nearest-even; small exact values only in tests). */
    private fun halfBits(f: Float): Int {
        val bits = f.toRawBits()
        val sign = (bits ushr 16) and 0x8000
        var exp = ((bits ushr 23) and 0xFF) - 127 + 15
        var mant = bits and 0x7FFFFF
        if (exp <= 0) return sign // flush tiny values; tests only use exact normal halves
        if (exp >= 31) return sign or 0x7C00
        // round mantissa 23 -> 10 bits
        mant += 0x1000
        if (mant and 0x800000 != 0) {
            mant = 0
            exp += 1
            if (exp >= 31) return sign or 0x7C00
        }
        return sign or (exp shl 10) or (mant ushr 13)
    }

    /**
     * Build one 24-byte GGUF Q5_1 block from [d], [m] and 32 5-bit [codes],
     * matching the engine layout: `d` f16 LE, `m` f16 LE, `qh` 4 bytes (bit j
     * = high bit of code j, LSB-first), `qs` 16 bytes (low nibbles of codes
     * 0..15 | low nibbles of codes 16..31 shifted). Dequant: `d * code + m`.
     */
    private fun q5_1Block(d: Float, m: Float, codes: IntArray): ByteArray {
        require(codes.size == 32)
        val out = ByteArray(24)
        val db = halfBits(d)
        val mb = halfBits(m)
        out[0] = (db and 0xFF).toByte(); out[1] = ((db ushr 8) and 0xFF).toByte()
        out[2] = (mb and 0xFF).toByte(); out[3] = ((mb ushr 8) and 0xFF).toByte()
        var qh = 0
        for (j in 0 until 32) if ((codes[j] ushr 4) and 1 == 1) qh = qh or (1 shl j)
        for (b in 0 until 4) out[4 + b] = ((qh ushr (8 * b)) and 0xFF).toByte()
        for (j in 0 until 16) out[8 + j] = ((codes[j] and 0xF) or ((codes[j + 16] and 0xF) shl 4)).toByte()
        return out
    }

    @Test
    fun preTransposed_q5_1_matches_classic_path_and_fp32_reference() {
        val outDim = 4
        val inDim = 64 // 2 blocks per row -> multi-block both ways, catches relayout mistakes
        val blocksPerRow = inDim / 32
        val shape = Shape(outDim, inDim)

        // Deterministic synthetic weight: per-block (d, m) + pseudo-random 5-bit codes.
        val expected = FloatArray(outDim * inDim)
        val ggufBytes = ByteArray(outDim * blocksPerRow * 24)
        for (r in 0 until outDim) {
            for (b in 0 until blocksPerRow) {
                val blockIdx = r * blocksPerRow + b
                val d = 0.25f * ((blockIdx % 4) + 1) // 0.25 / 0.5 / 0.75 / 1.0 — f16-exact
                val m = -2.0f + 0.5f * (blockIdx % 3)
                val codes = IntArray(32) { j -> (j * 7 + r * 13 + b * 5) % 32 }
                q5_1Block(d, m, codes).copyInto(ggufBytes, blockIdx * 24)
                for (j in 0 until 32) expected[r * inDim + b * 32 + j] = d * codes[j] + m
            }
        }

        val ctx = DirectCpuExecutionContext.create()
        val x = ctx.fromFloatArray<FP32, Float>(
            Shape(2, inDim), FP32::class,
            FloatArray(2 * inDim) { i -> ((i * 31 + 7) % 17 - 8) / 8.0f },
        )

        // FP32 reference from the analytic dequant.
        val wFp32 = ctx.fromFloatArray<FP32, Float>(shape, FP32::class, expected)
        val ref = linearProject(ctx.ops, x, wFp32).data.copyToFloatArray()

        // Classic path: [out, in] packed + lazy transpose inside linearProject.
        @Suppress("UNCHECKED_CAST")
        val packed = BlockQuantPacking.pack<FP32>(ggufBytes, TensorEncoding.Q5_1, shape)
            as TensorData<FP32, Float>
        val yClassic = linearProject(ctx.ops, x, ctx.fromData(packed, FP32::class)).data.copyToFloatArray()

        // Marked path: [in, out] pre-transposed, linearProject skips ops.transpose.
        val pre = BlockQuantPacking.packPreTransposed<FP32>(ggufBytes, TensorEncoding.Q5_1, shape)
            ?: error("packPreTransposed returned null for Q5_1")
        assertTrue(pre is PreTransposedWeight)
        assertEquals(Shape(inDim, outDim), pre.shape)
        @Suppress("UNCHECKED_CAST")
        val yMarked = linearProject(ctx.ops, x, ctx.fromData(pre as TensorData<FP32, Float>, FP32::class))
            .data.copyToFloatArray()

        // Same kernel, same bytes, same dims -> bit-identical between the two packed paths.
        assertTrue(yClassic.contentEquals(yMarked), "pre-transposed path diverged from classic packed path")
        // And both agree with the analytic FP32 reference (fp accumulation-order tolerance).
        for (i in ref.indices) {
            assertTrue(
                abs(ref[i] - yClassic[i]) <= 1e-3f * maxOf(1.0f, abs(ref[i])),
                "packed Q5_1 [$i]: ${yClassic[i]} vs FP32 ref ${ref[i]}",
            )
        }
    }

    @Test
    fun marker_skips_transpose_for_plain_float_data_too() {
        val ctx = DirectCpuExecutionContext.create()
        val outDim = 3
        val inDim = 5
        val w = FloatArray(outDim * inDim) { (it % 7 - 3).toFloat() }
        val x = ctx.fromFloatArray<FP32, Float>(Shape(2, inDim), FP32::class, FloatArray(2 * inDim) { (it % 5 - 2).toFloat() })

        val ref = linearProject(
            ctx.ops, x,
            ctx.fromFloatArray<FP32, Float>(Shape(outDim, inDim), FP32::class, w),
        ).data.copyToFloatArray()

        // Manually transposed [in, out] float weight + marker: linearProject
        // must consume it as-is (no second transpose).
        val wt = FloatArray(inDim * outDim)
        for (r in 0 until outDim) for (c in 0 until inDim) wt[c * outDim + r] = w[r * inDim + c]
        val wtTensor = ctx.fromFloatArray<FP32, Float>(Shape(inDim, outDim), FP32::class, wt)

        class Marked(private val d: TensorData<FP32, Float>) :
            TensorData<FP32, Float> by d, PreTransposedWeight

        val marked = ctx.fromData(Marked(wtTensor.data), FP32::class)
        val y = linearProject(ctx.ops, x, marked).data.copyToFloatArray()
        // Tolerance, not bit-equality: the marked weight is not FloatArray-backed,
        // so the backend may take a different (elementwise) matmul path than the
        // FloatArray fast path of the reference.
        assertEquals(ref.size, y.size)
        for (i in ref.indices) {
            assertTrue(
                abs(ref[i] - y[i]) <= 1e-4f * maxOf(1.0f, abs(ref[i])),
                "marked FP32 pre-transposed weight [$i]: ${y[i]} vs ${ref[i]}",
            )
        }
    }
}
