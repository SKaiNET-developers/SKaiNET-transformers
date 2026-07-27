package sk.ainet.models.llama

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Bf16DenseTensorData
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.Fp16DenseTensorData
import sk.ainet.lang.tensor.data.NarrowFloatInputMajorTensorData
import sk.ainet.lang.tensor.data.NarrowFloatTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.matmul
import sk.ainet.lang.tensor.t
import sk.ainet.lang.types.Bf16Codec
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Fp16Codec
import sk.ainet.lang.types.NarrowFloatCodec
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Decides whether reaching the narrow-float matmul kernel is worth the layout work.
 *
 * Skipped unless `-Dskainet.bench.narrow=true`. This is a measurement, not a test — it asserts
 * only the things that would invalidate its own numbers.
 *
 * ### What it answers
 *
 * Originally: is reaching the narrow-float kernel worth the layout work? It was — the answer
 * became engine issue #888, and this now doubles as the regression guard for that fix.
 *
 * Columns:
 *
 *  - **fp32**        — dense FP32 SGEMM. The baseline.
 *  - **fp16/bf16**   — weight handed over already `[in, out]`, so `chooseQuantizedMatmul`
 *                      dispatches to the narrow kernel. The ceiling.
 *  - **t()row-maj**  — row-major narrow weight in the real `[out, in]` orientation, `.t()` then
 *                      matmul. The pre-#888 path: transpose widens it elementwise through boxed
 *                      `get()`. Still measured, because row-major narrow tensors deliberately
 *                      keep this behaviour — only the input-major type may be reinterpreted.
 *  - **t()in-maj**   — same weight relaid input-major at load, so `.t()` is a zero-copy view.
 *                      What the loader produces for matmul weights now. Should match the direct
 *                      `fp16`/`bf16` columns; any gap means the transpose is copying again.
 *
 * Dispatch is guaranteed by construction rather than observed: `chooseQuantizedMatmul` requires an
 * FP32 rank-2 input, a rank-2 weight, and `weight.shape[0] == input.shape[1]`. [checkDispatchable]
 * asserts exactly that before timing, and the input-major weight is asserted to still be narrow
 * after `.t()`, so a silent fallback to the generic path cannot be mistaken for a fast kernel.
 *
 * ### Baseline, 2026-07-27 (engine 0.38.0-SNAPSHOT with the #888 fix)
 *
 * Intel i7-9750H (AVX2, no AVX-512), 12 threads, OpenJDK 21.0.11. Median ms per call:
 *
 * ```
 * shape          batch      fp32       fp16      bf16   t()row-maj  t()in-maj  t()in-maj
 *                                                           (fp16)     (fp16)     (bf16)
 * q_proj  1B         1     3.219     18.421     1.521      209.514     18.530      1.499
 * q_proj  1B        16    16.402    298.912    10.720      224.442    300.323     11.033
 * q_proj  8B         1    41.016     73.822    21.423     1368.664     74.180     21.254
 * q_proj  8B        16    98.640   1182.081    67.116     1394.869   1182.299     67.735
 * ffn_up  8B         1   110.393    199.491    58.476     4465.222    199.678     58.007
 * ffn_up  8B        16   273.756   3200.778   183.782     4591.961   3201.357    184.874
 * ffn_down 8B        1   111.234    199.131    58.412     2295.821    201.194     58.759
 * ffn_down 8B       16   263.774   3191.621   181.348     2397.454   3201.594    180.597
 * ```
 *
 * Three conclusions:
 *
 *  1. **The relayout removes the transpose cost entirely.** `t()in-maj` matches the direct column
 *     to within noise at every size, so the zero-copy view holds. Against the old path that is
 *     4465 ms → 58 ms for `ffn_up` BF16 at batch 1, a 77x reduction, and 209 ms → 1.5 ms for
 *     `q_proj 1B`. Before this, KEEP_NATIVE was unusably slow at real model sizes.
 *  2. **BF16 beats FP32 by 1.5–2.1x.** At batch 1 the matmul is memory-bandwidth bound, so halving
 *     the weight bytes roughly halves the time. This is the win the feature exists for.
 *  3. **FP16 is still 2–18x slower than FP32**, pinned near 0.5 GFLOP/s regardless of shape or
 *     batch — compute-bound on the decode, not on layout. Both Panama kernels fill a scratch lane
 *     array scalar-wise before the vector FMA, but BF16's decode is three integer ops while
 *     `Fp16Codec.decode` is a branchy `when` with a subnormal renormalization loop. Tracked
 *     separately as engine issue #887; until it lands, prefer BF16 for speed.
 */
class NarrowFloatMatmulBenchmark {

    private val enabled = System.getProperty("skainet.bench.narrow") == "true"

    /** `[inFeatures, outFeatures]` taken from real LLaMA projections, plus a small control. */
    private val shapes = listOf(
        Triple("q_proj  1B", 2048, 2048),
        Triple("q_proj  8B", 4096, 4096),
        Triple("ffn_up  8B", 4096, 11008),
        Triple("ffn_down 8B", 11008, 4096),
    )

    /** Decode is batch 1; the larger batch stands in for prefill, where SGEMM has more to work with. */
    private val batches = listOf(1, 16)

    private val minSamples = 5
    private val timeBudgetNanos = 1_500_000_000L
    private val warmupNanos = 500_000_000L

    private fun encode(values: FloatArray, codec: NarrowFloatCodec): ByteArray {
        val out = ByteArray(values.size * 2)
        for (i in values.indices) {
            val bits = codec.encode(values[i])
            out[i * 2] = (bits and 0xFF).toByte()
            out[i * 2 + 1] = ((bits ushr 8) and 0xFF).toByte()
        }
        return out
    }

    private fun weights(n: Int, seed: Int): FloatArray {
        val rng = kotlin.random.Random(seed)
        return FloatArray(n) { (rng.nextFloat() - 0.5f) * 0.1f }
    }

    /**
     * Fails rather than silently reporting generic-path timings as kernel timings.
     * Mirrors the preconditions in `DefaultCpuOpsJvm.chooseQuantizedMatmul`.
     */
    private fun checkDispatchable(x: Tensor<FP32, Float>, w: Tensor<FP32, Float>, label: String) {
        assertTrue(w.data is NarrowFloatTensorData, "$label: weight is not narrow — nothing to dispatch")
        assertTrue(x.shape.rank == 2 && w.shape.rank == 2, "$label: both operands must be rank 2")
        assertTrue(
            w.shape[0] == x.shape[1],
            "$label: weight is [${w.shape[0]}, ${w.shape[1]}] but input has ${x.shape[1]} columns — " +
                "chooseQuantizedMatmul would return null and the generic path would be timed instead",
        )
    }

    /** Median nanoseconds per call, after a fixed warmup window. */
    private fun measure(body: () -> Tensor<FP32, Float>): Long {
        var sink = 0.0f
        val warmupEnd = System.nanoTime() + warmupNanos
        while (System.nanoTime() < warmupEnd) {
            sink += body().data.copyToFloatArray()[0]
        }

        val samples = mutableListOf<Long>()
        val deadline = System.nanoTime() + timeBudgetNanos
        while (samples.size < minSamples || System.nanoTime() < deadline) {
            val t0 = System.nanoTime()
            val r = body()
            val elapsed = System.nanoTime() - t0
            sink += r.data.copyToFloatArray()[0]
            samples.add(elapsed)
            if (samples.size >= 2000) break
        }
        check(!sink.isNaN()) { "sink went NaN — results were not consumed" }
        samples.sort()
        return samples[samples.size / 2]
    }

    private fun gflops(batch: Int, inF: Int, outF: Int, nanos: Long): Double =
        (2.0 * batch * inF * outF) / nanos

    @Test
    fun `narrow float matmul throughput versus fp32`() {
        if (!enabled) {
            println("NarrowFloatMatmulBenchmark skipped — rerun with -Dskainet.bench.narrow=true")
            return
        }

        val ctx = DirectCpuExecutionContext()
        println()
        println("narrow-float matmul vs fp32 SGEMM   (median of timed samples)")
        println("direct columns take the weight already [in, out]; t() columns transpose [out, in] first")
        println()
        println(
            "%-13s %6s %10s %10s %10s %12s %11s %11s   %s".format(
                "shape", "batch", "fp32", "fp16", "bf16",
                "t()row-maj", "t()in-maj", "t()in-maj", "verdict",
            ),
        )
        println(
            "%-13s %6s %10s %10s %10s %12s %11s %11s".format(
                "", "", "", "", "", "(fp16)", "(fp16)", "(bf16)",
            ),
        )
        println("-".repeat(118))

        for ((label, inF, outF) in shapes) {
            val raw = weights(inF * outF, seed = inF + outF)
            val fp16Bytes = encode(raw, Fp16Codec)
            val bf16Bytes = encode(raw, Bf16Codec)
            // Decode back so the FP32 baseline holds the same values the narrow paths do —
            // otherwise the comparison is between different matrices.
            val fp32Values = FloatArray(raw.size) {
                Fp16Codec.decode(Fp16Codec.encode(raw[it]))
            }

            @Suppress("UNCHECKED_CAST")
            val wFp32 = ctx.fromData(
                DenseFloatArrayTensorData<FP32>(Shape(inF, outF), fp32Values) as TensorData<FP32, Float>,
                FP32::class,
            )
            @Suppress("UNCHECKED_CAST")
            val wFp16 = ctx.fromData(
                Fp16DenseTensorData(Shape(inF, outF), fp16Bytes) as TensorData<FP32, Float>,
                FP32::class,
            )
            @Suppress("UNCHECKED_CAST")
            val wBf16 = ctx.fromData(
                Bf16DenseTensorData(Shape(inF, outF), bf16Bytes) as TensorData<FP32, Float>,
                FP32::class,
            )
            // The production orientation: [out, in], transposed on every call. Row-major, so the
            // transpose falls to the generic elementwise path — what production did before #888.
            @Suppress("UNCHECKED_CAST")
            val wFp16Transposed = ctx.fromData(
                Fp16DenseTensorData(Shape(outF, inF), fp16Bytes) as TensorData<FP32, Float>,
                FP32::class,
            )
            // Same [out, in] orientation, but relaid input-major at load, so `.t()` is a
            // zero-copy view and the weight reaches the kernel still packed. What the loader
            // produces for matmul weights now.
            @Suppress("UNCHECKED_CAST")
            val wFp16InputMajor = ctx.fromData(
                NarrowFloatInputMajorTensorData.fromRowMajor(
                    Shape(outF, inF), fp16Bytes, Fp16Codec,
                ) as TensorData<FP32, Float>,
                FP32::class,
            )
            @Suppress("UNCHECKED_CAST")
            val wBf16InputMajor = ctx.fromData(
                NarrowFloatInputMajorTensorData.fromRowMajor(
                    Shape(outF, inF), bf16Bytes, Bf16Codec,
                ) as TensorData<FP32, Float>,
                FP32::class,
            )

            for (batch in batches) {
                val x = ctx.fromFloatArray<FP32, Float>(
                    Shape(batch, inF), FP32::class, weights(batch * inF, seed = 99),
                )

                checkDispatchable(x, wFp16, "$label fp16")
                checkDispatchable(x, wBf16, "$label bf16")

                val fp32Ns = measure { x.matmul(wFp32) }
                val fp16Ns = measure { x.matmul(wFp16) }
                val bf16Ns = measure { x.matmul(wBf16) }
                val transposeNs = measure { x.matmul(wFp16Transposed.t()) }
                val inMajFp16Ns = measure { x.matmul(wFp16InputMajor.t()) }
                val inMajBf16Ns = measure { x.matmul(wBf16InputMajor.t()) }

                // The relaid weight must still be packed after `.t()`, or the two columns below
                // are just re-measuring the generic path under a different name.
                assertTrue(
                    wBf16InputMajor.t().data is NarrowFloatTensorData,
                    "$label: input-major weight widened on transpose — engine #888 arm missing?",
                )

                // Sanity: the narrow kernel must agree with the FP32 baseline, or the timing is
                // measuring something that isn't a correct matmul.
                val ref = x.matmul(wFp32).data.copyToFloatArray()
                val got = x.matmul(wFp16).data.copyToFloatArray()
                val tol = 1e-2f * (1 + inF / 1024)
                val maxDelta = ref.indices.maxOf { abs(ref[it] - got[it]) }
                assertTrue(
                    maxDelta < tol,
                    "$label batch=$batch: fp16 kernel disagrees with fp32 by $maxDelta (tol $tol)",
                )

                val speedup = fp32Ns.toDouble() / fp16Ns.toDouble()
                val verdict = when {
                    speedup >= 1.15 -> "fp16 %.2fx faster".format(speedup)
                    speedup <= 0.87 -> "fp16 %.2fx SLOWER".format(1 / speedup)
                    else -> "no real difference"
                }

                println(
                    "%-13s %6d %8.3fms %8.3fms %8.3fms %10.3fms %9.3fms %9.3fms   %s".format(
                        label, batch,
                        fp32Ns / 1e6, fp16Ns / 1e6, bf16Ns / 1e6,
                        transposeNs / 1e6, inMajFp16Ns / 1e6, inMajBf16Ns / 1e6,
                        verdict,
                    ),
                )
            }
        }

        println()
        println("weight bytes at rest: fp32 = 2x narrow.")
        println("t()row-maj is the pre-#888 path: a row-major narrow weight widened elementwise on")
        println("every transpose. t()in-maj is the same weight relaid input-major at load, so the")
        println("transpose is a zero-copy view and the packed weight reaches the kernel.")
    }
}
