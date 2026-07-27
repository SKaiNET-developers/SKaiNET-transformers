package sk.ainet.models.llama

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Bf16DenseTensorData
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.Fp16DenseTensorData
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
 * `DecoderNarrowFloatForwardParityTest` established that KEEP_NATIVE is numerically correct but
 * that the FP16/BF16 SGEMM kernels are never reached: both `Linear.onForward` and
 * `LlamaRuntime.linearProject` call `w.t()`, transpose has no narrow-float arm and widens to a
 * dense FP32 buffer, and `DefaultCpuOpsJvm.chooseQuantizedMatmul` only engages for `[in, out]`
 * weights. Wiring the kernel up means a byte relayout at load plus a lazy-transpose arm in the
 * engine — the pattern the K-quants already use. That is only worth doing if the kernel actually
 * wins, so this measures three things at realistic projection sizes:
 *
 *  - **fp32**       — dense FP32 SGEMM. The baseline, and what runs today after the widening.
 *  - **fp16/bf16**  — weight handed over already `[in, out]`, so `chooseQuantizedMatmul` dispatches
 *                     to the narrow kernel. This is the best case the layout work could unlock.
 *  - **transpose**  — narrow weight in the real `[out, in]` orientation, `.t()` then matmul. This
 *                     is what production does per token today, and shows what the widening costs.
 *
 * Dispatch is guaranteed by construction rather than observed: `chooseQuantizedMatmul` requires an
 * FP32 rank-2 input, a rank-2 weight, and `weight.shape[0] == input.shape[1]`. [checkDispatchable]
 * asserts exactly that before timing, so a silent fallback to the generic path cannot be mistaken
 * for a fast kernel.
 *
 * ### Baseline, 2026-07-27
 *
 * Intel i7-9750H (AVX2, no AVX-512), 12 threads, OpenJDK 21.0.11, engine 0.38.0-SNAPSHOT.
 * Median ms per call:
 *
 * ```
 * shape          batch      fp32      fp16      bf16   transpose
 * q_proj  1B         1     3.210    18.601     1.591     206.458
 * q_proj  1B        16    16.449   297.873    11.125     215.698
 * q_proj  8B         1    40.335    73.793    21.472    1318.772
 * q_proj  8B        16    95.954  1166.224    66.227    1358.200
 * ffn_up  8B         1   111.922   197.337    60.372    4367.357
 * ffn_up  8B        16   272.889  3146.712   186.360    4511.455
 * ffn_down 8B        1   109.242   198.977    57.749    2195.538
 * ffn_down 8B       16   274.547  3172.639   177.142    2363.605
 * ```
 *
 * Three conclusions:
 *
 *  1. **BF16 beats FP32 by 1.5–2.1x everywhere.** At batch 1 the matmul is memory-bandwidth bound,
 *     so halving the weight bytes roughly halves the time. This is the case for doing the layout
 *     work.
 *  2. **FP16 is 2–18x slower, pinned at ~0.5 GFLOP/s regardless of shape or batch** — the signature
 *     of being compute-bound on the decode. Both Panama kernels fill a scratch lane array scalar-wise
 *     before the vector FMA, but BF16's decode is three integer ops while `Fp16Codec.decode` is a
 *     branchy `when` with a subnormal renormalization loop. The fix is engine-side and independent
 *     of layout: use `Float.float16ToFloat` (a JDK 20+ intrinsic) or a branch-free decode.
 *  3. **The `transpose` column is the alarming one.** 0.2–4.5 *seconds* for one projection, because
 *     the generic transpose walks a narrow tensor element by element through `get()`. That is the
 *     path production takes today, per weight, per token. KEEP_NATIVE is not merely un-accelerated
 *     right now — at real model sizes it is unusably slow.
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
        println("weight layout [in, out]; 'transpose' is the [out, in] + .t() path production uses today")
        println()
        println(
            "%-13s %6s %11s %11s %11s %11s   %s".format(
                "shape", "batch", "fp32", "fp16", "bf16", "transpose", "verdict",
            ),
        )
        println("-".repeat(96))

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
            // The production orientation: [out, in], transposed on every call.
            @Suppress("UNCHECKED_CAST")
            val wFp16Transposed = ctx.fromData(
                Fp16DenseTensorData(Shape(outF, inF), fp16Bytes) as TensorData<FP32, Float>,
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
                    "%-13s %6d %9.3fms %9.3fms %9.3fms %9.3fms   %s".format(
                        label, batch,
                        fp32Ns / 1e6, fp16Ns / 1e6, bf16Ns / 1e6, transposeNs / 1e6,
                        verdict,
                    ),
                )
                println(
                    "%-13s %6s %9.1fGF %9.1fGF %9.1fGF %9s".format(
                        "", "",
                        gflops(batch, inF, outF, fp32Ns),
                        gflops(batch, inF, outF, fp16Ns),
                        gflops(batch, inF, outF, bf16Ns),
                        "",
                    ),
                )
            }
        }

        println()
        println("weight bytes at rest: fp32 = 2x narrow. The 'transpose' column is the per-call cost")
        println("of the widening that happens today, and is what the layout work would remove.")
    }
}
