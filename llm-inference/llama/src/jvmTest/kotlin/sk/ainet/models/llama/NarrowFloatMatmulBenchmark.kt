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
 * became engine issue #888, and this now doubles as the regression guard for that fix and for
 * #887, the FP16 kernel gap it exposed next.
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
 * ### Baseline, 2026-07-30 (engine develop with #888, #887 and the BF16 amortization all merged)
 *
 * Intel i7-9750H (AVX2, no AVX-512), 12 threads, OpenJDK 21.0.11. Median ms per call:
 *
 * ```
 * shape          batch      fp32       fp16      bf16   t()row-maj  t()in-maj  t()in-maj
 *                                                           (fp16)     (fp16)     (bf16)
 * q_proj  1B         1     5.352      3.129     1.513      212.110      3.051      1.529
 * q_proj  1B        16    16.556     10.716     9.261      218.329     10.498      9.187
 * q_proj  8B         1    40.002     26.276    20.951     1373.229     26.626     20.975
 * q_proj  8B        16    95.454     58.566    53.329     1417.367     59.053     53.533
 * ffn_up  8B         1   108.667     71.156    57.428     4389.079     72.150     57.153
 * ffn_up  8B        16   271.129    159.255   143.489     4506.450    155.102    139.751
 * ffn_down 8B        1   107.462     70.868    56.134     2216.040     70.862     55.819
 * ffn_down 8B       16   263.130    155.509   141.947     2383.415    155.820    143.709
 * ```
 *
 * `q_proj 1B` at batch 1 is the shortest measurement here and the noisiest: its FP32 sample came
 * out at 5.35 ms against a 3.09–3.22 ms cluster in repeat runs, so read that row's ratios with
 * suspicion. The other seven are stable across runs.
 *
 * Four conclusions:
 *
 *  1. **The relayout removes the transpose cost entirely.** `t()in-maj` matches the direct column
 *     to within noise at every size, so the zero-copy view holds. Against the old path that is
 *     4389 ms → 57 ms for `ffn_up` BF16 at batch 1, a 77x reduction, and 212 ms → 1.5 ms for
 *     `q_proj 1B`. Before this, KEEP_NATIVE was unusably slow at real model sizes.
 *  2. **Both narrow formats now beat FP32** — BF16 by 1.8–1.9x, FP16 by 1.5–1.7x — at every shape
 *     and both batch sizes. Halving the weight bytes is most of it.
 *  3. **FP16 trails BF16 by only 10–24%**, which is the cost of its dequant: BF16 is one shift,
 *     binary16 needs rebiasing and gradual underflow. It used to trail by 2–18x, and the cause was
 *     not the decode at all — `NativeKernelProvider` carried `matmulBf16` but no `matmulFp16`, so
 *     BF16 ran the native FFM kernel at priority 100 while FP16 silently cascaded to the JVM
 *     Panama kernel at 50. Head to head the two Panama kernels are within ~15% of each other.
 *     Fixed by #887; the lesson is that a dtype benchmarking far off its siblings is more likely
 *     served by a different provider than by a worse kernel.
 *  4. **These kernels are compute-bound at batch 16, not bandwidth-bound.** Both native kernels
 *     now tile `j` and read B once per matmul instead of once per row of A, cutting B traffic 16x
 *     at m=16 — and that bought only 9–19%. Whatever is left is the FMA chain, so the next real
 *     win is a blocked microkernel or `bfdot`/`bfmmla` on ARMv8.6-A+, not more layout work.
 *     At m=1 both kernels deliberately keep the straight i-p-j pass: there is nothing to amortize,
 *     and tiling cost 15% there.
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
