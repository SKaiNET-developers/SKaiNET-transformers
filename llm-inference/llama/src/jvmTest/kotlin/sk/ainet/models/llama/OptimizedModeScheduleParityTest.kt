package sk.ainet.models.llama

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertTrue
import kotlinx.coroutines.runBlocking
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.schedule.Schedule
import sk.ainet.exec.schedule.CoroutineSchedule
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.nn.dsl.decoder.DECODER_DEQUANTIZE_ALL
import sk.ainet.lang.types.FP32

/**
 * SKEEP-005 phase 2: the compiled JVM leg (OPTIMIZED mode, `ComputeGraphExecutor` over
 * `ctx.ops`) runs under the context's schedule. Two compiled runtimes, one sequential and one on
 * the hardware schedule, must produce bit-identical logits on a real grouped-query model
 * (SmolLM2-135M: 9 heads over 3 KV heads — the GQA-native SDPA node in the executor), and the
 * compiled leg must agree with the eager DIRECT leg within rounding at position 0. Only there:
 * the compiled graph is a shape-[1] snapshot of one forward pass, so the KV cache and position
 * counter it replays are frozen (the documented OPTIMIZED-mode limitation) and later steps
 * legitimately diverge from the stateful eager loop. Schedule independence holds at every step.
 *
 * Same loading path as kllama's `RuntimeEquivalenceTest` (dequantized weights,
 * `compileUnoptimized`): OPTIMIZED mode with packed weights and the optimisation pipeline is a
 * known upstream limitation (see `StateManagementTest`). Model-gated on `SMOLLM2_MODEL`; skips
 * quietly otherwise.
 */
class OptimizedModeScheduleParityTest {

    private fun runtime(modelPath: String, schedule: Schedule, mode: OptimizedLLMMode): OptimizedLLMRuntime<FP32> {
        val ctx = DirectCpuExecutionContext(schedule = schedule)
        val model = runBlocking {
            LlamaNetworkLoader.fromGguf(
                randomAccessProvider = { JvmRandomAccessSource.open(modelPath) },
                weightForm = DECODER_DEQUANTIZE_ALL,
            ).load<FP32, Float>(ctx)
        }
        val rt = OptimizedLLMRuntime(model, ctx, mode, FP32::class)
        if (mode == OptimizedLLMMode.OPTIMIZED) {
            rt.compileUnoptimized().filter { it.contains("WARNING") || it.contains("Input node") }.forEach { println("  compile[${schedule.name}]: $it") }
        }
        return rt
    }

    @Test
    fun compiledLegIsScheduleIndependentAndMatchesEager() {
        val modelPath = System.getenv("SMOLLM2_MODEL")
        if (modelPath.isNullOrBlank()) { println("PARITY skipped: SMOLLM2_MODEL not set"); return }
        val hardware = CoroutineSchedule.hardware()
        val seq = runtime(modelPath, Schedule.Sequential, OptimizedLLMMode.OPTIMIZED)
        val par = runtime(modelPath, hardware, OptimizedLLMMode.OPTIMIZED)
        val eager = runtime(modelPath, hardware, OptimizedLLMMode.DIRECT)
        val tokens = intArrayOf(1, 504, 6124, 282, 4649, 314, 5623, 30)
        // Warm-up (#1261: the Panama reduceLanes order settles after the JIT's first pass), then clean state.
        for (rt in listOf(seq, par, eager)) { rt.forward(tokens[0]); rt.reset() }

        var maxDiffEager = 0f
        val msSeq = ArrayList<Double>(); val msPar = ArrayList<Double>()
        for ((step, t) in tokens.withIndex()) {
            val t0 = System.nanoTime(); val a = seq.forward(t).data.copyToFloatArray()
            val t1 = System.nanoTime(); val b = par.forward(t).data.copyToFloatArray()
            val t2 = System.nanoTime()
            msSeq += (t1 - t0) / 1e6; msPar += (t2 - t1) / 1e6
            assertContentEquals(a, b, "OPTIMIZED sequential vs hardware must be bit-identical at token $t")
            if (step == 0) {
                val e = eager.forward(t).data.copyToFloatArray()
                for (i in a.indices) maxDiffEager = maxOf(maxDiffEager, kotlin.math.abs(a[i] - e[i]))
            }
        }
        println("OPTIMIZED ms/step sequential=${"%.1f".format(msSeq.drop(1).average())} hardware(${hardware.parallelism})=${"%.1f".format(msPar.drop(1).average())}; max |OPTIMIZED - DIRECT| at position 0 = $maxDiffEager")
        assertTrue(maxDiffEager < 1e-3f, "compiled leg must agree with the eager leg at position 0 within rounding, got $maxDiffEager")
    }
}
