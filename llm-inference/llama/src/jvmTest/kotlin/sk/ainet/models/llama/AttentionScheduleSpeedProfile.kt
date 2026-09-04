package sk.ainet.models.llama

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlinx.coroutines.runBlocking
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.tokenizer.TokenizerFactory
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.schedule.Schedule
import sk.ainet.exec.schedule.CoroutineSchedule
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.lang.nn.dsl.decoder.DecoderKVCacheKind
import sk.ainet.lang.nn.transformer.PhaseProfile
import sk.ainet.lang.types.FP32

/**
 * SKEEP-005 measurement lane (transformers#412/#413): the same model, prompt and greedy decode
 * under every {schedule} × {KV cache} combination, printing tokens/s and the [PhaseProfile]
 * attention buckets, and asserting the greedy token sequence is identical across all four.
 *
 * Opt-in: runs only with `ATTN_SCHEDULE_SPEED=1` and a GGUF in `ATTN_SPEED_GGUF` (falls back to
 * `LLAMA32_1B_GGUF`). Tune with `ATTN_SPEED_PREFILL` (default 64) and `ATTN_SPEED_DECODE` (32).
 */
class AttentionScheduleSpeedProfile {

    private data class Config(val label: String, val schedule: Schedule, val kv: DecoderKVCacheKind)

    private data class Run(val tokens: List<Int>, val prefillMs: Double, val decodeMs: Double, val report: String)

    @Test
    fun scheduleAndCacheVariantsAgreeAndReportSpeed() {
        if (System.getenv("ATTN_SCHEDULE_SPEED") != "1") {
            println("ATTN_SCHEDULE_SPEED not set — profile skipped")
            return
        }
        val modelPath = System.getenv("ATTN_SPEED_GGUF")?.takeIf { it.isNotBlank() }
            ?: System.getenv("LLAMA32_1B_GGUF")?.takeIf { it.isNotBlank() }
            ?: run { println("no GGUF in ATTN_SPEED_GGUF / LLAMA32_1B_GGUF — profile skipped"); return }
        val prefillLen = (System.getenv("ATTN_SPEED_PREFILL") ?: "64").toInt()
        val decodeSteps = (System.getenv("ATTN_SPEED_DECODE") ?: "32").toInt()

        val fields = StreamingGGUFReader.open(JvmRandomAccessSource.open(modelPath)).use { it.fields }
        val tokenizer = TokenizerFactory.fromGgufFields(fields)
        val text = "The daily stand-up is a short meeting in which every team member reports what was done " +
            "yesterday, what is planned for today and which obstacles are in the way. "
        val encoded = tokenizer.encode(text.repeat(8))
        val prompt = IntArray(prefillLen) { encoded[it % encoded.size] }

        val hardware = CoroutineSchedule.hardware()
        val configs = listOf(
            Config("sequential/append", Schedule.Sequential, DecoderKVCacheKind.APPEND),
            Config("sequential/positional", Schedule.Sequential, DecoderKVCacheKind.POSITIONAL),
            Config("${hardware.name}/append", hardware, DecoderKVCacheKind.APPEND),
            Config("${hardware.name}/positional", hardware, DecoderKVCacheKind.POSITIONAL),
        )
        val runs = LinkedHashMap<String, Run>()
        for (cfg in configs) {
            val ctx = DirectCpuExecutionContext(schedule = cfg.schedule)
            val weights = runBlocking {
                LlamaWeightLoader.loadToMapStreaming<FP32, Float>(ctx, { JvmRandomAccessSource.open(modelPath) })
            }
            val runtime = OptimizedLLMRuntime(
                LlamaNetworkLoader.fromWeights(weights, kvCacheKind = cfg.kv), ctx,
                OptimizedLLMMode.DIRECT, FP32::class, bos = weights.metadata.bosTokenId,
            )
            // Warm-up: JIT the kernels on a short forward, then start from a clean cache.
            runtime.forwardBatched(prompt.copyOf(8)); runtime.forward(prompt[8]); runtime.reset()
            PhaseProfile.reset()

            val tokens = ArrayList<Int>(decodeSteps)
            val t0 = System.nanoTime()
            var next = argmax(runtime.forwardBatched(prompt).data.copyToFloatArray())
            val t1 = System.nanoTime()
            repeat(decodeSteps) {
                tokens += next
                next = argmax(runtime.forward(next).data.copyToFloatArray())
            }
            val t2 = System.nanoTime()
            runs[cfg.label] = Run(tokens, (t1 - t0) / 1e6, (t2 - t1) / 1e6, PhaseProfile.report())
        }

        println("=== AttentionScheduleSpeedProfile: $modelPath prefill=$prefillLen decode=$decodeSteps")
        for ((label, r) in runs) {
            val tps = decodeSteps / (r.decodeMs / 1000.0)
            println("--- $label: prefill ${"%.0f".format(r.prefillMs)} ms, decode ${"%.0f".format(r.decodeMs)} ms (${"%.2f".format(tps)} tok/s)")
            println(r.report.lines().filter { it.contains("attn.") || it.contains("phase total") }.joinToString("\n"))
        }
        val reference = runs.values.first().tokens
        for ((label, r) in runs) assertEquals(reference, r.tokens, "greedy tokens must not depend on the schedule/cache ($label)")
        println("greedy tokens identical across ${runs.size} configurations: ${tokenizer.decode(reference.toIntArray())}")
    }

    private fun argmax(logits: FloatArray): Int {
        var best = 0
        for (i in 1 until logits.size) if (logits[i] > logits[best]) best = i
        return best
    }
}
