package sk.ainet.apps.kgemma

import kotlinx.coroutines.runBlocking
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.nn.hooks.ForwardHooks
import sk.ainet.lang.nn.topology.ModuleNode
import sk.ainet.lang.tensor.data.MemorySegmentTensorDataFactory
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.time.measureTime

/**
 * Complete decode-time breakdown by module, using forward hooks rather than the hand-placed
 * [sk.ainet.lang.nn.transformer.PhaseProfile] buckets.
 *
 * Why: `PhaseProfile` only covers call sites someone remembered to wrap, and after the fused
 * RMS-norm fix its buckets accounted for **8.6% of decode wall time** — the FFN, which carries
 * most of a transformer's FLOPs, has no bucket at all. Hooks fire for every module, so nothing
 * hides.
 *
 * Reports **self time** (exclusive of children), so a block's cost is attributed to the leaf that
 * actually spent it.
 */
class Gemma4ModuleTimingProfile {

    private class Timing : ForwardHooks {
        private class Frame(val name: String, val start: Long) { var childNanos: Long = 0 }
        private val stack = ArrayDeque<Frame>()
        val selfNanos = LinkedHashMap<String, Long>()
        val calls = LinkedHashMap<String, Long>()
        var enabled = false

        override fun onForwardBegin(module: ModuleNode, input: Any) {
            if (!enabled) return
            stack.addLast(Frame(label(module), System.nanoTime()))
        }

        override fun onForwardEnd(module: ModuleNode, input: Any, output: Any) {
            if (!enabled || stack.isEmpty()) return
            val f = stack.removeLast()
            val elapsed = System.nanoTime() - f.start
            selfNanos[f.name] = (selfNanos[f.name] ?: 0L) + (elapsed - f.childNanos)
            calls[f.name] = (calls[f.name] ?: 0L) + 1L
            stack.lastOrNull()?.let { it.childNanos += elapsed }
        }

        /** Collapse per-layer ids (`blk.17.ffn`) onto one row per role. */
        private fun label(m: ModuleNode): String {
            val raw = m.id.ifBlank { m.name }
            return raw.replace(Regex("""blk\.\d+\.?"""), "blk.*.").ifBlank { "?" }
        }
    }

    @Test
    fun decode_time_by_module() {
        val gguf = System.getenv("GEMMA4_E2B_GGUF_PATH")
        if (gguf.isNullOrBlank() || System.getenv("GEMMA4_SPEED") != "1") {
            println("[skip] set GEMMA4_E2B_GGUF_PATH and GEMMA4_SPEED=1"); return
        }
        val steps = System.getenv("GEMMA4_SPEED_TOKENS")?.toIntOrNull() ?: 8

        val timing = Timing()
        val factory = MemorySegmentTensorDataFactory()
        val ctx = DirectCpuExecutionContext(tensorDataFactory = factory, _hooks = timing)
        try {
            val rt = runBlocking {
                GemmaIngestion<FP32>(ctx = ctx, dtype = FP32::class, config = Gemma4LoadConfig())
                    .loadDslRuntimeStreaming { JvmRandomAccessSource.open(gguf) }
            }
            val prompt = intArrayOf(
                2, 105, 2364, 107, 40414, 756, 23391, 1902, 236789, 531, 9115, 106, 107, 105, 4368, 107,
            )
            rt.reset()
            for (id in prompt) rt.forward(id)
            repeat(2) { rt.forward(818) }          // warm up

            timing.enabled = true
            var token = 818
            val wall = measureTime {
                repeat(steps) {
                    val l = rt.forward(token)
                    val b = l.data.copyToFloatArray()
                    var best = 0
                    for (i in b.indices) if (b[i] > b[best]) best = i
                    token = best
                }
            }
            timing.enabled = false

            val total = timing.selfNanos.values.sum()
            println("MODT decode ${steps} tokens in ${wall.inWholeMilliseconds} ms " +
                "= %.1f ms/tok; hooks account for %.1f%% of it".format(
                    wall.inWholeMilliseconds.toDouble() / steps,
                    100.0 * total / wall.inWholeNanoseconds,
                ))
            println("MODT self-time by module (exclusive of children):")
            timing.selfNanos.entries.sortedByDescending { it.value }.take(14).forEach { (name, ns) ->
                println("MODT   %-26s %8.1f ms  %5.1f%%  over %d calls".format(
                    name, ns / 1e6, 100.0 * ns / total, timing.calls[name] ?: 0L,
                ))
            }
        } finally {
            factory.close()
        }
    }
}
