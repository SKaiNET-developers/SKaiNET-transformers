package sk.ainet.apps.kgemma

import kotlinx.coroutines.runBlocking
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.nn.transformer.PhaseProfile
import sk.ainet.lang.tensor.data.MemorySegmentTensorDataFactory
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.time.measureTime

/**
 * Where decode time actually goes on Gemma 4 E2B Q4_K_M.
 *
 * Reports prefill and decode separately — the CLI's single tok/s number divides generated tokens
 * by wall time that includes prefill, which flatters short runs and hides which half is slow —
 * then prints the [PhaseProfile] breakdown for the decode steps alone.
 *
 * Reference on this machine (llama.cpp b10621, same checkpoint, `llama-bench`):
 * Metal 384 pp / 103 tg tok/s; **CPU-only 144 pp / 79 tg tok/s** — the CPU number is the fair
 * comparison for the eager path.
 *
 * `GEMMA4_SPEED_TOKENS` overrides the decode length (default 24).
 */
class Gemma4SpeedProfile {

    @Test
    fun prefill_and_decode_breakdown() {
        val gguf = System.getenv("GEMMA4_E2B_GGUF_PATH")
        if (gguf.isNullOrBlank() || System.getenv("GEMMA4_SPEED") != "1") {
            println("[skip] set GEMMA4_E2B_GGUF_PATH and GEMMA4_SPEED=1"); return
        }
        val decodeTokens = System.getenv("GEMMA4_SPEED_TOKENS")?.toIntOrNull() ?: 24

        val factory = MemorySegmentTensorDataFactory()
        val ctx = DirectCpuExecutionContext(tensorDataFactory = factory)
        try {
            val loadTime = measureTime {
                runtime = runBlocking {
                    Gemma4Ingestion<FP32>(ctx = ctx, dtype = FP32::class, config = Gemma4LoadConfig())
                        .loadDslRuntimeStreaming { JvmRandomAccessSource.open(gguf) }
                }
            }
            val rt = runtime!!
            println("SPEED load: ${loadTime.inWholeMilliseconds} ms")

            // The rendered single-turn chat prompt from the golden-token gate.
            val prompt = intArrayOf(
                2, 105, 2364, 107, 40414, 756, 23391, 1902, 236789, 531, 9115, 106, 107, 105, 4368, 107,
            )

            // Warm up JIT on a throwaway pass so the measured numbers are steady-state.
            rt.reset()
            for (id in prompt) rt.forward(id)
            repeat(3) { rt.forward(818) }

            rt.reset()
            PhaseProfile.reset()
            var logits = rt.forward(prompt[0])
            val prefill = measureTime {
                for (i in 1 until prompt.size) logits = rt.forward(prompt[i])
            }
            val prefillReport = PhaseProfile.report()

            PhaseProfile.reset()
            sk.ainet.exec.tensor.ops.KernelProfile.reset()
            var token = 818
            val decode = measureTime {
                repeat(decodeTokens) {
                    val l = rt.forward(token)
                    val buf = l.data.copyToFloatArray()
                    var best = 0
                    for (i in buf.indices) if (buf[i] > buf[best]) best = i
                    token = best
                }
            }

            val prefillPerTok = prefill.inWholeMicroseconds / 1000.0 / (prompt.size - 1)
            val decodePerTok = decode.inWholeMicroseconds / 1000.0 / decodeTokens
            println("SPEED prefill: ${prompt.size - 1} tokens in ${prefill.inWholeMilliseconds} ms " +
                "= %.1f ms/tok = %.2f tok/s".format(prefillPerTok, 1000.0 / prefillPerTok))
            println("SPEED decode : $decodeTokens tokens in ${decode.inWholeMilliseconds} ms " +
                "= %.1f ms/tok = %.2f tok/s".format(decodePerTok, 1000.0 / decodePerTok))
            println("SPEED reference (llama.cpp CPU-only, same checkpoint): 144 pp / 79 tg tok/s")
            println("SPEED --- prefill phases ---")
            println(prefillReport)
            println("SPEED --- matmul dispatch paths (which branch serves them) ---")
            println(sk.ainet.exec.tensor.ops.KernelProfile.report())
            println("SPEED --- decode phases ---")
            println(PhaseProfile.report())
        } finally {
            factory.close()
        }
    }

    private var runtime: sk.ainet.apps.llm.InferenceRuntime<FP32>? = null
}
