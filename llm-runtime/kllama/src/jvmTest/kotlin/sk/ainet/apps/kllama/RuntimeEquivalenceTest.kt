package sk.ainet.apps.kllama

import kotlin.math.abs
import kotlin.test.Test
import org.junit.jupiter.api.Tag
import kotlin.test.assertTrue
import kotlinx.coroutines.runBlocking
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.models.llama.DECODER_DEQUANTIZE_ALL
import sk.ainet.models.llama.LlamaNetworkLoader
import sk.ainet.models.llama.LlamaRuntime
import sk.ainet.lang.types.FP32
import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.time.measureTime

/**
 * End-to-end equivalence tests between old [LlamaRuntime] and new [OptimizedLLMRuntime].
 *
 * Loads TinyLlama 1.1B Q8_0 GGUF and compares:
 * 1. Old LlamaRuntime (hand-coded) vs OptimizedLLMRuntime DIRECT mode
 * 2. OptimizedLLMRuntime DIRECT vs OPTIMIZED mode
 *
 * Requires model at ~/.lmstudio/models/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/
 */
@Tag("integration")
class RuntimeEquivalenceTest {

    companion object {
        private val MODEL_PATH = Path.of(
            System.getProperty("user.home"),
            ".lmstudio/models/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
        )
    }

    private fun skipIfNoModel(): Boolean {
        if (!MODEL_PATH.exists()) {
            println("SKIPPING: Model not found at $MODEL_PATH")
            return true
        }
        return false
    }

    private fun compareLogits(
        reference: FloatArray,
        candidate: FloatArray,
        tolerance: Float = 1e-3f,
        label: String = "logits",
        verbose: Boolean = false
    ): Boolean {
        require(reference.size == candidate.size) {
            "Shape mismatch: reference=${reference.size} vs candidate=${candidate.size}"
        }
        var maxDiff = 0f
        var mismatches = 0
        for (i in reference.indices) {
            val diff = abs(reference[i] - candidate[i])
            if (diff > maxDiff) maxDiff = diff
            if (diff > tolerance) mismatches++
        }
        val mismatchFrac = if (reference.isNotEmpty()) mismatches.toFloat() / reference.size else 0f
        println("  [$label] maxDiff=${"%.6f".format(maxDiff)}, mismatches=$mismatches/${reference.size} (${"%.4f".format(mismatchFrac * 100)}%)")
        if (verbose || mismatchFrac > 0.01f) {
            val n = minOf(5, reference.size)
            println("    DIRECT first $n: ${reference.take(n).map { "%.6f".format(it) }}")
            println("    OPTIM  first $n: ${candidate.take(n).map { "%.6f".format(it) }}")
        }
        return mismatchFrac <= 0.01f
    }

    @Test
    fun `LlamaRuntime vs OptimizedLLMRuntime DIRECT mode`() {
        if (skipIfNoModel()) return
        runBlocking {
            val ctx = DirectCpuExecutionContext()

            // --- Old path: LlamaRuntime ---
            val ingestion = LlamaIngestion<FP32>(
                ctx = ctx,
                dtype = FP32::class,
                config = LlamaLoadConfig(
                )
            )
            val oldWeights = ingestion.loadStreaming {
                JvmRandomAccessSource.open(MODEL_PATH.toString())
            }
            val backend = CpuAttentionBackend<FP32>(ctx, oldWeights, FP32::class)
            @Suppress("DEPRECATION")
            val oldRuntime = LlamaRuntime<FP32>(ctx, oldWeights, backend, FP32::class)

            // --- New path: OptimizedLLMRuntime DIRECT ---
            val dslModel = LlamaNetworkLoader.fromGguf(
                randomAccessProvider = { JvmRandomAccessSource.open(MODEL_PATH.toString()) },
                weightForm = DECODER_DEQUANTIZE_ALL
            ).load<FP32, Float>(ctx)

            val newRuntime = OptimizedLLMRuntime(
                model = dslModel,
                ctx = ctx,
                mode = OptimizedLLMMode.DIRECT,
                dtype = FP32::class
            )

            // --- Compare first few tokens ---
            val testTokens = intArrayOf(1, 15043, 29892, 590) // <s>, Hello, ,, my

            println("=== LlamaRuntime vs OptimizedLLMRuntime DIRECT ===")
            var allPassed = true
            for (tokenId in testTokens) {
                val refLogits = oldRuntime.forward(tokenId).data.copyToFloatArray()
                val candLogits = newRuntime.forward(tokenId).data.copyToFloatArray()
                if (!compareLogits(refLogits, candLogits, label = "token=$tokenId")) {
                    allPassed = false
                }
            }

            assertTrue(allPassed, "LlamaRuntime vs DIRECT mode: some steps failed")
            println("=== PASS ===")
        }
    }

    @Test
    fun `OptimizedLLMRuntime DIRECT vs OPTIMIZED mode`() {
        if (skipIfNoModel()) return
        runBlocking {
            val ctx = DirectCpuExecutionContext()

            // --- DIRECT mode ---
            val directModel = LlamaNetworkLoader.fromGguf(
                randomAccessProvider = { JvmRandomAccessSource.open(MODEL_PATH.toString()) },
                weightForm = DECODER_DEQUANTIZE_ALL
            ).load<FP32, Float>(ctx)

            val directRuntime = OptimizedLLMRuntime(
                model = directModel,
                ctx = ctx,
                mode = OptimizedLLMMode.DIRECT,
                dtype = FP32::class
            )

            // --- OPTIMIZED mode ---
            val optimizedModel = LlamaNetworkLoader.fromGguf(
                randomAccessProvider = { JvmRandomAccessSource.open(MODEL_PATH.toString()) },
                weightForm = DECODER_DEQUANTIZE_ALL
            ).load<FP32, Float>(ctx)

            val optimizedRuntime = OptimizedLLMRuntime(
                model = optimizedModel,
                ctx = ctx,
                mode = OptimizedLLMMode.OPTIMIZED,
                dtype = FP32::class
            )

            println("Compiling optimized graph (unoptimized path for debugging)...")
            val compileDuration = measureTime {
                val diagnostics = optimizedRuntime.compileUnoptimized()
                println("  Diagnostics: ${diagnostics.size} messages")
                diagnostics.forEach { println("    - $it") }
            }
            println("  Compile time: $compileDuration")

            // --- Compare ---
            val testTokens = intArrayOf(1, 15043, 29892) // <s>, Hello, ,

            println("=== OptimizedLLMRuntime DIRECT vs OPTIMIZED ===")
            var allPassed = true
            for (tokenId in testTokens) {
                val refLogits = directRuntime.forward(tokenId).data.copyToFloatArray()
                val candLogits = optimizedRuntime.forward(tokenId).data.copyToFloatArray()
                if (!compareLogits(refLogits, candLogits, tolerance = 1e-4f, label = "token=$tokenId", verbose = true)) {
                    allPassed = false
                }
            }

            assertTrue(allPassed, "DIRECT vs OPTIMIZED: some steps failed")
            println("=== PASS ===")
        }
    }

    @Test
    fun `benchmark old vs direct vs optimized`() {
        if (skipIfNoModel()) return
        runBlocking {
            val ctx = DirectCpuExecutionContext()

            val tokenizer = JvmRandomAccessSource.open(MODEL_PATH.toString()).use { source ->
                GGUFTokenizer.fromRandomAccessSource(source)
            }

            val prompts = listOf(
                "Hello" to "short",
                "The capital of France is" to "medium",
                "Explain the theory of relativity in simple terms for a student who has never studied physics before" to "long"
            )
            val stepCounts = listOf(16, 64)
            val warmupRuns = 3
            val measuredRuns = 3

            // --- Load all three runtimes ---
            println("[BENCH] Loading runtimes...")

            val ingestion = LlamaIngestion<FP32>(
                ctx = ctx,
                dtype = FP32::class,
                config = LlamaLoadConfig(
                )
            )
            val oldWeights = ingestion.loadStreaming {
                JvmRandomAccessSource.open(MODEL_PATH.toString())
            }
            val backend = CpuAttentionBackend<FP32>(ctx, oldWeights, FP32::class)
            @Suppress("DEPRECATION")
            val oldRuntime = LlamaRuntime<FP32>(ctx, oldWeights, backend, FP32::class)

            val directModel = LlamaNetworkLoader.fromGguf(
                randomAccessProvider = { JvmRandomAccessSource.open(MODEL_PATH.toString()) },
                weightForm = DECODER_DEQUANTIZE_ALL
            ).load<FP32, Float>(ctx)
            val directRuntime = OptimizedLLMRuntime(
                model = directModel,
                ctx = ctx,
                mode = OptimizedLLMMode.DIRECT,
                dtype = FP32::class
            )

            var optimizedRuntime: OptimizedLLMRuntime<FP32>? = null
            try {
                val optimizedModel = LlamaNetworkLoader.fromGguf(
                    randomAccessProvider = { JvmRandomAccessSource.open(MODEL_PATH.toString()) },
                    weightForm = DECODER_DEQUANTIZE_ALL
                ).load<FP32, Float>(ctx)
                val rt = OptimizedLLMRuntime(
                    model = optimizedModel,
                    ctx = ctx,
                    mode = OptimizedLLMMode.OPTIMIZED,
                    dtype = FP32::class
                )
                rt.compile()
                // Verify execution works with a single forward pass
                rt.forward(1)
                rt.reset()
                optimizedRuntime = rt
            } catch (e: Exception) {
                println("[BENCH] OPTIMIZED mode failed: ${e.message}")
                println("[BENCH] Skipping OPTIMIZED benchmarks")
            }

            // --- Benchmark helper ---
            data class BenchResult(val name: String, val prompt: String, val steps: Int, val medianTokPerSec: Double)

            fun benchmarkRuntime(name: String, steps: Int, promptTokens: IntArray, run: (IntArray, Int) -> Unit): Double {
                // Warmup
                repeat(warmupRuns) { run(promptTokens, steps) }

                // Measured runs
                val times = (1..measuredRuns).map { measureTime { run(promptTokens, steps) }.inWholeMilliseconds }
                val median = times.sorted()[measuredRuns / 2]
                val tokPerSec = steps.toDouble() / median * 1000
                return tokPerSec
            }

            val results = mutableListOf<BenchResult>()

            println("[BENCH] TinyLlama 1.1B Q8_0 → FP32, CPU (Vector API SIMD)")
            println("[BENCH] ${warmupRuns} warmup + ${measuredRuns} measured runs, median reported")
            println()

            for (steps in stepCounts) {
                for ((prompt, label) in prompts) {
                    val promptTokens = tokenizer.encode(prompt)
                    println("[BENCH] steps=$steps, prompt=$label (${promptTokens.size} tokens)")

                    val oldTps = benchmarkRuntime("LlamaRuntime", steps, promptTokens) { pt, s ->
                        oldRuntime.reset(); oldRuntime.generate(pt, s, 0.0f) { _ -> }
                    }
                    val directTps = benchmarkRuntime("DIRECT", steps, promptTokens) { pt, s ->
                        directRuntime.reset(); directRuntime.generate(pt, s, 0.0f) { _ -> }
                    }
                    val optimizedTps = if (optimizedRuntime != null) {
                        benchmarkRuntime("OPTIMIZED", steps, promptTokens) { pt, s ->
                            optimizedRuntime.reset(); optimizedRuntime.generate(pt, s, 0.0f) { _ -> }
                        }
                    } else 0.0

                    results.add(BenchResult("LlamaRuntime", label, steps, oldTps))
                    results.add(BenchResult("DIRECT", label, steps, directTps))
                    if (optimizedTps > 0) results.add(BenchResult("OPTIMIZED", label, steps, optimizedTps))

                    println("  LlamaRuntime: ${"%.2f".format(oldTps)} tok/s")
                    println("  DIRECT:       ${"%.2f".format(directTps)} tok/s (${"%.2f".format(directTps / oldTps)}x)")
                    if (optimizedTps > 0) println("  OPTIMIZED:    ${"%.2f".format(optimizedTps)} tok/s (${"%.2f".format(optimizedTps / oldTps)}x)")
                    else println("  OPTIMIZED:    SKIPPED")
                    println()
                }
            }

            // --- Summary table ---
            println("[BENCH] ========== SUMMARY ==========")
            println("[BENCH] | Runtime      | Steps | Short  | Medium | Long   |")
            println("[BENCH] |--------------|-------|--------|--------|--------|")
            for (name in listOf("LlamaRuntime", "DIRECT", "OPTIMIZED")) {
                for (steps in stepCounts) {
                    val cols = prompts.map { (_, label) ->
                        val r = results.find { it.name == name && it.prompt == label && it.steps == steps }
                        r?.let { "${"%.2f".format(it.medianTokPerSec)}" } ?: "N/A"
                    }
                    println("[BENCH] | %-12s | %5d | %6s | %6s | %6s |".format(name, steps, cols[0], cols[1], cols[2]))
                }
            }
            println("[BENCH] ================================")

            assertTrue(results.all { it.medianTokPerSec > 0 })
        }
    }
}
