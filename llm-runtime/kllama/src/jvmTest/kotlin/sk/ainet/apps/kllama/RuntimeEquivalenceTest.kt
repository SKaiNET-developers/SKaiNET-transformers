package sk.ainet.apps.kllama

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlinx.coroutines.runBlocking
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.model.QuantPolicy
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
        label: String = "logits"
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
                    quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
                    allowQuantized = true
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
                quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32
            ).load<FP32, Float>(ctx)

            val newRuntime = OptimizedLLMRuntime(
                model = dslModel,
                ctx = ctx,
                mode = OptimizedLLMRuntime.Mode.DIRECT,
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
                quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32
            ).load<FP32, Float>(ctx)

            val directRuntime = OptimizedLLMRuntime(
                model = directModel,
                ctx = ctx,
                mode = OptimizedLLMRuntime.Mode.DIRECT,
                dtype = FP32::class
            )

            // --- OPTIMIZED mode ---
            val optimizedModel = LlamaNetworkLoader.fromGguf(
                randomAccessProvider = { JvmRandomAccessSource.open(MODEL_PATH.toString()) },
                quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32
            ).load<FP32, Float>(ctx)

            val optimizedRuntime = OptimizedLLMRuntime(
                model = optimizedModel,
                ctx = ctx,
                mode = OptimizedLLMRuntime.Mode.OPTIMIZED,
                dtype = FP32::class
            )

            println("Compiling optimized graph...")
            val compileDuration = measureTime {
                val diagnostics = optimizedRuntime.compile()
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
                if (!compareLogits(refLogits, candLogits, tolerance = 1e-4f, label = "token=$tokenId")) {
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

            // Load tokenizer
            val tokenizer = JvmRandomAccessSource.open(MODEL_PATH.toString()).use { source ->
                GGUFTokenizer.fromRandomAccessSource(source)
            }

            val prompt = "The capital of France is"
            val promptTokens = tokenizer.encode(prompt)
            val steps = 16

            // --- Old LlamaRuntime ---
            val ingestion = LlamaIngestion<FP32>(
                ctx = ctx,
                dtype = FP32::class,
                config = LlamaLoadConfig(
                    quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
                    allowQuantized = true
                )
            )
            val oldWeights = ingestion.loadStreaming {
                JvmRandomAccessSource.open(MODEL_PATH.toString())
            }
            val backend = CpuAttentionBackend<FP32>(ctx, oldWeights, FP32::class)
            @Suppress("DEPRECATION")
            val oldRuntime = LlamaRuntime<FP32>(ctx, oldWeights, backend, FP32::class)

            // --- DIRECT mode ---
            val directModel = LlamaNetworkLoader.fromGguf(
                randomAccessProvider = { JvmRandomAccessSource.open(MODEL_PATH.toString()) },
                quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32
            ).load<FP32, Float>(ctx)
            val directRuntime = OptimizedLLMRuntime(
                model = directModel,
                ctx = ctx,
                mode = OptimizedLLMRuntime.Mode.DIRECT,
                dtype = FP32::class
            )

            // --- OPTIMIZED mode ---
            val optimizedModel = LlamaNetworkLoader.fromGguf(
                randomAccessProvider = { JvmRandomAccessSource.open(MODEL_PATH.toString()) },
                quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32
            ).load<FP32, Float>(ctx)
            val optimizedRuntime = OptimizedLLMRuntime(
                model = optimizedModel,
                ctx = ctx,
                mode = OptimizedLLMRuntime.Mode.OPTIMIZED,
                dtype = FP32::class
            )
            optimizedRuntime.compile()

            // --- Benchmark each ---
            fun benchmarkRuntime(name: String, run: () -> Unit): Double {
                // Warmup
                run()

                val duration = measureTime { run() }.inWholeMilliseconds
                val tokPerSec = steps.toDouble() / duration * 1000
                println("  $name: ${duration}ms (${"%.2f".format(tokPerSec)} tok/s)")
                return tokPerSec
            }

            println("=== Benchmark: $steps generation steps ===")
            println("Prompt: '$prompt'")

            val oldTps = benchmarkRuntime("LlamaRuntime (old)") {
                oldRuntime.reset()
                oldRuntime.generate(promptTokens, steps, 0.0f) { _ -> }
            }

            val directTps = benchmarkRuntime("OptimizedLLMRuntime DIRECT") {
                directRuntime.reset()
                directRuntime.generate(promptTokens, steps, 0.0f) { _ -> }
            }

            val optimizedTps = benchmarkRuntime("OptimizedLLMRuntime OPTIMIZED") {
                optimizedRuntime.reset()
                optimizedRuntime.generate(promptTokens, steps, 0.0f) { _ -> }
            }

            println("=== Relative Performance ===")
            println("  DIRECT vs old:     ${"%.2f".format(directTps / oldTps)}x")
            println("  OPTIMIZED vs old:  ${"%.2f".format(optimizedTps / oldTps)}x")
            println("  OPTIMIZED vs DIRECT: ${"%.2f".format(optimizedTps / directTps)}x")
            println("============================")

            assertTrue(oldTps > 0 && directTps > 0 && optimizedTps > 0)
        }
    }
}
