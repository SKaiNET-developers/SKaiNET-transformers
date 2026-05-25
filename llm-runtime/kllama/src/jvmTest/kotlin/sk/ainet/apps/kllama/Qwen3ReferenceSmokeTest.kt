package sk.ainet.apps.kllama

import java.io.File
import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.time.measureTime
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Assumptions.assumeTrue
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.types.FP32
import sk.ainet.models.qwen.QwenNetworkLoader

/**
 * Reference smoke test #1 — Qwen3-1.7B Q8_0 (kllama runner, GGUF).
 *
 * Locked in as part of the SKaiNET 0.25.0 bump. Exercises:
 *  - GGUF Q8_0 load path through the new 0.25.0 `Q8_0MatmulKernel` (BF16/Q8_0
 *    matmul kernels were re-platformed onto the `KernelRegistry` SPI in 0.25.0).
 *  - Decoder-only LLM generation via [OptimizedLLMRuntime] DIRECT mode.
 *  - Qwen-specific `RoPEMode.SPLIT_HALF` + QK-Norm code paths.
 *
 * Tagged `@Tag("smoke-reference")` so it runs only under
 * `./gradlew test -PsmokeReference -PincludeIntegration`. Self-skips via
 * [assumeTrue] when the model file is not present, so CI without the artifact
 * stays green.
 *
 * Model fallback chain (first match wins):
 *  1. `QWEN3_1B7_MODEL_PATH` env var
 *  2. `~/.cache/standapp/models/Qwen3-1.7B-Q8_0.gguf`
 *  3. Recursive scan under `~/.lmstudio/models/` for `Qwen3-1.7B-Q8_0.gguf`
 *  4. Recursive scan under `~/.cache/huggingface/hub/` for the same filename
 */
@Tag("smoke-reference")
@Tag("integration")
class Qwen3ReferenceSmokeTest {

    @Test
    fun `Qwen3-1_7B Q8_0 generates non-empty greedy continuation`() {
        val modelPath = locateModel()
        assumeTrue(modelPath != null, "No Qwen3-1.7B-Q8_0 GGUF found — set QWEN3_1B7_MODEL_PATH.")

        runBlocking {
            val ctx = DirectCpuExecutionContext()

            println("[smoke-reference] Loading Qwen3 tokenizer from $modelPath")
            val tokenizer = JvmRandomAccessSource.open(modelPath.toString()).use { source ->
                GGUFTokenizer.fromRandomAccessSource(source)
            }

            println("[smoke-reference] Loading Qwen3 model (Q8_0, DEQUANTIZE_TO_FP32)")
            val model = QwenNetworkLoader.fromGguf(
                randomAccessProvider = { JvmRandomAccessSource.open(modelPath.toString()) },
                quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
            ).load<FP32, Float>(ctx)

            val runtime = OptimizedLLMRuntime(
                model = model,
                ctx = ctx,
                mode = OptimizedLLMMode.DIRECT,
                dtype = FP32::class,
            )

            val prompt = "What is the capital of France?"
            val promptTokens = tokenizer.encode(prompt)
            val steps = 16

            val output = StringBuilder()
            val elapsed = measureTime {
                runtime.generate(prompt = promptTokens, steps = steps, temperature = 0.0f) { tokenId ->
                    output.append(tokenizer.decode(tokenId))
                }
            }
            val tokPerSec = steps.toDouble() / elapsed.inWholeMilliseconds * 1000
            println("[smoke-reference] Qwen3 produced '${output}' (${"%.2f".format(tokPerSec)} tok/s)")

            assertTrue(output.isNotBlank(), "Qwen3 produced blank text — generation pipeline broke")
        }
    }

    private fun locateModel(): Path? {
        val explicit = System.getenv("QWEN3_1B7_MODEL_PATH")?.trim().orEmpty()
        if (explicit.isNotEmpty()) {
            val p = Path.of(explicit)
            return if (p.exists()) p else null
        }
        val home = System.getProperty("user.home")
        val direct = listOf(
            Path.of(home, ".cache", "standapp", "models", "Qwen3-1.7B-Q8_0.gguf"),
        )
        for (p in direct) if (p.exists()) return p

        val searchRoots = listOf(
            Path.of(home, ".lmstudio", "models"),
            Path.of(home, ".cache", "huggingface", "hub"),
        )
        val targetName = "Qwen3-1.7B-Q8_0.gguf"
        for (root in searchRoots) {
            val rootFile = root.toFile()
            if (!rootFile.isDirectory) continue
            rootFile.walkTopDown()
                .filter { it.isFile && it.name == targetName }
                .firstOrNull()
                ?.let { return it.toPath() }
        }
        return null
    }
}
