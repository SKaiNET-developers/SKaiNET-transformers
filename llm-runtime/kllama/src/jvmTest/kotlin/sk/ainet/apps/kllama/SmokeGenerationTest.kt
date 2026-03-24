package sk.ainet.apps.kllama

import kotlin.test.Test
import kotlin.test.assertTrue
import kotlinx.coroutines.runBlocking
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.models.llama.LlamaNetworkLoader
import sk.ainet.lang.types.FP32
import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.time.measureTime

class SmokeGenerationTest {

    companion object {
        private val MODEL_PATH = Path.of(
            System.getProperty("user.home"),
            ".lmstudio/models/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
        )
    }

    @Test
    fun `generate text with OptimizedLLMRuntime`() {
        if (!MODEL_PATH.exists()) { println("SKIPPING"); return }
        runBlocking {
            val ctx = DirectCpuExecutionContext()

            println("Loading tokenizer...")
            val tokenizer = JvmRandomAccessSource.open(MODEL_PATH.toString()).use { source ->
                GGUFTokenizer.fromRandomAccessSource(source)
            }

            println("Loading model (new DSL path)...")
            val model = LlamaNetworkLoader.fromGguf(
                randomAccessProvider = { JvmRandomAccessSource.open(MODEL_PATH.toString()) },
                quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32
            ).load<FP32, Float>(ctx)

            val runtime = OptimizedLLMRuntime(
                model = model,
                ctx = ctx,
                mode = OptimizedLLMMode.DIRECT,
                dtype = FP32::class
            )

            // Test: factual question
            val prompt = "The capital of France is"
            val promptTokens = tokenizer.encode(prompt)
            println("Prompt: '$prompt' (${promptTokens.size} tokens)")

            val output = StringBuilder()
            val steps = 32
            print(prompt)
            val elapsed = measureTime {
                runtime.generate(prompt = promptTokens, steps = steps, temperature = 0.0f) { tokenId ->
                    val decoded = tokenizer.decode(tokenId)
                    output.append(decoded)
                    print(decoded)
                }
            }
            val tokPerSec = steps.toDouble() / elapsed.inWholeMilliseconds * 1000
            println("\n--- ${"%.2f".format(tokPerSec)} tok/s ---")
            println("Full output: $output")

            assertTrue(output.contains("Paris", ignoreCase = true),
                "Expected output to mention Paris, got: $output")
            println("SMOKE TEST PASSED - OptimizedLLMRuntime generates correct text!")
        }
    }
}
