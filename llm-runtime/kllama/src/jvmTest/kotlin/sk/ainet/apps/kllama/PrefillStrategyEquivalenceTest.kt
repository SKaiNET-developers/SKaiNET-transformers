package sk.ainet.apps.kllama

import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlinx.coroutines.runBlocking
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.PrefillStrategy
import sk.ainet.apps.llm.generate
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.types.FP32
import sk.ainet.lang.nn.dsl.decoder.DECODER_DEQUANTIZE_ALL
import sk.ainet.models.llama.LlamaNetworkLoader

/**
 * Verifies that `generate(prefillStrategy = Batched)` produces the same
 * generated tokens as `generate(prefillStrategy = Autoregressive)` at
 * temperature=0 (greedy) on a fixed prompt. This is the user-facing
 * counterpart to [BatchedPrefillEquivalenceTest], which exercises the
 * lower-level `forwardBatched` directly.
 *
 * Without this gate, `generate()` could ship a default that silently
 * picks the wrong path, or a future change could re-introduce a `seqLen
 * > 1`-only bug akin to the one fixed in PR #81.
 *
 * Skipped if the model is not present.
 */
class PrefillStrategyEquivalenceTest {

    companion object {
        private val MODEL_PATH = Path.of(
            System.getProperty("user.home"),
            ".lmstudio/models/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
        )
    }

    @Test
    fun `Batched prefill yields identical greedy generation`() {
        if (!MODEL_PATH.exists()) {
            println("[skip] Model not at $MODEL_PATH")
            return
        }
        runBlocking {
            val ctx = DirectCpuExecutionContext()
            val tokenizer = JvmRandomAccessSource.open(MODEL_PATH.toString()).use { source ->
                GGUFTokenizer.fromRandomAccessSource(source)
            }
            val prompt = "The capital of France is"
            val promptTokens = tokenizer.encode(prompt)
            val steps = 16

            val auto = mutableListOf<Int>()
            run {
                val model = LlamaNetworkLoader.fromGguf(
                    randomAccessProvider = { JvmRandomAccessSource.open(MODEL_PATH.toString()) },
                    weightForm = DECODER_DEQUANTIZE_ALL
                ).load<FP32, Float>(ctx)
                val runtime = OptimizedLLMRuntime(
                    model = model, ctx = ctx,
                    mode = OptimizedLLMMode.DIRECT, dtype = FP32::class
                )
                runtime.generate(
                    prompt = promptTokens,
                    steps = steps,
                    temperature = 0f,
                    random = Random(0),
                    prefillStrategy = PrefillStrategy.Autoregressive
                ) { auto.add(it) }
            }

            val batched = mutableListOf<Int>()
            run {
                val model = LlamaNetworkLoader.fromGguf(
                    randomAccessProvider = { JvmRandomAccessSource.open(MODEL_PATH.toString()) },
                    weightForm = DECODER_DEQUANTIZE_ALL
                ).load<FP32, Float>(ctx)
                val runtime = OptimizedLLMRuntime(
                    model = model, ctx = ctx,
                    mode = OptimizedLLMMode.DIRECT, dtype = FP32::class
                )
                runtime.generate(
                    prompt = promptTokens,
                    steps = steps,
                    temperature = 0f,
                    random = Random(0),
                    prefillStrategy = PrefillStrategy.Batched(maxBatch = 64)
                ) { batched.add(it) }
            }

            println("[diag] auto    : $auto")
            println("[diag] batched : $batched")
            assertEquals(auto, batched, "Batched generation diverges from autoregressive at temperature=0")
        }
    }

    @Test
    fun `Batched prefill with chunking yields identical greedy generation`() {
        // Force chunked prefill (maxBatch < prompt length) to exercise the
        // multi-chunk path through forwardBatched.
        if (!MODEL_PATH.exists()) {
            println("[skip] Model not at $MODEL_PATH")
            return
        }
        runBlocking {
            val ctx = DirectCpuExecutionContext()
            val tokenizer = JvmRandomAccessSource.open(MODEL_PATH.toString()).use { source ->
                GGUFTokenizer.fromRandomAccessSource(source)
            }
            val prompt = "The capital of France is Paris and"
            val promptTokens = tokenizer.encode(prompt)
            val steps = 8

            val auto = mutableListOf<Int>()
            run {
                val model = LlamaNetworkLoader.fromGguf(
                    randomAccessProvider = { JvmRandomAccessSource.open(MODEL_PATH.toString()) },
                    weightForm = DECODER_DEQUANTIZE_ALL
                ).load<FP32, Float>(ctx)
                val runtime = OptimizedLLMRuntime(
                    model = model, ctx = ctx,
                    mode = OptimizedLLMMode.DIRECT, dtype = FP32::class
                )
                runtime.generate(
                    prompt = promptTokens,
                    steps = steps,
                    temperature = 0f,
                    random = Random(0),
                    prefillStrategy = PrefillStrategy.Autoregressive
                ) { auto.add(it) }
            }

            val batched = mutableListOf<Int>()
            run {
                val model = LlamaNetworkLoader.fromGguf(
                    randomAccessProvider = { JvmRandomAccessSource.open(MODEL_PATH.toString()) },
                    weightForm = DECODER_DEQUANTIZE_ALL
                ).load<FP32, Float>(ctx)
                val runtime = OptimizedLLMRuntime(
                    model = model, ctx = ctx,
                    mode = OptimizedLLMMode.DIRECT, dtype = FP32::class
                )
                runtime.generate(
                    prompt = promptTokens,
                    steps = steps,
                    temperature = 0f,
                    random = Random(0),
                    prefillStrategy = PrefillStrategy.Batched(maxBatch = 3)  // forces 2-3 chunks
                ) { batched.add(it) }
            }

            println("[diag] auto (chunked):    $auto")
            println("[diag] batched (chunked): $batched")
            assertEquals(auto, batched, "Chunked batched generation diverges from autoregressive")
        }
    }
}
