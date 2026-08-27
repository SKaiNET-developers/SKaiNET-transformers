package sk.ainet.apps.kllama

import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlinx.coroutines.runBlocking
import sk.ainet.apps.kllama.agent.GenerateResult
import sk.ainet.apps.kllama.agent.generateUntilStop
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.PrefillStrategy
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.types.FP32
import sk.ainet.models.llama.DECODER_DEQUANTIZE_ALL
import sk.ainet.models.llama.LlamaNetworkLoader

/**
 * Verifies that `generateUntilStop(prefillStrategy = Batched)` produces the
 * same greedy token sequence as the autoregressive default. This is the
 * agent-loop-level counterpart to [PrefillStrategyEquivalenceTest] (which
 * covers llm-core's `generate`): `generateUntilStop` is what [sk.ainet.apps.kllama.chat.AgentLoop]
 * and `KLlamaSession.generate` actually call, and it historically kept a
 * hardcoded autoregressive prefill long after the batched path reached
 * parity — this test pins the wiring so it cannot silently regress.
 *
 * Both the single-chunk and the forced multi-chunk (`maxBatch` < prompt
 * length) paths are covered in ONE model pairing per test to keep the peak
 * footprint at two loaded models per JVM.
 *
 * Skipped if the model is not present.
 */
class GenerateUntilStopPrefillEquivalenceTest {

    companion object {
        private val MODEL_PATH = Path.of(
            System.getProperty("user.home"),
            ".lmstudio/models/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
        )
    }

    private suspend fun greedyRun(prefillStrategy: PrefillStrategy, promptTokens: IntArray, maxTokens: Int): GenerateResult {
        val ctx = DirectCpuExecutionContext()
        val model = LlamaNetworkLoader.fromGguf(
            randomAccessProvider = { JvmRandomAccessSource.open(MODEL_PATH.toString()) },
            weightForm = DECODER_DEQUANTIZE_ALL
        ).load<FP32, Float>(ctx)
        val runtime = OptimizedLLMRuntime(
            model = model, ctx = ctx,
            mode = OptimizedLLMMode.DIRECT, dtype = FP32::class
        )
        return runtime.generateUntilStop(
            prompt = promptTokens,
            maxTokens = maxTokens,
            eosTokenId = 2, // llama </s>
            temperature = 0f,
            random = Random(0),
            prefillStrategy = prefillStrategy
        )
    }

    @Test
    fun `Batched prefill yields identical greedy generateUntilStop output`() {
        if (!MODEL_PATH.exists()) {
            println("[skip] Model not at $MODEL_PATH")
            return
        }
        runBlocking {
            val tokenizer = JvmRandomAccessSource.open(MODEL_PATH.toString()).use { source ->
                GGUFTokenizer.fromRandomAccessSource(source)
            }
            val promptTokens = tokenizer.encode("The capital of France is")
            val maxTokens = 12

            val auto = greedyRun(PrefillStrategy.Autoregressive, promptTokens, maxTokens)
            val batched = greedyRun(PrefillStrategy.Batched(maxBatch = 64), promptTokens, maxTokens)

            println("[diag] auto    : ${auto.tokens}")
            println("[diag] batched : ${batched.tokens}")
            assertEquals(auto.tokens, batched.tokens,
                "Batched generateUntilStop diverges from autoregressive at temperature=0")
            assertEquals(auto.stoppedByEos, batched.stoppedByEos)
        }
    }

    @Test
    fun `Chunked batched prefill yields identical greedy generateUntilStop output`() {
        if (!MODEL_PATH.exists()) {
            println("[skip] Model not at $MODEL_PATH")
            return
        }
        runBlocking {
            val tokenizer = JvmRandomAccessSource.open(MODEL_PATH.toString()).use { source ->
                GGUFTokenizer.fromRandomAccessSource(source)
            }
            val promptTokens = tokenizer.encode("The capital of France is Paris and")
            val maxTokens = 8

            val auto = greedyRun(PrefillStrategy.Autoregressive, promptTokens, maxTokens)
            // maxBatch < prompt length forces the multi-chunk path, i.e.
            // forwardBatched with a non-empty KV cache (seqKV > seqQ).
            val chunked = greedyRun(PrefillStrategy.Batched(maxBatch = 3), promptTokens, maxTokens)

            println("[diag] auto (chunked)   : ${auto.tokens}")
            println("[diag] batched (chunked): ${chunked.tokens}")
            assertEquals(auto.tokens, chunked.tokens,
                "Chunked batched generateUntilStop diverges from autoregressive at temperature=0")
        }
    }
}
