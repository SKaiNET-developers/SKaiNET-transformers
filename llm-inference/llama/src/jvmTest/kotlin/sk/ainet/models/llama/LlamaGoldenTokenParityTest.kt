package sk.ainet.models.llama

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlinx.coroutines.runBlocking
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.tokenizer.TokenizerFactory
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.lang.types.FP32

/**
 * The #346 maturity-gate parity probe for the Llama family (transformers#373), against
 * **mainline llama.cpp** — Q8_0 is its native case, so unlike the BitNet gate this one asserts
 * FULL cross-implementation greedy text equality, on the DSL path the CLIs ship
 * ([LlamaWeightLoader]-style engine loading → [LlamaNetworkLoader.fromWeights] →
 * [OptimizedLLMRuntime]) — deliberately NOT the deprecated `LlamaRuntime` (#354).
 *
 * Model-gated: runs only when `LLAMA32_1B_GGUF` points at `Llama-3.2-1B-Instruct-Q8_0.gguf`;
 * skips quietly otherwise. The fixture header records the exact oracle build and commands.
 */
class LlamaGoldenTokenParityTest {

    private data class Fixture(
        val prompt: String,
        val steps: Int,
        val promptTokens: List<Int>,
        val oracleText: String,
    )

    private fun loadFixture(): Fixture {
        val raw = checkNotNull(javaClass.getResourceAsStream("/llama32-1b/golden-greedy-1b.txt")) {
            "fixture /llama32-1b/golden-greedy-1b.txt missing from test resources"
        }.bufferedReader().readLines()
        val map = raw.filter { it.isNotBlank() && !it.startsWith("#") }
            .associate { it.substringBefore('=') to it.substringAfter('=') }
        return Fixture(
            prompt = map.getValue("prompt"),
            steps = map.getValue("steps").toInt(),
            promptTokens = map.getValue("prompt_tokens").split(',').map { it.trim().toInt() },
            oracleText = map.getValue("oracle_text"),
        )
    }

    @Test
    fun greedyDecodeMatchesMainlineLlamaCpp() {
        val modelPath = System.getenv("LLAMA32_1B_GGUF")
        if (modelPath.isNullOrBlank()) {
            println("PARITY skipped: LLAMA32_1B_GGUF not set")
            return
        }
        val fixture = loadFixture()
        val ctx = ParityEnv.context()
        println("PARITY llama ${ParityEnv.describe()}")

        // 1 — prompt tokenization parity (a failure here names the tokenizer, not the model).
        val fields = StreamingGGUFReader.open(JvmRandomAccessSource.open(modelPath)).use { it.fields }
        val tokenizer = TokenizerFactory.fromGgufFields(fields)
        val raw = tokenizer.encode(fixture.prompt)
        val promptTokens = if (raw.isNotEmpty() && raw[0] == tokenizer.bosTokenId) raw
        else intArrayOf(tokenizer.bosTokenId) + raw
        assertEquals(
            fixture.promptTokens, promptTokens.toList(),
            "prompt tokenization must match mainline llama.cpp",
        )

        // 2 — greedy continuation, DSL path, keep-packed engine loading.
        val weights = runBlocking {
            LlamaWeightLoader.loadToMapStreaming<FP32, Float>(
                ctx, { JvmRandomAccessSource.open(modelPath) },
            )
        }
        val runtime = OptimizedLLMRuntime(
            LlamaNetworkLoader.fromWeights(weights, kvCacheKind = ParityEnv.kvCacheKind), ctx,
            OptimizedLLMMode.DIRECT, FP32::class, bos = weights.metadata.bosTokenId,
        )
        val text = StringBuilder()
        runtime.generate(promptTokens, fixture.steps, temperature = 0f) { text.append(tokenizer.decode(it)) }
        assertEquals(
            fixture.oracleText, text.toString(),
            "greedy decode must equal mainline llama.cpp token-for-token",
        )
    }
}
