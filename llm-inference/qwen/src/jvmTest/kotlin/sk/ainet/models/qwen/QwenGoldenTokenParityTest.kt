package sk.ainet.models.qwen

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlinx.coroutines.runBlocking
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.sampleFromTensor
import sk.ainet.apps.llm.tokenizer.TokenizerFactory
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.lang.types.FP32

/**
 * The #346 maturity-gate parity probe for the Qwen family, against **mainline llama.cpp** —
 * Q8_0 is its native case, so like the Llama gate this asserts FULL cross-implementation
 * greedy text equality on the DSL path ([QwenWeightLoader] engine loading →
 * [QwenNetworkLoader.fromWeights] → [OptimizedLLMRuntime]).
 *
 * Two variants, because they exercise disjoint machinery:
 * - **Qwen2.5** (`QWEN25_05B_GGUF`) — attention projection biases, no QK-norm. This is the
 *   #352 regression gate: the biases are enormous (blk.0 K bias moves the channel sum from
 *   ~11 to ~507), so any break in bias loading/binding turns the output to garbage.
 * - **Qwen3** (`QWEN3_17B_GGUF`) — QK-norm, no biases.
 *
 * Model-gated: each variant runs only when its env var points at the Q8_0 GGUF; skips
 * quietly otherwise. Fixture headers record the exact oracle build and commands. Qwen adds
 * no BOS token — generation feeds the fixture's raw prompt tokens, prepending nothing.
 */
@org.junit.jupiter.api.Tag("smoke-reference")
@org.junit.jupiter.api.Tag("integration")
class QwenGoldenTokenParityTest {

    private data class Fixture(
        val prompt: String,
        val steps: Int,
        val promptTokens: List<Int>,
        val oracleText: String,
    )

    private fun loadFixture(resource: String): Fixture {
        val raw = checkNotNull(javaClass.getResourceAsStream(resource)) {
            "fixture $resource missing from test resources"
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

    private fun assertParity(modelPath: String, fixture: Fixture) {
        val ctx = ParityEnv.context()
        println("PARITY qwen ${ParityEnv.describe()}")

        // 1 — prompt tokenization parity (a failure here names the tokenizer, not the model).
        val fields = StreamingGGUFReader.open(JvmRandomAccessSource.open(modelPath)).use { it.fields }
        val tokenizer = TokenizerFactory.fromGgufFields(fields)
        assertEquals(
            fixture.promptTokens, tokenizer.encode(fixture.prompt).toList(),
            "prompt tokenization must match mainline llama.cpp",
        )

        // 2 — greedy continuation, DSL path, keep-packed engine loading. No BOS prepend:
        // llama.cpp adds none for qwen2/qwen3 and the fixture tokens are the raw encoding.
        val weights = runBlocking {
            QwenWeightLoader.loadToMapStreaming<FP32, Float>(
                ctx, { JvmRandomAccessSource.open(modelPath) },
            )
        }
        val runtime = OptimizedLLMRuntime(
            QwenNetworkLoader.fromWeights(weights, kvCacheKind = ParityEnv.kvCacheKind), ctx,
            OptimizedLLMMode.DIRECT, FP32::class,
        )
        for (i in 0 until fixture.promptTokens.size - 1) runtime.forward(fixture.promptTokens[i])
        var token = fixture.promptTokens.last()
        val text = StringBuilder()
        repeat(fixture.steps) {
            token = sampleFromTensor(runtime.forward(token), 0f)
            text.append(tokenizer.decode(token))
        }
        assertEquals(
            fixture.oracleText, text.toString(),
            "greedy decode must equal mainline llama.cpp token-for-token",
        )
    }

    @Test
    fun qwen25GreedyDecodeMatchesMainlineLlamaCpp() {
        val modelPath = System.getenv("QWEN25_05B_GGUF")
        if (modelPath.isNullOrBlank()) {
            println("PARITY skipped: QWEN25_05B_GGUF not set")
            return
        }
        assertParity(modelPath, loadFixture("/qwen25-05b/golden-greedy-05b.txt"))
    }

    @Test
    fun qwen3GreedyDecodeMatchesMainlineLlamaCpp() {
        val modelPath = System.getenv("QWEN3_17B_GGUF")
        if (modelPath.isNullOrBlank()) {
            println("PARITY skipped: QWEN3_17B_GGUF not set")
            return
        }
        assertParity(modelPath, loadFixture("/qwen3-17b/golden-greedy-17b.txt"))
    }
}
