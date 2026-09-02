package sk.ainet.models.gemma3n

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlinx.coroutines.runBlocking
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.sampleFromTensor
import sk.ainet.apps.llm.tokenizer.TokenizerFactory
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.lang.types.FP32

/**
 * The #346 maturity-gate parity probe for the Gemma 3n family (#377 DSL migration), against
 * **mainline llama.cpp**: full 32-step greedy text equality on the DSL path
 * ([Gemma3nWeightLoader] engine loading, packed/MAPPED → [Gemma3nNetworkLoader.fromWeights]
 * → [OptimizedLLMRuntime]). This is the first time the family has a reference gate at all —
 * and it exercises everything gemma3n adds over gemma-4: AltUp's four parallel streams with
 * the tanh router, Laurel, Gaussian-top-k activation sparsity on the first ten layers, PLE
 * feeding the non-active streams, and per-type shared KV for the last ten layers.
 *
 * Model-gated: runs only when `GEMMA3N_E2B_GGUF` points at `gemma-3n-E2B-it-Q4_K_M.gguf`
 * AND the test JVM has ≥ 16 GB heap (`-PgemmaTestMaxHeap=20g`); skips quietly otherwise.
 * The fixture header records the exact oracle build and commands.
 */
@org.junit.jupiter.api.Tag("smoke-reference")
@org.junit.jupiter.api.Tag("integration")
class Gemma3nGoldenTokenParityTest {

    private data class Fixture(
        val prompt: String,
        val steps: Int,
        val promptTokens: List<Int>,
        val oracleText: String,
    )

    private fun loadFixture(): Fixture {
        val raw = checkNotNull(javaClass.getResourceAsStream("/gemma3n-e2b/golden-greedy-e2b.txt")) {
            "fixture /gemma3n-e2b/golden-greedy-e2b.txt missing from test resources"
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
        val modelPath = System.getenv("GEMMA3N_E2B_GGUF")
        if (modelPath.isNullOrBlank()) {
            println("PARITY skipped: GEMMA3N_E2B_GGUF not set")
            return
        }
        val maxHeapGb = Runtime.getRuntime().maxMemory() / (1024L * 1024L * 1024L)
        if (maxHeapGb < 16) {
            println("PARITY skipped: heap=$maxHeapGb GB < 16 GB; rerun with -PgemmaTestMaxHeap=20g")
            return
        }
        val fixture = loadFixture()
        val ctx = DirectCpuExecutionContext()

        // 1 — prompt tokenization parity (a failure here names the tokenizer, not the model).
        val fields = StreamingGGUFReader.open(JvmRandomAccessSource.open(modelPath)).use { it.fields }
        val tokenizer = TokenizerFactory.fromGgufFields(fields)
        // gemma adds a BOS token (id 2); mirror llama.cpp's add_special encoding.
        val raw = tokenizer.encode(fixture.prompt)
        val encoded = if (raw.isNotEmpty() && raw[0] == tokenizer.bosTokenId) raw
        else intArrayOf(tokenizer.bosTokenId) + raw
        assertEquals(
            fixture.promptTokens, encoded.toList(),
            "prompt tokenization must match mainline llama.cpp",
        )

        // 2 — greedy continuation on the DSL path.
        val weights = runBlocking {
            Gemma3nWeightLoader(
                randomAccessProvider = { JvmRandomAccessSource.open(modelPath) },
            ).loadToMapStreaming<FP32, Float>(ctx)
        }
        val model = Gemma3nNetworkLoader.fromWeights(ctx, weights)
        val runtime = OptimizedLLMRuntime(model, ctx, OptimizedLLMMode.DIRECT, FP32::class)
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
}
