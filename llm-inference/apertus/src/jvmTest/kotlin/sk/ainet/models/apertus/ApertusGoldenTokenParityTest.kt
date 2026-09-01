package sk.ainet.models.apertus

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
 * The #346 maturity-gate parity probe for the Apertus family, against **mainline llama.cpp** —
 * the gate retrofit that closes the "no parity probe yet" row in the #346 table. Like the
 * Llama/Qwen gates it asserts FULL cross-implementation greedy text equality on the DSL path
 * the CLI ships ([ApertusWeightLoader] engine loading → [ApertusNetworkLoader.fromWeights] →
 * [OptimizedLLMRuntime]) — exercising the family's distinguishing machinery end-to-end:
 * QK-norm, xIELU per-layer activation parameters, and the ungated FFN.
 *
 * Model-gated: runs only when `APERTUS_GGUF_PATH` points at
 * `Apertus-8B-Instruct-2509-Q4_K_S.gguf` (the same artifact `ApertusRealGgufLoadingTest`
 * uses) AND the test JVM has ≥ 8 GB heap (`-PapertusTestMaxHeap=12g`); skips quietly
 * otherwise. The fixture header records the exact oracle build and commands.
 */
@org.junit.jupiter.api.Tag("smoke-reference")
@org.junit.jupiter.api.Tag("integration")
class ApertusGoldenTokenParityTest {

    private data class Fixture(
        val prompt: String,
        val steps: Int,
        val promptTokens: List<Int>,
        val oracleText: String,
    )

    private fun loadFixture(): Fixture {
        val raw = checkNotNull(javaClass.getResourceAsStream("/apertus-8b/golden-greedy-8b.txt")) {
            "fixture /apertus-8b/golden-greedy-8b.txt missing from test resources"
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
        val modelPath = System.getenv("APERTUS_GGUF_PATH")
        if (modelPath.isNullOrBlank()) {
            println("PARITY skipped: APERTUS_GGUF_PATH not set")
            return
        }
        val maxHeapGb = Runtime.getRuntime().maxMemory() / (1024L * 1024L * 1024L)
        if (maxHeapGb < 8) {
            println("PARITY skipped: heap=$maxHeapGb GB < 8 GB; rerun with -PapertusTestMaxHeap=12g")
            return
        }
        val fixture = loadFixture()
        val ctx = DirectCpuExecutionContext()

        // 1 — prompt tokenization parity (a failure here names the tokenizer, not the model).
        val fields = StreamingGGUFReader.open(JvmRandomAccessSource.open(modelPath)).use { it.fields }
        val tokenizer = TokenizerFactory.fromGgufFields(fields)
        // Apertus adds a BOS token (id 1); mirror llama.cpp's add_special encoding.
        val raw = tokenizer.encode(fixture.prompt)
        val encoded = if (raw.isNotEmpty() && raw[0] == tokenizer.bosTokenId) raw
        else intArrayOf(tokenizer.bosTokenId) + raw
        assertEquals(
            fixture.promptTokens, encoded.toList(),
            "prompt tokenization must match mainline llama.cpp",
        )

        // 2 — greedy continuation on the DSL path, keep-packed engine loading. The fixture's
        // prompt_tokens are the oracle's exact encoding (add_special included), fed verbatim.
        val weights = runBlocking {
            ApertusWeightLoader.fromRandomAccess(
                randomAccessProvider = { JvmRandomAccessSource.open(modelPath) },
            ).loadToMap<FP32, Float>(ctx)
        }
        val model = ApertusNetworkLoader.fromWeights(ctx, weights)
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
