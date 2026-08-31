package sk.ainet.models.bitnet

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlinx.coroutines.runBlocking
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.tokenizer.TokenizerFactory
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.exec.kernel.NativeTernaryF32GemvKernel
import sk.ainet.exec.kernel.NativeTernaryLmheadKernel
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.lang.types.FP32

/**
 * The #346 maturity-gate parity probe (transformers#360), against the **bitnet.cpp oracle** —
 * microsoft/BitNet's llama.cpp fork, the reference implementation for the I2_S-quantized
 * `BitNet-b1.58-2B-4T` GGUF (mainline llama.cpp cannot read I2_S, type 36).
 *
 * Model-gated: runs only when `BITNET_2B4T_GGUF` points at the real GGUF; skips quietly
 * otherwise. The fixture (`golden-greedy-2b4t.txt`) records the oracle command, its outputs,
 * and SKaiNET's own greedy continuation.
 *
 * What is asserted, and what deliberately is not:
 * 1. **Prompt tokenization parity** (asserted, vs oracle): SKaiNET's GGUF tokenizer must
 *    produce the oracle's prompt ids exactly.
 * 2. **Self-golden regression** (asserted): both decode paths — the full planes matmul and the
 *    two-stage candidate loop — must keep reproducing the recorded SKaiNET greedy text. Any
 *    numerics regression that flips an argmax anywhere in 32 tokens fails here.
 * 3. **Cross-implementation text equality is NOT asserted** — the oracle's CPU i2_s matmul
 *    quantizes activations to int8 while SKaiNET's LUT path is exact FP32×ternary, and the
 *    greedy texts genuinely diverge (details in the fixture header). The comparison is printed
 *    for the record; BF16 arbitration is the open follow-up on #360.
 */
class BitNetGoldenTokenParityTest {

    private data class Fixture(
        val prompt: String,
        val steps: Int,
        val promptTokens: List<Int>,
        val oracleText: String,
        val skainetText: String,
    )

    private fun loadFixture(): Fixture {
        val raw = checkNotNull(javaClass.getResourceAsStream("/bitnet2b4t/golden-greedy-2b4t.txt")) {
            "fixture /bitnet2b4t/golden-greedy-2b4t.txt missing from test resources"
        }.bufferedReader().readLines()
        val map = raw.filter { it.isNotBlank() && !it.startsWith("#") }
            .associate { line -> line.substringBefore('=') to line.substringAfter('=') }
        return Fixture(
            prompt = map.getValue("prompt"),
            steps = map.getValue("steps").toInt(),
            promptTokens = map.getValue("prompt_tokens").split(',').map { it.trim().toInt() },
            oracleText = map.getValue("oracle_text"),
            skainetText = map.getValue("skainet_text"),
        )
    }

    @Test
    fun greedyDecodeHoldsTheRecordedParityContract() {
        val modelPath = System.getenv("BITNET_2B4T_GGUF")
        if (modelPath.isNullOrBlank()) {
            println("PARITY skipped: BITNET_2B4T_GGUF not set")
            return
        }
        val fixture = loadFixture()
        val ctx = DirectCpuExecutionContext()
        NativeTernaryF32GemvKernel.install()
        NativeTernaryLmheadKernel.install()

        // 1 — prompt tokenization parity with the bitnet.cpp oracle.
        val fields = StreamingGGUFReader.open(JvmRandomAccessSource.open(modelPath)).use { it.fields }
        val tokenizer = TokenizerFactory.fromGgufFields(fields)
        val promptTokens = intArrayOf(tokenizer.bosTokenId) + tokenizer.encode(fixture.prompt)
        assertEquals(
            fixture.promptTokens, promptTokens.toList(),
            "prompt tokenization must match the bitnet.cpp oracle",
        )

        // 2a — self-golden regression, full planes-matmul loop.
        val loaded = runBlocking {
            BitNetWeightLoader.loadWithMetadata(ctx, { JvmRandomAccessSource.open(modelPath) })
        }
        val runtime = OptimizedLLMRuntime(
            loaded.model, ctx, OptimizedLLMMode.DIRECT, FP32::class, bos = loaded.metadata.bosTokenId,
        )
        val fullText = StringBuilder()
        runtime.generate(promptTokens, fixture.steps, temperature = 0f) { fullText.append(tokenizer.decode(it)) }
        assertEquals(fixture.skainetText, fullText.toString(), "full-matmul greedy self-golden")

        // 2b — self-golden regression, two-stage candidate loop (fresh model for a fresh KV cache).
        val head = bitnetPlanesHead(loaded.model)
        assertTrue(head != null, "2B4T must load with a planes head")
        val runtime2 = OptimizedLLMRuntime(
            runBlocking { BitNetWeightLoader.load(ctx, { JvmRandomAccessSource.open(modelPath) }) },
            ctx, OptimizedLLMMode.DIRECT, FP32::class, bos = loaded.metadata.bosTokenId,
        )
        val twoStageText = StringBuilder()
        runtime2.generateTwoStage(
            promptTokens, fixture.steps, temperature = 0f, head = head,
            native = BitNetStage1Kernel(NativeTernaryLmheadKernel::lmheadStage1),
        ) { twoStageText.append(tokenizer.decode(it)) }
        assertEquals(fixture.skainetText, twoStageText.toString(), "two-stage greedy self-golden")

        // 3 — informational cross-implementation comparison (not asserted; see fixture header).
        val shared = fixture.oracleText.commonPrefixWith(fullText.toString())
        println("PARITY oracle-vs-skainet shared prefix: ${shared.length} chars: '${shared.take(60)}'")
    }
}
