package sk.ainet.models.gemma

import java.io.File
import java.lang.foreign.Arena
import kotlinx.coroutines.runBlocking
import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import org.junit.jupiter.api.Assumptions
import org.junit.jupiter.api.Tag
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.tokenizer.GGUFTokenizer
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * End-to-end check that the NEW Q5_K packed in-kernel dequant path (upstream
 * SKaiNET `Q5_KBlockTensorData` + `Q5KMatmulKernel`, wired here via
 * [convertGemmaWeightsToMemSeg]) decodes FunctionGemma-270M (`Q5_K_M`)
 * identically to the FP32-dequant baseline, and reports tokens/sec.
 *
 * Before this, the converter dequantized Q5_K weights to FP32 on load ("no
 * native matmul kernel yet for Q5_K"). Now Q5_K stays packed (176 B/block)
 * and runs the in-kernel dequant matmul. Both paths decode the same weights,
 * so greedy argmax token sequences must match.
 *
 * Skips when the GGUF isn't present (CI without the checkpoint).
 */
@Tag("integration")
class GemmaQ5KPackedParityTest {

    private val gguf =
        FunctionGemmaFixture.gguf

    private fun argmax(a: FloatArray): Int {
        var bi = 0; var bv = a[0]
        for (i in 1 until a.size) if (a[i] > bv) { bv = a[i]; bi = i }
        return bi
    }

    private fun buildPrompt(u: String) =
        "<start_of_turn>user\n$u<end_of_turn>\n<start_of_turn>model\n"

    private fun decode(
        runtime: OptimizedLLMRuntime<FP32>,
        promptTokens: List<Int>,
        maxNew: Int,
        eos: Int,
        eot: Int,
    ): List<Int> {
        runtime.reset()
        var logits = FloatArray(0)
        for (t in promptTokens) logits = runtime.forward(t).data.copyToFloatArray()
        val gen = mutableListOf<Int>()
        while (gen.size < maxNew) {
            val next = argmax(logits)
            gen.add(next)
            if (next == eos || next == eot) break
            logits = runtime.forward(next).data.copyToFloatArray()
        }
        return gen
    }

    @Test
    fun q5kPackedMatchesFp32() = runBlocking {
        Assumptions.assumeTrue(File(gguf).exists(), "FunctionGemma GGUF not present — skipping")

        val ctx = DirectCpuExecutionContext.create()
        val tokenizer = GGUFTokenizer.fromSource(SystemFileSystem.source(Path(gguf)).buffered())
        val eot = tokenizer.encode("<end_of_turn>").single()
        val eos = tokenizer.eosTokenId
        val promptTokens =
            listOf(tokenizer.bosTokenId) + tokenizer.encode(buildPrompt("Turn the light on.")).toList()
        val maxNew = 12

        // --- FP32 dequant-on-load baseline ---
        val wFp32 = Gemma4WeightLoader(
            randomAccessProvider = { JvmRandomAccessSource.open(gguf) },
            quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
        ).loadToMapStreaming<FP32, Float>(ctx, FP32::class)
        val mFp32 = GemmaNetworkLoader.fromWeights(ctx, wFp32, FP32::class)
        val rtFp32 = OptimizedLLMRuntime(
            model = mFp32, ctx = ctx, mode = OptimizedLLMMode.DIRECT,
            dtype = FP32::class, bos = tokenizer.bosTokenId,
        )
        val genFp32 = decode(rtFp32, promptTokens, maxNew, eos, eot)

        // --- Q5_K packed in-kernel dequant path (NATIVE_OPTIMIZED + convert) ---
        Arena.ofConfined().use { arena ->
            val wNat = Gemma4WeightLoader(
                randomAccessProvider = { JvmRandomAccessSource.open(gguf) },
                quantPolicy = QuantPolicy.NATIVE_OPTIMIZED,
            ).loadToMapStreaming<FP32, Float>(ctx, FP32::class)
            val wConv = convertGemmaWeightsToMemSeg(wNat, ctx, arena)
            @Suppress("UNCHECKED_CAST")
            val mNat = GemmaNetworkLoader.fromWeights(
                ctx, wConv as Gemma4Weights<FP32, Float>, FP32::class,
            )
            val rtNat = OptimizedLLMRuntime(
                model = mNat, ctx = ctx, mode = OptimizedLLMMode.DIRECT,
                dtype = FP32::class, bos = tokenizer.bosTokenId,
            )

            // Warmup one decode (JIT + kernel-provider resolution), then time.
            decode(rtNat, promptTokens, 2, eos, eot)
            val t0 = System.nanoTime()
            val genNat = decode(rtNat, promptTokens, maxNew, eos, eot)
            val ms = (System.nanoTime() - t0) / 1e6
            val toks = genNat.size + promptTokens.size

            println("Q5K-packed gen=$genNat")
            println("FP32-base  gen=$genFp32")
            println("Q5K decoded='${tokenizer.decode(genNat.toIntArray()).replace("\n", "\\n")}'")
            println(
                "Q5K-packed throughput: $toks tok in ${"%.0f".format(ms)} ms " +
                    "(${"%.2f".format(toks * 1000.0 / ms)} tok/s incl. prefill)",
            )

            assertEquals(genFp32, genNat, "Q5_K packed decode diverged from FP32 baseline")
        }

        // The wired path: GemmaNetworkLoader.load(NATIVE_OPTIMIZED) applies the
        // commonMain convertGemmaWeightsPacked (the board path) — no MemSeg, no
        // Arena. Must decode identically to the FP32 baseline too.
        val mLoad = GemmaNetworkLoader.fromGguf(
            randomAccessProvider = { JvmRandomAccessSource.open(gguf) },
            quantPolicy = QuantPolicy.NATIVE_OPTIMIZED,
        ).load<FP32, Float>(ctx)
        val rtLoad = OptimizedLLMRuntime(
            model = mLoad, ctx = ctx, mode = OptimizedLLMMode.DIRECT,
            dtype = FP32::class, bos = tokenizer.bosTokenId,
        )
        val genLoad = decode(rtLoad, promptTokens, maxNew, eos, eot)
        println("load(NATIVE_OPTIMIZED) gen=$genLoad")
        assertEquals(genFp32, genLoad, "load(NATIVE_OPTIMIZED) packed decode diverged from FP32 baseline")
    }
}
