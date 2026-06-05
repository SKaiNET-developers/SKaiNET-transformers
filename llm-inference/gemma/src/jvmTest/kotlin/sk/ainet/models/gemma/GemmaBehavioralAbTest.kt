package sk.ainet.models.gemma

import org.junit.jupiter.api.Tag
import kotlinx.coroutines.runBlocking
import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonPrimitive
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.tokenizer.GGUFTokenizer
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.types.FP32
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * Behavioral A/B vs the Python FunctionGemma app: same Octopus-v2 prompt, greedy
 * decode through the REAL SKaiNET DSL runtime (OptimizedLLMRuntime, eager), and
 * assert the generated token sequence + decoded `<tool_N>(args)<end>` match
 * llama.cpp (reference captured by build-mlir/ab_generate_llama.py -> gen_*.json).
 * Generation is driven from llama's prompt tokens to isolate it from
 * tokenization; tokenizer parity is asserted separately.
 */
@Tag("integration")
class GemmaBehavioralAbTest {
    private val gguf = "/home/miso/projects/coral/sl2610-voice-cc-kt/models/functiongemma-physical-ai-v10-Q5_K_M.gguf"

    private fun argmax(a: FloatArray): Int {
        var bi = 0; var bv = a[0]
        for (i in 1 until a.size) if (a[i] > bv) { bv = a[i]; bi = i }
        return bi
    }

    private fun buildPrompt(u: String) = "<start_of_turn>user\n$u<end_of_turn>\n<start_of_turn>model\n"

    @Test
    fun behavioralParity() = runBlocking {
        val ctx = DirectCpuExecutionContext.create()
        val tokenizer = GGUFTokenizer.fromSource(SystemFileSystem.source(Path(gguf)).buffered())
        val weights = Gemma4WeightLoader(
            randomAccessProvider = { JvmRandomAccessSource.open(gguf) },
            quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
        ).loadToMapStreaming<FP32, Float>(ctx, FP32::class)
        val model = GemmaNetworkLoader.fromWeights(ctx, weights, FP32::class)
        val runtime = OptimizedLLMRuntime(
            model = model, ctx = ctx, mode = OptimizedLLMMode.DIRECT,
            dtype = FP32::class, bos = tokenizer.bosTokenId,
        )
        val eot = tokenizer.encode("<end_of_turn>").single()
        val eos = tokenizer.eosTokenId
        println("EOT=$eot EOS=$eos BOS=${tokenizer.bosTokenId}")

        val outDir = File("/home/miso/projects/coral/build-mlir/out")
        var allMatch = true
        var i = 0
        while (true) {
            val f = File(outDir, "gen_$i.json"); if (!f.exists()) break
            val rec = Json.parseToJsonElement(f.readText()) as JsonObject
            val prompt = rec["prompt"]!!.jsonPrimitive.content
            val promptTokens = rec["prompt_tokens"]!!.jsonArray.map { it.jsonPrimitive.int }
            val refGen = rec["gen_tokens"]!!.jsonArray.map { it.jsonPrimitive.int }
            val refRaw = rec["raw"]!!.jsonPrimitive.content

            // (1) tokenizer parity: our encode (+bos) vs llama's prompt_tokens.
            val enc = tokenizer.encode(buildPrompt(prompt))
            val encWithBos = listOf(tokenizer.bosTokenId) + enc.toList()
            val tokOk = encWithBos == promptTokens
            println("[$i] '$prompt' tokenizer ${if (tokOk) "MATCH" else "DIFF enc=$encWithBos vs ref=$promptTokens"}")

            // (2) generation: feed llama's prompt tokens, greedy decode.
            runtime.reset()
            var logits = FloatArray(0)
            for (t in promptTokens) logits = runtime.forward(t).data.copyToFloatArray()
            val gen = mutableListOf<Int>()
            while (gen.size < 48) {
                val next = argmax(logits)
                if (next == eos) break
                gen.add(next)
                if (next == eot) break
                logits = runtime.forward(next).data.copyToFloatArray()
            }
            val raw = tokenizer.decode(gen.toIntArray())
            val genOk = gen == refGen
            println("    gen ${if (genOk) "MATCH" else "DIFF"} ours=$gen")
            println("    ref =$refGen")
            println("    raw ours=${raw.replace("\n", "\\n")}  ref=${refRaw.replace("\n", "\\n")}")
            allMatch = allMatch && tokOk && genOk
            i++
        }
        assertEquals(true, allMatch, "behavioral parity failed (see DIFF above)")
    }
}
