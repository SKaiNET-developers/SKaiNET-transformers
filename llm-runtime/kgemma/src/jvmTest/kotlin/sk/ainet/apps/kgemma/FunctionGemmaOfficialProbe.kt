package sk.ainet.apps.kgemma

import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.putJsonObject
import kotlinx.serialization.json.add
import sk.ainet.apps.kllama.chat.ToolDefinition
import kotlin.random.Random
import kotlin.test.Test

/**
 * Diagnostic probe (env-gated, FUNCTIONGEMMA_PROBE=1 additionally required):
 * runs two official-format tasks under greedy AND the model card's recommended
 * sampling (temperature 1.0, topK 64, topP 0.95, seeded) and prints outcomes.
 * Not a gate — used to choose the e2e test's decoding config empirically.
 */
class FunctionGemmaOfficialProbe {

    /**
     * Forward-pass parity probe: prefill llama.cpp's EXACT reference token ids
     * (so tokenization is out of the picture — `FunctionGemmaOfficialGgufTest`
     * already pins that our encode matches byte-for-byte) and print our top-5
     * logits for the first predicted token.
     *
     * llama.cpp b10621 reference on this BF16 GGUF, greedy:
     *   step 0 → id 48 `<start_function_call>` at logprob ≈ -1e-5 (≈100%),
     *   runner-up id 49 at -11.9. Then 6639 `call`, 236787 `:`.
     */
    @Test
    fun probe_first_token_logits_vs_llamacpp() {
        val gguf = System.getenv("FUNCTIONGEMMA_GGUF")
        if (gguf.isNullOrBlank() || System.getenv("FUNCTIONGEMMA_PROBE") != "1") {
            println("SKIP: set FUNCTIONGEMMA_GGUF and FUNCTIONGEMMA_PROBE=1"); return
        }
        val referenceIds = intArrayOf(
            2, 105, 55060, 107, 3048, 659, 496, 2028, 600, 740, 776, 1292, 11687, 607, 506, 2269,
            5151, 46, 163688, 236787, 828, 236779, 19323, 236782, 7777, 236787, 52, 3407, 1873,
            7606, 573, 496, 4563, 52, 236764, 19031, 29616, 15921, 29616, 7125, 29616, 7777,
            236787, 52, 17698, 1463, 52, 236764, 2084, 236787, 52, 35410, 52, 5237, 15979, 24845,
            52, 7125, 52, 1604, 2084, 236787, 52, 60688, 52, 1807, 47, 106, 107, 105, 2364, 107,
            3689, 236789, 236751, 506, 7606, 528, 9079, 236881, 106, 107, 105, 4368, 107,
        )
        val ctx = sk.ainet.context.DirectCpuExecutionContext.create()
        val weights = kotlinx.coroutines.runBlocking {
            sk.ainet.models.gemma.Gemma4WeightLoader(
                randomAccessProvider = { sk.ainet.io.JvmRandomAccessSource.open(gguf) },
                weightForm = sk.ainet.models.gemma.GEMMA_DEQUANTIZE_ALL,
            ).loadToMapStreaming<sk.ainet.lang.types.FP32, Float>(ctx, sk.ainet.lang.types.FP32::class)
        }
        val md = weights.metadata
        println(
            "META arch=${md.architecture} layers=${md.blockCount} dim=${md.embeddingLength} " +
                "heads=${md.headCount}/${md.kvHeadCount} headDim=${md.headDim}/${md.globalHeadDim} " +
                "ffn=${md.intermediateSize} vocab=${md.vocabSize} sw=${md.slidingWindow} " +
                "kvShared=${md.kvSharedLayers} eps=${md.rmsNormEps} bos=${md.bosTokenId} eos=${md.eosTokenId} " +
                "softcap=${md.finalLogitSoftcapping} ple=${md.perLayerEmbeddingLength}"
        )
        println("META ropeFull=${md.ropeParametersFull} ropeSliding=${md.ropeParametersSliding}")
        println("META layerTypes=${md.layerTypes}")
        val patched = weights.copy(
            metadata = md.copy(
                ropeParametersFull = md.ropeParametersFull.copy(partialRotaryFactor = 1.0f),
            ),
        )
        val model = sk.ainet.models.gemma.GemmaNetworkLoader.fromWeights(
            ctx, patched, sk.ainet.lang.types.FP32::class
        )
        val runtime = sk.ainet.apps.llm.OptimizedLLMRuntime(
            model, ctx, sk.ainet.apps.llm.OptimizedLLMMode.DIRECT,
            sk.ainet.lang.types.FP32::class, random = Random.Default,
        )
        var logits = runtime.forward(referenceIds[0])
        for (i in 1 until referenceIds.size) logits = runtime.forward(referenceIds[i])
        val buf = logits.data.copyToFloatArray()
        val top5 = buf.toList().mapIndexed { i, v -> i to v }.sortedByDescending { it.second }.take(5)
        println("OURS top-5 after ${referenceIds.size} reference prompt tokens:")
        val tok = sk.ainet.apps.kllama.GGUFTokenizer.fromRandomAccessSource(
            sk.ainet.io.JvmRandomAccessSource.open(gguf)
        )
        for ((id, score) in top5) {
            println("  id=%6d score=%+.4f piece=%s".format(id, score, runCatching { tok.decode(id) }.getOrElse { "<err>" }))
        }
        println("REFERENCE step0: id=48 '<start_function_call>' (~100%), runner-up id=49 at -11.9 logprob")
        println("OURS logit[48]=${buf.getOrNull(48)} logit[49]=${buf.getOrNull(49)}")
    }

    @Test
    fun probe_prompts_and_sampling() {
        val gguf = System.getenv("FUNCTIONGEMMA_GGUF")
        if (gguf.isNullOrBlank() || System.getenv("FUNCTIONGEMMA_PROBE") != "1") {
            println("SKIP: set FUNCTIONGEMMA_GGUF and FUNCTIONGEMMA_PROBE=1"); return
        }
        val fg = FunctionGemma.fromGguf(gguf, style = FunctionGemma.Style.OFFICIAL)

        val weather = ToolDefinition(
            name = "get_weather",
            description = "Get current weather for a location",
            parameters = buildJsonObject {
                put("type", "object")
                putJsonObject("properties") {
                    putJsonObject("location") { put("type", "string"); put("description", "City name") }
                }
                putJsonArray("required") { add("location") }
            },
        )
        val date = ToolDefinition(
            name = "get_today_date",
            description = "Gets today's date",
            parameters = buildJsonObject { put("type", "object") },
        )

        data class Case(val label: String, val q: String, val tool: ToolDefinition)
        val cases = listOf(
            Case("weather", "What's the weather in Paris?", weather),
            Case("date", "What is today's date?", date),
        )
        for (c in cases) {
            val greedy = fg.callWithTools(c.q, listOf(c.tool), maxTokens = 48, temperature = 0f)
            println("PROBE ${c.label} greedy: calls=${greedy.calls} text='${greedy.text.take(160)}'")
            val sampled = fg.callWithTools(
                c.q, listOf(c.tool), maxTokens = 48,
                temperature = 1.0f, topK = 64, topP = 0.95f, random = Random(42),
            )
            println("PROBE ${c.label} sampled(seed42): calls=${sampled.calls} text='${sampled.text.take(160)}'")
        }
    }
}
