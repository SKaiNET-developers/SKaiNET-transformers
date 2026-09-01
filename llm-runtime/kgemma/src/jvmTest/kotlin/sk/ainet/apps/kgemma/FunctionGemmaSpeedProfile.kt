package sk.ainet.apps.kgemma

import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.putJsonObject
import kotlinx.serialization.json.add
import sk.ainet.apps.kllama.chat.ToolDefinition
import kotlin.test.Test
import kotlin.time.measureTime

/**
 * End-to-end timing for official FunctionGemma 270M function calling: how long from "user asks a
 * question with one tool declared" to "a parsed ToolCall", broken into load, prompt ingestion and
 * generation.
 *
 * This is the whole product path — official chat template, GGUF tokenizer, eager CPU runtime,
 * tool-call parser — not a kernel microbenchmark.
 */
class FunctionGemmaSpeedProfile {

    @Test
    fun end_to_end_tool_call_timing() {
        val gguf = System.getenv("FUNCTIONGEMMA_GGUF")
        if (gguf.isNullOrBlank() || System.getenv("GEMMA4_SPEED") != "1") {
            println("[skip] set FUNCTIONGEMMA_GGUF and GEMMA4_SPEED=1"); return
        }
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

        // What form do the weights actually land in with the BF16 policy?
        val probeCtx = sk.ainet.context.DirectCpuExecutionContext.create()
        val probe = kotlinx.coroutines.runBlocking {
            sk.ainet.models.gemma.GemmaWeightLoader(
                randomAccessProvider = { sk.ainet.io.JvmRandomAccessSource.open(gguf) },
                dtypePolicy = sk.ainet.lang.types.DTypePolicy.Prefer(sk.ainet.lang.types.BF16),
            ).loadToMapStreaming<sk.ainet.lang.types.FP32, Float>(probeCtx, sk.ainet.lang.types.FP32::class)
        }
        for (nm in listOf("blk.0.attn_q.weight", "blk.0.ffn_gate.weight", "token_embd.weight")) {
            val t = probe.tensors[nm]
            println("FGSPEED form %-24s %s".format(nm, t?.let { "${it.data::class.simpleName}" } ?: "<absent>"))
        }

        lateinit var fg: FunctionGemma
        val load = measureTime { fg = FunctionGemma.fromGguf(gguf, style = FunctionGemma.Style.OFFICIAL) }

        // Warm the JIT on the same path before measuring.
        fg.callWithTools("What's the weather in Berlin?", listOf(weather), maxTokens = 16, temperature = 0f)

        var turn: FunctionGemma.Turn? = null
        val answer = measureTime {
            turn = fg.callWithTools(
                "What's the weather in Paris?", listOf(weather), maxTokens = 64, temperature = 0f,
            )
        }
        val t = turn!!
        println("FGSPEED load           : ${load.inWholeMilliseconds} ms")
        println("FGSPEED full tool call : ${answer.inWholeMilliseconds} ms  (85-token prompt + generation)")
        println("FGSPEED parsed calls   : ${t.calls}")
        println("FGSPEED raw text       : '${t.text.replace("\n", "\\n").take(120)}'")

        // Decode rate on its own, from a longer continuation.
        val steps = 24
        var count = 0
        val decode = measureTime {
            val t2 = fg.callWithTools(
                "What's the weather in Paris?", listOf(weather), maxTokens = steps, temperature = 0f,
            )
            count = t2.text.length
        }
        println("FGSPEED $steps-token cap    : ${decode.inWholeMilliseconds} ms (produced $count chars)")
    }
}
