package sk.ainet.apps.kgemma.cli

import java.io.File
import java.lang.foreign.Arena
import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.io.path.isDirectory
import kotlin.io.path.readText
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.fail
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonObject
import sk.ainet.apps.kgemma.Gemma4Ingestion
import sk.ainet.apps.kgemma.Gemma4LoadConfig
import sk.ainet.apps.kgemma.loadDslRuntimeNativeStreaming
import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.apps.kllama.chat.AgentListener
import sk.ainet.apps.kllama.chat.ChatSession
import sk.ainet.apps.kllama.chat.ModelMetadata
import sk.ainet.apps.kllama.chat.Tool
import sk.ainet.apps.kllama.chat.ToolCall
import sk.ainet.apps.kllama.chat.ToolDefinition
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.tensor.data.MemorySegmentTensorDataFactory
import sk.ainet.lang.types.FP32

/**
 * End-to-end smoke test for Gemma 4 tool calling against a **real** E2B
 * checkpoint. Gated on the `GEMMA4_E2B_MODEL_PATH` env var so CI stays green
 * without the ~3 GB weights on disk.
 *
 * Purpose: this is the first test that empirically pins down whether the
 * trained Gemma 4 model natively emits `<|tool_call>...<tool_call|>` against
 * our [sk.ainet.apps.kllama.chat.Gemma4ChatTemplate]. Every other test in
 * this repo uses synthetic weights or a mock runtime with a scripted token
 * stream, so the grammar choices in the template (including the paired
 * `<|think>` / `<think|>` convention introduced for thinking mode) rest on
 * spec reading, not observed behavior. A pass here closes that gap.
 *
 * To run locally:
 * ```
 * export GEMMA4_E2B_MODEL_PATH=/path/to/gemma-4-e2b.gguf
 * ./gradlew :llm-runtime:kgemma:jvmTest --tests '*Gemma4E2BToolCallSmokeTest*'
 * ```
 *
 * If the test finds the model but the model does NOT produce a parseable
 * tool call, it fails with the raw response captured so the grammar
 * mismatch can be diagnosed and written up in
 * `gemma4-research/findings/tool_calling.md` (tracked as Group 5 in
 * `PLAN-tool-calling.md`).
 */
class Gemma4E2BToolCallSmokeTest {

    private class CalculatorTool : Tool {
        var invoked = false
            private set
        var receivedExpression: String? = null
            private set
        override val definition: ToolDefinition = ToolDefinition(
            name = "calculator",
            description = "Evaluate a mathematical expression. Supports +, -, *, / and parentheses.",
            parameters = buildJsonObject {
                put("type", "object")
                putJsonObject("properties") {
                    putJsonObject("expression") {
                        put("type", "string")
                        put("description", "The mathematical expression to evaluate, e.g. '17 * 23'")
                    }
                }
            }
        )
        override fun execute(arguments: JsonObject): String {
            invoked = true
            receivedExpression = arguments["expression"]?.jsonPrimitive?.content
            // Compute a plausible answer so the model can carry on if it wants to.
            val expr = receivedExpression ?: return "error: no expression"
            return try {
                // This test only asks for a single multiplication — trivial parse.
                val parts = expr.split("*", " ").map { it.trim() }.filter { it.isNotEmpty() }
                if (parts.size == 2) {
                    val a = parts[0].toLong()
                    val b = parts[1].toLong()
                    (a * b).toString()
                } else {
                    "error: unexpected expression shape"
                }
            } catch (e: Exception) {
                "error: ${e.message}"
            }
        }
    }

    @Test
    fun `real Gemma 4 E2B emits parseable tool_call against Gemma4ChatTemplate`() {
        val modelPath = System.getenv("GEMMA4_E2B_MODEL_PATH")?.trim().orEmpty()
        if (modelPath.isEmpty()) {
            println("[skip] GEMMA4_E2B_MODEL_PATH not set — skipping real-checkpoint smoke test.")
            return
        }
        val path = Path.of(modelPath)
        if (!path.exists()) {
            println("[skip] GEMMA4_E2B_MODEL_PATH=$modelPath does not exist — skipping.")
            return
        }
        // GGUF-only for this smoke test; SafeTensors models need a different
        // loader wiring and aren't required to verify the template grammar.
        val isGguf = !path.isDirectory() && path.toString().endsWith(".gguf")
        if (!isGguf) {
            println("[skip] GEMMA4_E2B_MODEL_PATH=$modelPath is not a .gguf file — skipping (this test only exercises the GGUF DSL path).")
            return
        }

        runBlocking {
            val memSegFactory = MemorySegmentTensorDataFactory()
            val ctx = DirectCpuExecutionContext(tensorDataFactory = memSegFactory)
            val quantArena = Arena.ofShared()
            try {
                val ingestion = Gemma4Ingestion<FP32>(
                    ctx = ctx,
                    dtype = FP32::class,
                    config = Gemma4LoadConfig(
                        quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
                        allowQuantized = true
                    )
                )

                println("Loading Gemma 4 from $path via DSL NATIVE_OPTIMIZED...")
                val runtime = ingestion.loadDslRuntimeNativeStreaming(
                    randomAccessProvider = { JvmRandomAccessSource.open(path.toString()) },
                    ctx = ctx,
                    dtype = FP32::class,
                    arena = quantArena
                )

                val tokenizer = JvmRandomAccessSource.open(path.toString()).use { source ->
                    GGUFTokenizer.fromRandomAccessSource(source)
                }

                val metadata = ModelMetadata(
                    family = "gemma",
                    architecture = "gemma4",
                    sourceFormat = "gguf"
                )
                val session = ChatSession(runtime, tokenizer, metadata)

                val tool = CalculatorTool()
                val rawResponses = mutableListOf<String>()
                val toolCallsHeard = mutableListOf<ToolCall>()
                val listener = object : AgentListener {
                    override fun onAssistantMessage(text: String) { rawResponses += text }
                    override fun onToolCalls(calls: List<ToolCall>) { toolCallsHeard += calls }
                }

                val prompt = "What is 17 * 23? Use the calculator tool."
                val response = session.runSingleTurn(
                    prompt = prompt,
                    tools = listOf(tool),
                    maxTokens = 256,
                    temperature = 0.0f,
                    listener = listener
                )

                val allText = rawResponses.joinToString("\n")
                println("--- raw round 1 response ---\n$allText\n---")
                println("final response: $response")

                assertTrue(
                    allText.contains("<|tool_call>"),
                    "real Gemma 4 E2B did not emit a <|tool_call> marker — the template grammar " +
                        "may not match the trained model. Raw response captured above. Record findings in " +
                        "gemma4-research/findings/tool_calling.md before adjusting the template."
                )
                assertTrue(
                    allText.contains("<tool_call|>"),
                    "found opener but no closer <tool_call|> — check if the model uses a different " +
                        "closing delimiter."
                )
                assertTrue(
                    toolCallsHeard.any { it.name == "calculator" },
                    "Gemma4ChatTemplate.parseToolCalls did not recover a 'calculator' call from the " +
                        "model output. Raw response above."
                )
            } finally {
                quantArena.close()
                memSegFactory.close()
            }
        }
    }

    companion object {
        @Suppress("unused") // referenced from KDoc
        private val modelPathHelp: String = "Set GEMMA4_E2B_MODEL_PATH=/path/to/gemma-4-e2b.gguf to enable."
    }
}
