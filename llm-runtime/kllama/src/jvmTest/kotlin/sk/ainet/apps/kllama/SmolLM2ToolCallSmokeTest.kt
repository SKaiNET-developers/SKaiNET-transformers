package sk.ainet.apps.kllama

import java.lang.foreign.Arena
import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.io.path.isDirectory
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonObject
import sk.ainet.apps.kllama.chat.AgentListener
import sk.ainet.apps.kllama.chat.ChatSession
import sk.ainet.apps.kllama.chat.ModelMetadata
import sk.ainet.apps.kllama.chat.Tool
import sk.ainet.apps.kllama.chat.ToolCall
import sk.ainet.apps.kllama.chat.ToolDefinition
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.tensor.data.MemorySegmentTensorDataFactory
import sk.ainet.lang.types.FP32
import sk.ainet.models.llama.LlamaRuntime

/**
 * End-to-end smoke test for SmolLM2-1.7B-Instruct tool calling against a
 * **real** GGUF Q8_0 checkpoint. Gated on the `SMOLLM2_MODEL_PATH` env var so
 * CI stays green without the ~1.8 GB weights on disk.
 *
 * Purpose: empirically pin down whether the trained SmolLM2-1.7B-Instruct
 * model emits a `<tool_call>...</tool_call>` block parseable by
 * [sk.ainet.apps.kllama.chat.SmolLMChatTemplate] when prompted with the
 * official system-prompt recipe. A pass closes the question "can a 1.7B
 * model usefully tool-call on this runtime?".
 *
 * To run locally:
 * ```
 * export SMOLLM2_MODEL_PATH=/path/to/SmolLM2-1.7B-Instruct-Q8_0.gguf
 * ./gradlew :llm-runtime:kllama:jvmTest \
 *     --tests '*SmolLM2ToolCallSmokeTest*'
 * ```
 *
 * Q8_0 of a 1.7B model fits well under the default `-Xmx`, so no heap-tuning
 * properties are required (unlike `Gemma4E2BToolCallSmokeTest`).
 *
 * On failure the raw response is printed so the format deviation can be
 * characterized.
 *
 * Loads via the legacy [LlamaRuntime] path, matching how kllama's own CLI
 * still runs Llama-architecture GGUFs on develop (the DSL/OptimizedLLMRuntime
 * path is ~8x slower for Q8/Q4 Llama today — see cli/Main.kt). Migrating both
 * to the packed DSL path is tracked with the SmolLM2-135M cross-target
 * reproducer (transformers#272).
 */
@Suppress("DEPRECATION") // legacy LlamaRuntime ctor — consistent with cli/Main.kt's Llama path
class SmolLM2ToolCallSmokeTest {

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
            val expr = receivedExpression ?: return "error: no expression"
            return try {
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
    fun `real SmolLM2-1_7B-Instruct emits parseable tool_call against SmolLMChatTemplate`() {
        val modelPath = System.getenv("SMOLLM2_MODEL_PATH")?.trim().orEmpty()
        if (modelPath.isEmpty()) {
            println("[skip] SMOLLM2_MODEL_PATH not set — skipping real-checkpoint smoke test.")
            return
        }
        val path = Path.of(modelPath)
        if (!path.exists()) {
            println("[skip] SMOLLM2_MODEL_PATH=$modelPath does not exist — skipping.")
            return
        }
        val isGguf = !path.isDirectory() && path.toString().endsWith(".gguf")
        if (!isGguf) {
            println("[skip] SMOLLM2_MODEL_PATH=$modelPath is not a .gguf file — skipping (this test only exercises the GGUF path).")
            return
        }

        runBlocking {
            val memSegFactory = MemorySegmentTensorDataFactory()
            val ctx = DirectCpuExecutionContext(tensorDataFactory = memSegFactory)
            val quantArena = Arena.ofShared()
            try {
                val ingestion = LlamaIngestion<FP32>(
                    ctx = ctx,
                    dtype = FP32::class,
                    config = LlamaLoadConfig(
                        acceptedArchitectures = setOf("llama")
                    )
                )

                println("Loading SmolLM2 from $path via streaming Llama loader...")
                val runtimeWeights = ingestion.loadStreaming {
                    JvmRandomAccessSource.open(path.toString())
                }

                val backend = CpuAttentionBackend<FP32>(
                    ctx, runtimeWeights, FP32::class,
                    ropeFreqBase = runtimeWeights.metadata.ropeFreqBase,
                    maxContextLength = null
                )
                val runtime = LlamaRuntime<FP32>(
                    ctx, runtimeWeights, backend, FP32::class,
                    eps = runtimeWeights.metadata.rmsNormEps
                )

                val tokenizer = JvmRandomAccessSource.open(path.toString()).use { source ->
                    GGUFTokenizer.fromRandomAccessSource(source)
                }

                val metadata = ModelMetadata(
                    family = "smollm",
                    architecture = "llama",
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
                    allText.contains("<tool_call>"),
                    "real SmolLM2 did not emit a <tool_call> opener — the system-prompt recipe " +
                        "or template grammar may not match what the trained model expects. " +
                        "Raw response captured above."
                )
                assertTrue(
                    allText.contains("</tool_call>"),
                    "found opener but no closer </tool_call> — the model may be truncating before " +
                        "completing the tool-call block (try a higher maxTokens) or using a different " +
                        "closing delimiter."
                )
                assertTrue(
                    toolCallsHeard.any { it.name == "calculator" },
                    "SmolLMChatTemplate.parseToolCalls did not recover a 'calculator' call from the " +
                        "model output. Raw response above. Likely causes: argument key naming " +
                        "(`arguments` vs `args`), array-vs-object form, or hallucinated tool name."
                )
            } finally {
                quantArena.close()
                memSegFactory.close()
            }
        }
    }

    companion object {
        @Suppress("unused")
        private val modelPathHelp: String =
            "Set SMOLLM2_MODEL_PATH=/path/to/SmolLM2-1.7B-Instruct-Q8_0.gguf to enable."
    }
}
