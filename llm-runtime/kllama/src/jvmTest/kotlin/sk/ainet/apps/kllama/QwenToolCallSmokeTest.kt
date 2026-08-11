package sk.ainet.apps.kllama

import java.lang.foreign.Arena
import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.io.path.isDirectory
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonObject
import sk.ainet.apps.kllama.chat.AgentListener
import sk.ainet.apps.kllama.chat.ChatSession
import sk.ainet.apps.kllama.chat.ModelMetadataExtraction
import sk.ainet.apps.kllama.chat.Tool
import sk.ainet.apps.kllama.chat.ToolCall
import sk.ainet.apps.kllama.chat.ToolCallingMode
import sk.ainet.apps.kllama.chat.ToolCallingSupportResolver
import sk.ainet.apps.kllama.chat.ToolDefinition
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.tokenizer.TokenizerFactory
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.tensor.data.MemorySegmentTensorDataFactory
import sk.ainet.lang.types.FP32
import sk.ainet.models.llama.DecoderGgufMemSegConverter
import sk.ainet.models.llama.DecoderGgufWeightLoader
import sk.ainet.models.qwen.QwenNetworkLoader

/**
 * End-to-end validation of the generalized tool-calling architecture against
 * a **real** Qwen instruct GGUF (#43): metadata is extracted from the file
 * with [ModelMetadataExtraction.fromGgufFields], the provider is resolved by
 * **auto-detection only** (no explicit template name), and the agent loop
 * must round-trip a tool call.
 *
 * Gated on the `QWEN_MODEL_PATH` env var so CI stays green without weights
 * on disk. To run locally:
 * ```
 * export QWEN_MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct-F16.gguf
 * ./gradlew :llm-runtime:kllama:jvmTest --tests '*QwenToolCallSmokeTest*'
 * ```
 *
 * Loads via the same DSL path kllama's CLI uses for Qwen GGUFs
 * ([DecoderGgufWeightLoader] → [QwenNetworkLoader] → [OptimizedLLMRuntime]
 * in DIRECT mode; parity with the legacy path pinned by QwenDslLegacyParityTest).
 */
class QwenToolCallSmokeTest {

    private class CalculatorTool : Tool {
        var invoked = false
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
            val expr = arguments["expression"]?.jsonPrimitive?.content
                ?: return "error: no expression"
            return try {
                val parts = expr.split("*").map { it.trim() }.filter { it.isNotEmpty() }
                if (parts.size == 2) (parts[0].toLong() * parts[1].toLong()).toString()
                else "error: unexpected expression shape"
            } catch (e: Exception) {
                "error: ${e.message}"
            }
        }
    }

    private fun realModelPathOrNull(): Path? {
        val modelPath = System.getenv("QWEN_MODEL_PATH")?.trim().orEmpty()
        if (modelPath.isEmpty()) {
            println("[skip] QWEN_MODEL_PATH not set — skipping real-checkpoint smoke test.")
            return null
        }
        val path = Path.of(modelPath)
        if (!path.exists() || path.isDirectory() || !path.toString().endsWith(".gguf")) {
            println("[skip] QWEN_MODEL_PATH=$modelPath is not an existing .gguf file — skipping.")
            return null
        }
        return path
    }

    private fun extractMetadata(path: Path) =
        JvmRandomAccessSource.open(path.toString()).use { source ->
            StreamingGGUFReader.open(source).use { reader ->
                ModelMetadataExtraction.fromGgufFields(reader.fields)
            }
        }

    /**
     * #43 step 1: metadata extraction + provider resolution against the real
     * file — reads only GGUF header fields, no weights, so it runs in seconds.
     *
     * Verified passing against Qwen2.5-0.5B-Instruct-F16.gguf:
     * `family=qwen, arch=qwen2, hints=[<tool_call>, <tool_response>, <|im_start|>]`
     * → provider `qwen`, mode NATIVE, auto-detected (no explicit family).
     */
    @Test
    fun `real Qwen GGUF metadata auto-resolves the qwen provider`() {
        val path = realModelPathOrNull() ?: return
        val metadata = extractMetadata(path)
        println("Extracted metadata: family=${metadata.family}, arch=${metadata.architecture}, " +
            "hints=${metadata.tokenizerHints}")
        val resolution = ToolCallingSupportResolver.resolveWithDiagnostics(metadata)
        println("Resolved provider: ${resolution.provider.family} (mode=${resolution.mode}, reason: ${resolution.reason})")
        assertEquals("qwen", resolution.provider.family, "auto-detection must select the Qwen provider")
        assertEquals(ToolCallingMode.NATIVE, resolution.mode)
    }

    /**
     * #43 step 2: end-to-end agent loop over the real checkpoint.
     *
     * NOTE: currently expected to fail on develop for reasons *outside* the
     * tool-calling architecture — the DSL Qwen runtime emits degenerate text
     * on real checkpoints (garbage multilingual tokens; observed with
     * Qwen2.5-0.5B-Instruct F16). That output-quality gate is tracked in
     * #118 (validate DSL Qwen output against the HF reference). Resolution,
     * template rendering, and the agent loop itself all execute; once #118
     * lands this test pins the full round-trip.
     */
    @Test
    fun `real Qwen instruct GGUF tool-calls through auto-detected provider`() {
        val path = realModelPathOrNull() ?: return
        val metadata = extractMetadata(path)

        runBlocking {
            val memSegFactory = MemorySegmentTensorDataFactory()
            val ctx = DirectCpuExecutionContext(tensorDataFactory = memSegFactory)
            val quantArena = Arena.ofShared()
            try {
                val loader = DecoderGgufWeightLoader(
                    randomAccessProvider = { JvmRandomAccessSource.open(path.toString()) },
                    quantPolicy = QuantPolicy.NATIVE_OPTIMIZED,
                    acceptedArchitectures = setOf("qwen2", "qwen3", "qwen35"),
                )
                println("Loading Qwen GGUF from $path (DSL streaming mode)...")
                val rawWeights = loader.loadToMapStreaming<FP32, Float>(ctx)
                val weights = if (rawWeights.quantTypes.isNotEmpty()) {
                    DecoderGgufMemSegConverter.convert(rawWeights, ctx, quantArena)
                } else rawWeights

                val qwenModel = QwenNetworkLoader.fromWeights(weights)
                val runtime = OptimizedLLMRuntime(
                    model = qwenModel,
                    ctx = ctx,
                    mode = OptimizedLLMMode.DIRECT,
                    dtype = FP32::class,
                    bos = weights.metadata.bosTokenId,
                )
                val tokenizer = JvmRandomAccessSource.open(path.toString()).use { source ->
                    TokenizerFactory.fromGgufSource(source)
                }

                // No explicit template name: ChatSession must resolve via metadata.
                val session = ChatSession(runtime, tokenizer, metadata)
                assertEquals("qwen", session.providerFamily)

                val tool = CalculatorTool()
                val rawResponses = mutableListOf<String>()
                val toolCallsHeard = mutableListOf<ToolCall>()
                val listener = object : AgentListener {
                    override fun onAssistantMessage(text: String) { rawResponses += text }
                    override fun onToolCalls(calls: List<ToolCall>) { toolCallsHeard += calls }
                }

                val response = session.runSingleTurn(
                    prompt = "What is 17 * 23? Use the calculator tool.",
                    tools = listOf(tool),
                    maxTokens = 256,
                    temperature = 0.0f,
                    listener = listener
                )

                val allText = rawResponses.joinToString("\n")
                println("--- raw responses ---\n$allText\n---")
                println("final response: $response")

                assertTrue(
                    toolCallsHeard.any { it.name == "calculator" },
                    "Qwen did not produce a parseable 'calculator' tool call through the " +
                        "auto-detected provider. Raw response captured above."
                )
                assertTrue(tool.invoked, "calculator tool was never executed by the agent loop")
            } finally {
                quantArena.close()
                memSegFactory.close()
            }
        }
    }
}
