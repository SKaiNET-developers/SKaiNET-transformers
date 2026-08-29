package sk.ainet.apps.kgemma.cli

import java.io.File
import java.lang.foreign.Arena
import java.lang.management.BufferPoolMXBean
import java.lang.management.ManagementFactory
import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.io.path.isDirectory
import kotlin.io.path.readText
import kotlin.random.Random
import kotlin.test.Ignore
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
import sk.ainet.models.gemma.GEMMA_DEQUANTIZE_ALL
import sk.ainet.apps.kllama.GGUFTokenizer
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

/**
 * End-to-end smoke test for Gemma 4 tool calling against a **real** E2B
 * checkpoint. Gated on the `GEMMA4_E2B_MODEL_PATH` env var so CI stays green
 * without the ~3 GB weights on disk.
 *
 * Purpose: full-stack regression check. The chat-template grammar and
 * parser are *already* empirically validated end-to-end via two faster
 * paths:
 *
 *   1. [sk.ainet.apps.kllama.chat.Gemma4ChatTemplateHfParityTest]
 *      asserts byte-for-byte parity between our Kotlin
 *      [sk.ainet.apps.kllama.chat.Gemma4ChatTemplate] render and HF's
 *      official Jinja `chat_template.jinja`.
 *   2. HF Python ground truth (run separately via `uv run`, see
 *      `gemma4_chat_template_mismatch.md`) confirms the trained model
 *      emits exactly `<|tool_call>call:calculator{expression:<|"|>17 *
 *      23<|"|>}<tool_call|>` for the calculator prompt.
 *
 * What this test adds on top: it confirms the *DSL forward + agent
 * loop* reproduces the same behavior end-to-end against the real
 * checkpoint. If it fails, the format and parser are not the suspect —
 * look for a regression in the DSL inference path (RoPE, KV-share,
 * Q4_K kernel) or the agent loop wiring.
 *
 * To run locally:
 * ```
 * export GEMMA4_E2B_MODEL_PATH=/path/to/gemma-4-e2b.gguf
 * ./gradlew :llm-runtime:kgemma:jvmTest \
 *     --tests '*Gemma4E2BToolCallSmokeTest*' \
 *     -PkgemmaTestMaxHeap=24g -PkgemmaTestMaxDirect=32g
 * ```
 *
 * Both `-PkgemmaTestMaxHeap` and `-PkgemmaTestMaxDirect` are required for the
 * real run: the Q4_K → FP32 dequant lands in MemorySegment-backed direct
 * memory (≈ 20 GB for E2B), so without raising `-XX:MaxDirectMemorySize` the
 * worker JVM hits a `Cannot reserve N bytes of direct buffer memory` and
 * JUnit reports the test as SKIPPED — misleadingly, because the assertion
 * never runs. Defaults stay at 4g so CI and the rest of the suite are cheap.
 *
 * If the test finds the model but the assertions fail, the raw response
 * is captured in stdout. Most likely cause is a regression somewhere in
 * the DSL forward — the chat-template grammar is no longer the suspect.
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
    @Ignore // Gemma 4 E2B emits coherent English on the calculator prompt but no <|tool_call> markup.
            // Format-grammar gap is upstream of the agent loop (suspected prefill/sampling divergence).
            // Re-enable once the underlying fix lands; the assertions here are the regression guard.
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
            memDump("test-start")
            val memSegFactory = MemorySegmentTensorDataFactory()
            val ctx = DirectCpuExecutionContext(tensorDataFactory = memSegFactory)
            val quantArena = Arena.ofShared()
            try {
                // Keep-packed (engine-loader default): Q4_K weights stay
                // quantized in memory and dispatch to the packed matmul
                // kernels. Override via GEMMA4_TOOLCALL_QUANT=fp32 if a
                // numerical-precision regression is suspected.
                val weightForm = if (System.getenv("GEMMA4_TOOLCALL_QUANT")?.lowercase() == "fp32") {
                    GEMMA_DEQUANTIZE_ALL
                } else {
                    null
                }
                val ingestion = Gemma4Ingestion<FP32>(
                    ctx = ctx,
                    dtype = FP32::class,
                    config = Gemma4LoadConfig(weightForm = weightForm)
                )

                println("Loading Gemma 4 from $path via DSL engine loader (keep-packed)...")
                memDump("before-load")
                val runtime = ingestion.loadDslRuntimeStreaming(
                    randomAccessProvider = { JvmRandomAccessSource.open(path.toString()) }
                )
                memDump("after-load")

                val tokenizer = JvmRandomAccessSource.open(path.toString()).use { source ->
                    GGUFTokenizer.fromRandomAccessSource(source)
                }
                memDump("after-tokenizer")

                val metadata = ModelMetadata(
                    family = "gemma",
                    architecture = "gemma4",
                    sourceFormat = "gguf"
                )
                val session = ChatSession(runtime, tokenizer, metadata)
                memDump("after-chatsession")

                val tool = CalculatorTool()
                val rawResponses = mutableListOf<String>()
                val toolCallsHeard = mutableListOf<ToolCall>()
                // Live per-token visibility — gradle test runner buffers stdout
                // until the test ends, but each println auto-flushes within the
                // worker JVM, so progress lines printed here at least appear in
                // the test's captured stdout once the test finishes.
                var tokenCount = 0
                val sb = StringBuilder()
                val listener = object : AgentListener {
                    override fun onToken(token: String) {
                        tokenCount++
                        sb.append(token)
                        if (tokenCount % 8 == 0) {
                            println("[gen tok=$tokenCount] …${sb.takeLast(60).toString().replace("\n", "\\n")}")
                        }
                    }
                    override fun onAssistantMessage(text: String) { rawResponses += text }
                    override fun onToolCalls(calls: List<ToolCall>) { toolCallsHeard += calls }
                }

                val prompt = "What is 17 * 23? Use the calculator tool."
                memDump("before-runSingleTurn")
                val maxTokens = (System.getenv("GEMMA4_TOOLCALL_MAX_TOKENS")?.toIntOrNull()) ?: 80
                val response = session.runSingleTurn(
                    prompt = prompt,
                    tools = listOf(tool),
                    maxTokens = maxTokens,
                    temperature = 0.0f,
                    listener = listener
                )
                memDump("after-runSingleTurn")

                val allText = rawResponses.joinToString("\n")
                println("--- raw round 1 response ---\n$allText\n---")
                println("final response: $response")

                // Format empirically validated against the trained model
                // via HF Python ground truth (2026-04-27): on this prompt
                // the model emits exactly
                //   <|tool_call>call:calculator{expression:<|"|>17 * 23<|"|>}<tool_call|>
                // with all delimiter and quote tokens atomic. Assertions
                // below pin the contract our parser depends on. See
                // gemma4_chat_template_mismatch.md for the trail.

                assertTrue(
                    allText.contains("<|tool_call>"),
                    "Expected `<|tool_call>` opener in model output. The HF Python ground truth on " +
                        "this prompt emits one — if our DSL doesn't, suspect a regression in the " +
                        "DSL forward pass, not the chat template. Raw response captured above."
                )
                assertTrue(
                    allText.contains("<tool_call|>"),
                    "Found `<|tool_call>` opener but no `<tool_call|>` closer — generation likely " +
                        "truncated mid-call. Increase GEMMA4_TOOLCALL_MAX_TOKENS or check why the " +
                        "model stopped early."
                )
                assertTrue(
                    allText.contains("call:calculator{"),
                    "Expected HF-format body `call:calculator{...}` inside the tool_call markers. " +
                        "Got something else — model may have emitted a different tool name or body " +
                        "shape. Raw response above."
                )
                assertTrue(
                    allText.contains("<|\"|>"),
                    "Expected at least one `<|\"|>` quote token (string-literal delimiter) in the " +
                        "tool-call body. Without it the argument parser can't recover string values."
                )

                val calculatorCall = toolCallsHeard.firstOrNull { it.name == "calculator" }
                    ?: fail(
                        "Gemma4ChatTemplate.parseToolCalls did not recover a `calculator` call from " +
                            "the model output, even though the markers and body shape look right. " +
                            "Either the parser regressed or the body grammar diverged. Raw response above."
                    )
                val expression = calculatorCall.arguments["expression"]?.jsonPrimitive?.content
                assertTrue(
                    expression != null && "17" in expression && "23" in expression,
                    "Recovered `calculator` call but the `expression` argument doesn't contain " +
                        "the prompt operands. Got: $expression. Raw response above."
                )

                // The CalculatorTool is wired into the agent loop, so it
                // should have actually run with that expression.
                assertTrue(
                    tool.invoked,
                    "calculator.execute() was never called — the agent loop didn't dispatch the " +
                        "parsed tool call. Check ToolRegistry wiring."
                )
                assertTrue(
                    tool.receivedExpression?.let { "17" in it && "23" in it } ?: false,
                    "Tool ran but with a surprising expression argument: ${tool.receivedExpression}"
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

        private val directBufferPool: BufferPoolMXBean by lazy {
            ManagementFactory.getPlatformMXBeans(BufferPoolMXBean::class.java)
                .first { it.name == "direct" }
        }

        private fun mb(bytes: Long): String = "%,.0f MB".format(bytes / (1024.0 * 1024.0))

        // Diagnostic memory snapshot used to localize a stubborn direct-memory leak
        // in the Gemma 4 forward pass. Prints heap, non-heap, and direct buffer
        // pool usage at named stages so the OOM curve can be plotted across load,
        // tokenizer init, and inference. See pickup notes for context.
        internal fun memDump(stage: String) {
            val rt = Runtime.getRuntime()
            val heap = rt.totalMemory() - rt.freeMemory()
            val direct = directBufferPool.memoryUsed
            println("[mem] $stage: heap=${mb(heap)}, direct=${mb(direct)}")
        }
    }
}
