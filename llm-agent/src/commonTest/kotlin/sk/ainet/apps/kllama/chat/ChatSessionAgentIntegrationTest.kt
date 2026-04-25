package sk.ainet.apps.kllama.chat

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.add
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonObject
import sk.ainet.apps.llm.InferenceRuntime
import sk.ainet.apps.llm.Tokenizer
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.types.FP32

/**
 * End-to-end wiring test for [ChatSession] → [AgentLoop] → [Tool] via a
 * mock inference runtime that emits a pre-baked Gemma-4-formatted tool
 * call. Validates every step of the agent pipeline except the actual
 * model math:
 *
 *   1. [ToolCallingSupportResolver] picks [Gemma4ToolCallingSupport]
 *      when metadata says `architecture = "gemma4"`.
 *   2. [Gemma4ChatTemplate] formats the prompt with `<|turn>…<turn|>`
 *      markers.
 *   3. The mock runtime "generates" `<|tool_call>{"name":"calculator",
 *      "args":{"expression":"3+4"}}<tool_call|>`.
 *   4. [Gemma4ChatTemplate.parseToolCalls] pulls the call out.
 *   5. The [Tool] executes with `expression="3+4"` and returns `7`.
 *   6. Agent loop records the call + result.
 *
 * No real model required — the mock's `forward()` returns a logits
 * tensor whose argmax points to the next byte in the scripted output.
 * Temperature 0 (greedy) in the agent loop turns argmax into the
 * sampled token.
 */
class ChatSessionAgentIntegrationTest {

    /** Build a `[vocabSize]`-shape FP32 tensor where one index has a big value. */
    private fun oneHotLogits(argmax: Int, vocabSize: Int): Tensor<FP32, Float> {
        val buf = FloatArray(vocabSize)
        buf[argmax] = 100f
        val data = DenseFloatArrayTensorData<FP32>(Shape(vocabSize), buf)
        return VoidOpsTensor(data, FP32::class)
    }

    /** Byte-level tokenizer — each character gets its own token id. */
    private class ByteTokenizer : Tokenizer {
        override val eosTokenId: Int = 0
        override val bosTokenId: Int = 1
        override val vocabSize: Int = 1024  // generous, each char fits in Int
        override fun encode(text: String): IntArray =
            text.map { it.code + 2 }.toIntArray()  // reserve 0/1 for EOS/BOS
        override fun decode(tokens: IntArray): String =
            tokens.joinToString("") { decode(it) }
        override fun decode(token: Int): String = when (token) {
            eosTokenId, bosTokenId -> ""
            else -> (token - 2).toChar().toString()
        }
    }

    /**
     * Mock runtime that emits a fixed sequence of token IDs from
     * [scriptedOutput], regardless of the incoming `tokenId`. The
     * sampler sees logits with `argmax == scriptedOutput[cursor]` and
     * picks that token.
     *
     * Prompt forwards are folded into the same cursor: they advance
     * through filler positions (argmax = 0 = EOS-equivalent for our
     * tokenizer, but the sampler discards prompt logits). The
     * [promptLen]-th forward drives the first sample, and subsequent
     * forwards drive the next samples in order.
     */
    private inner class MockRuntime(
        val promptLen: Int,
        val scriptedOutput: IntArray,
        val eosTokenId: Int,
        val vocabSize: Int
    ) : InferenceRuntime<FP32> {
        var cursor = 0
            private set
        override fun reset() {
            cursor = 0
        }
        override fun forward(tokenId: Int): Tensor<FP32, Float> {
            // cursor index semantics: after N forwards, `cursor` == N
            // BEFORE this call's returned logits drive the next sample.
            // - During prompt feed (cursor < promptLen - 1): logits
            //   discarded by generateUntilStop.
            // - After last prompt token (cursor == promptLen - 1): logits
            //   drive scriptedOutput[0].
            // - Each subsequent forward(sampledToken) (cursor =
            //   promptLen..): logits drive scriptedOutput[cursor -
            //   promptLen + 1].
            val argmax = when {
                cursor < promptLen - 1 -> eosTokenId
                cursor - promptLen + 1 < scriptedOutput.size ->
                    scriptedOutput[cursor - promptLen + 1]
                else -> eosTokenId
            }
            cursor++
            return oneHotLogits(argmax, vocabSize)
        }
    }

    /** Simple calculator tool that handles a fixed expression for this test. */
    private class FixedCalculatorTool(private val expected: String, private val result: String) : Tool {
        var invoked = false
            private set
        var receivedExpression: String? = null
            private set
        override val definition = ToolDefinition(
            name = "calculator",
            description = "Compute a simple arithmetic expression",
            parameters = JsonObject(emptyMap())
        )
        override fun execute(arguments: JsonObject): String {
            invoked = true
            receivedExpression = arguments["expression"]?.jsonPrimitive?.content
            return result
        }
    }

    @Test
    fun `ChatSession drives mock Gemma 4 runtime through agent loop with tool call`() {
        val metadata = ModelMetadata(architecture = "gemma4", family = "gemma4")
        val tokenizer = ByteTokenizer()

        // Compute the exact prompt length the agent loop will feed. The
        // template is deterministic, so we can round-trip to get the
        // token count for round 1 (before any tool messages are added).
        val template = Gemma4ChatTemplate()
        val tool = FixedCalculatorTool(expected = "3+4", result = "7")
        val round1Messages = listOf(
            ChatMessage(ChatRole.SYSTEM, "You are a helpful assistant with access to tools."),
            ChatMessage(ChatRole.USER, "what is 3 + 4?")
        )
        val round1Prompt = template.apply(
            round1Messages,
            listOf(tool.definition),
            addGenerationPrompt = true
        )
        val round1PromptLen = tokenizer.encode(round1Prompt).size

        val toolCallText =
            "I'll compute that.\n<|tool_call>{\"name\":\"calculator\",\"args\":{\"expression\":\"3+4\"}}<tool_call|>"
        val scriptedOutput = tokenizer.encode(toolCallText)

        val mock = MockRuntime(
            promptLen = round1PromptLen,
            scriptedOutput = scriptedOutput,
            eosTokenId = tokenizer.eosTokenId,
            vocabSize = tokenizer.vocabSize
        )

        val session = ChatSession(mock, tokenizer, metadata)

        // Sanity: provider and template came out right.
        assertEquals("gemma4", session.providerFamily)
        assertTrue(session.chatTemplate is Gemma4ChatTemplate)

        val toolCallsSeen = mutableListOf<ToolCall>()
        val toolResultsSeen = mutableListOf<Pair<String, String>>()
        val listener = object : AgentListener {
            override fun onToolCalls(calls: List<ToolCall>) { toolCallsSeen += calls }
            override fun onToolResult(call: ToolCall, result: String) {
                toolResultsSeen += call.name to result
            }
        }

        // Drive one agent round. The mock stops with EOS right after the
        // tool call, which forces AgentLoop to register the call + run
        // the tool. The second round would need a fresh MockRuntime with
        // an updated promptLen — we don't exercise that here because
        // this test is about wiring, not the full multi-round loop.
        session.runSingleTurn(
            prompt = "what is 3 + 4?",
            tools = listOf(tool),
            maxTokens = scriptedOutput.size + 4,
            temperature = 0.0f,
            listener = listener
        )

        // Agent loop found the tool call.
        assertEquals(1, toolCallsSeen.size, "expected exactly 1 parsed tool call, got ${toolCallsSeen.size}")
        assertEquals("calculator", toolCallsSeen[0].name)
        assertEquals("3+4", toolCallsSeen[0].arguments["expression"]?.jsonPrimitive?.content)

        // Tool got invoked.
        assertTrue(tool.invoked, "calculator tool never executed")
        assertEquals("3+4", tool.receivedExpression)
        assertEquals(1, toolResultsSeen.size)
        assertEquals("calculator" to "7", toolResultsSeen[0])
    }

    @Test
    fun `ChatSession honors systemPrompt override in runSingleTurn`() {
        val metadata = ModelMetadata(architecture = "gemma4", family = "gemma4")
        val tokenizer = ByteTokenizer()
        val template = Gemma4ChatTemplate()
        val tool = FixedCalculatorTool(expected = "3+4", result = "7")

        // Distinctive custom system prompt — different length from the default
        // so any fall-through to the hard-coded default would desync the mock.
        val customSystemPrompt = "Custom prompt for this turn only."

        val round1Prompt = template.apply(
            listOf(
                ChatMessage(ChatRole.SYSTEM, customSystemPrompt),
                ChatMessage(ChatRole.USER, "what is 3 + 4?")
            ),
            listOf(tool.definition),
            addGenerationPrompt = true
        )
        val round1PromptLen = tokenizer.encode(round1Prompt).size

        val toolCallText =
            "<|tool_call>{\"name\":\"calculator\",\"args\":{\"expression\":\"3+4\"}}<tool_call|>"
        val scriptedOutput = tokenizer.encode(toolCallText)

        val mock = MockRuntime(
            promptLen = round1PromptLen,
            scriptedOutput = scriptedOutput,
            eosTokenId = tokenizer.eosTokenId,
            vocabSize = tokenizer.vocabSize
        )

        val session = ChatSession(mock, tokenizer, metadata)
        val parsed = mutableListOf<ToolCall>()
        val listener = object : AgentListener {
            override fun onToolCalls(calls: List<ToolCall>) { parsed += calls }
            override fun onToolResult(call: ToolCall, result: String) {}
        }

        session.runSingleTurn(
            prompt = "what is 3 + 4?",
            tools = listOf(tool),
            maxTokens = scriptedOutput.size + 4,
            temperature = 0.0f,
            systemPrompt = customSystemPrompt,
            listener = listener
        )

        // If the override was ignored, the prompt bytes fed to the mock
        // would differ from round1PromptLen and the cursor would desync,
        // breaking the tool call parse. A clean parse is the assertion.
        assertEquals(1, parsed.size)
        assertEquals("calculator", parsed[0].name)
        assertEquals("3+4", parsed[0].arguments["expression"]?.jsonPrimitive?.content)
    }

    @Test
    fun `ChatSession exposes constructor-level defaultSystemPrompt`() {
        val metadata = ModelMetadata(architecture = "gemma4", family = "gemma4")
        val tokenizer = ByteTokenizer()
        val mock = MockRuntime(promptLen = 1, scriptedOutput = IntArray(0), tokenizer.eosTokenId, tokenizer.vocabSize)

        val defaulted = ChatSession(mock, tokenizer, metadata)
        assertEquals(ChatSession.DEFAULT_SYSTEM_PROMPT, defaulted.defaultSystemPrompt)

        val customized = ChatSession(mock, tokenizer, metadata, defaultSystemPrompt = "Session-level default")
        assertEquals("Session-level default", customized.defaultSystemPrompt)
    }

    /**
     * Tool that declares `expression` as a required string argument. Used to
     * drive a scripted model output that omits it and prove the validator
     * intercepts before the tool's `execute()` runs.
     */
    private class StrictCalculatorTool : Tool {
        var invoked = false
            private set
        override val definition = ToolDefinition(
            name = "calculator",
            description = "Evaluate a math expression",
            parameters = buildJsonObject {
                put("type", "object")
                putJsonObject("properties") {
                    putJsonObject("expression") { put("type", "string") }
                }
                put("required", buildJsonArray { add("expression") })
            }
        )
        override fun execute(arguments: JsonObject): String {
            invoked = true
            return "unreachable"
        }
    }

    @Test
    fun `AgentLoop feeds validation error back when tool call omits required argument`() {
        val tokenizer = ByteTokenizer()
        val template = Gemma4ChatTemplate()
        val tool = StrictCalculatorTool()

        val round1Messages = listOf(
            ChatMessage(ChatRole.SYSTEM, "You are a helpful assistant with access to tools."),
            ChatMessage(ChatRole.USER, "calc please")
        )
        val round1Prompt = template.apply(
            round1Messages,
            listOf(tool.definition),
            addGenerationPrompt = true
        )
        val round1PromptLen = tokenizer.encode(round1Prompt).size

        // Scripted tool call WITHOUT the required "expression" field.
        val badToolCallText = "<|tool_call>{\"name\":\"calculator\",\"args\":{\"precision\":2}}<tool_call|>"
        val scriptedOutput = tokenizer.encode(badToolCallText)

        val mock = MockRuntime(
            promptLen = round1PromptLen,
            scriptedOutput = scriptedOutput,
            eosTokenId = tokenizer.eosTokenId,
            vocabSize = tokenizer.vocabSize
        )

        // Drive AgentLoop directly with maxToolRounds=1 so only round 1 runs —
        // proving validation + error-feedback happen without needing to script
        // a recovery round.
        val registry = ToolRegistry().also { it.register(tool) }
        val loop = AgentLoop<FP32>(
            runtime = mock,
            template = template,
            toolRegistry = registry,
            eosTokenId = tokenizer.eosTokenId,
            config = AgentConfig(
                maxToolRounds = 1,
                maxTokensPerRound = scriptedOutput.size + 4,
                temperature = 0.0f
            ),
            decode = { tokenizer.decode(it) }
        )

        val validationFailures = mutableListOf<Pair<String, String>>()
        val toolResults = mutableListOf<String>()
        val listener = object : AgentListener {
            override fun onToolCallValidationFailed(call: ToolCall, reason: String) {
                validationFailures += call.name to reason
            }
            override fun onToolResult(call: ToolCall, result: String) {
                toolResults += result
            }
        }

        val messages = round1Messages.toMutableList()
        loop.runWithEncoder(messages, encode = { tokenizer.encode(it) }, listener = listener)

        assertEquals(1, validationFailures.size, "expected one validation failure")
        assertEquals("calculator", validationFailures[0].first)
        assertTrue(
            validationFailures[0].second.contains("missing required argument 'expression'"),
            "reason should name the missing field, got: ${validationFailures[0].second}"
        )
        assertEquals(1, toolResults.size)
        assertTrue(
            toolResults[0].startsWith("validation error:"),
            "tool result should report validation error, got: ${toolResults[0]}"
        )
        // Tool's execute() must never run when validation fails.
        assertFalse(tool.invoked, "tool.execute() must not run when validation fails")
    }

    @Test
    fun `AgentLoop surfaces thinking blocks to listener and strips them from persisted messages`() {
        val tokenizer = ByteTokenizer()
        val template = Gemma4ChatTemplate()
        val tool = FixedCalculatorTool(expected = "3+4", result = "7")

        val round1Messages = listOf(
            ChatMessage(ChatRole.SYSTEM, "You are a helpful assistant with access to tools."),
            ChatMessage(ChatRole.USER, "what is 3 + 4?")
        )
        val round1Prompt = template.apply(
            round1Messages,
            listOf(tool.definition),
            addGenerationPrompt = true
        )
        val round1PromptLen = tokenizer.encode(round1Prompt).size

        // Scripted output: thinking block followed by a tool call — the model
        // reasons, then decides to call the calculator.
        val scriptedText =
            "<|think>I should use the calculator for this.<think|>\n" +
            "<|tool_call>{\"name\":\"calculator\",\"args\":{\"expression\":\"3+4\"}}<tool_call|>"
        val scriptedOutput = tokenizer.encode(scriptedText)

        val mock = MockRuntime(
            promptLen = round1PromptLen,
            scriptedOutput = scriptedOutput,
            eosTokenId = tokenizer.eosTokenId,
            vocabSize = tokenizer.vocabSize
        )

        val registry = ToolRegistry().also { it.register(tool) }
        val loop = AgentLoop<FP32>(
            runtime = mock,
            template = template,
            toolRegistry = registry,
            eosTokenId = tokenizer.eosTokenId,
            config = AgentConfig(
                maxToolRounds = 1,
                maxTokensPerRound = scriptedOutput.size + 4,
                temperature = 0.0f
            ),
            decode = { tokenizer.decode(it) }
        )

        val thinkingHeard = mutableListOf<String>()
        val listener = object : AgentListener {
            override fun onThinking(text: String) { thinkingHeard += text }
        }

        val messages = round1Messages.toMutableList()
        loop.runWithEncoder(messages, encode = { tokenizer.encode(it) }, listener = listener)

        // Listener saw the thinking content.
        assertEquals(1, thinkingHeard.size)
        assertEquals("I should use the calculator for this.", thinkingHeard[0])

        // Thinking must not leak into the persisted assistant message.
        val assistantMsg = messages.firstOrNull { it.role == ChatRole.ASSISTANT }
            ?: error("expected an assistant message in conversation")
        assertFalse(assistantMsg.content.contains("<|think>"), "thinking opener leaked into assistant message")
        assertFalse(assistantMsg.content.contains("I should use the calculator"), "thinking text leaked into assistant message")

        // Tool still fired based on the raw text.
        assertTrue(tool.invoked, "calculator tool should still run — tool_call parsing operates on the raw response")
        assertEquals("3+4", tool.receivedExpression)
    }
}
