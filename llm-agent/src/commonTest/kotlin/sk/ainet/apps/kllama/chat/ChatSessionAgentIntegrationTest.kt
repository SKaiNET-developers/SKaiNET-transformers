package sk.ainet.apps.kllama.chat

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonPrimitive
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
}
