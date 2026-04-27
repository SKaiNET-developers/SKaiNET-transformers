package sk.ainet.apps.kllama.chat

import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.assertFalse
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonObject
import kotlinx.serialization.json.putJsonArray

/**
 * Tests for the HF-faithful Gemma 4 chat template (per the Jinja
 * `chat_template.jinja` shipped with `google/gemma-4-e2b-it`). Earlier
 * versions of this file tested a JSON-flavored grammar that turned out
 * not to match what the model was trained on; running the smoke test
 * against the real E2B checkpoint produced prose instead of tool
 * calls. The format is now: per-tool `<|tool>declaration:NAME{…}<tool|>`
 * blocks, `call:NAME{key:value,…}` tool-call bodies, and
 * `response:NAME{key:value}` tool-response bodies, with `<|"|>` as the
 * string-literal quote token. See `gemma4_chat_template_mismatch.md`
 * memory for the empirical evidence trail.
 */
class Gemma4ChatTemplateTest {

    private val systemMsg = ChatMessage(ChatRole.SYSTEM, "You are helpful.")
    private val userMsg = ChatMessage(ChatRole.USER, "Hello!")

    private val sampleTool = ToolDefinition(
        name = "calculator",
        description = "Evaluate math expressions",
        parameters = buildJsonObject {
            put("type", "object")
            putJsonObject("properties") {
                putJsonObject("expression") {
                    put("type", "string")
                    put("description", "The expression to evaluate")
                }
            }
            putJsonArray("required") { add(JsonPrimitive("expression")) }
        }
    )

    @Test
    fun bosIsPrependedExactlyOnce() {
        val result = Gemma4ChatTemplate().apply(listOf(userMsg))
        assertTrue(result.startsWith("<bos>"), "rendered prompt must start with <bos>")
        assertEquals(1, countOf(result, "<bos>"), "<bos> must appear exactly once")
    }

    @Test
    fun basicFormat() {
        val result = Gemma4ChatTemplate().apply(listOf(systemMsg, userMsg))
        assertContains(result, "<|turn>system")
        assertContains(result, "You are helpful.")
        assertContains(result, "<|turn>user")
        assertContains(result, "Hello!")
        assertContains(result, "<turn|>")
        assertTrue(result.endsWith("<|turn>model\n"))
    }

    @Test
    fun systemRoleUsesSystemTurn() {
        val result = Gemma4ChatTemplate().apply(listOf(systemMsg), addGenerationPrompt = false)
        assertContains(result, "<|turn>system\n")
        assertContains(result, "You are helpful.")
        assertContains(result, "<turn|>")
        assertFalse(result.contains("<|turn>user"))
    }

    @Test
    fun assistantRoleUsesModelTurn() {
        val assistantMsg = ChatMessage(ChatRole.ASSISTANT, "Hi there!")
        val result = Gemma4ChatTemplate().apply(listOf(assistantMsg), addGenerationPrompt = false)
        assertContains(result, "<|turn>model")
        assertContains(result, "Hi there!")
        assertContains(result, "<turn|>")
    }

    @Test
    fun noGenerationPrompt() {
        val result = Gemma4ChatTemplate().apply(listOf(userMsg), addGenerationPrompt = false)
        assertFalse(result.endsWith("<|turn>model\n"))
        assertTrue(result.endsWith("<turn|>\n"))
    }

    // ---- HF tool-definition format ----

    @Test
    fun toolDefinitionUsesDeclarationFormatWithCustomQuote() {
        val result = Gemma4ChatTemplate().apply(listOf(userMsg), tools = listOf(sampleTool))

        // Per-tool wrapping with HF `declaration:NAME{...}` body
        assertContains(result, "<|tool>declaration:calculator{")
        assertContains(result, "<tool|>")

        // Description uses the <|"|> quote token
        assertContains(result, "description:<|\"|>Evaluate math expressions<|\"|>")

        // Type names are uppercased
        assertContains(result, "type:<|\"|>OBJECT<|\"|>")
        assertContains(result, "type:<|\"|>STRING<|\"|>")

        // Required list uses <|"|> quotes
        assertContains(result, "required:[<|\"|>expression<|\"|>]")

        // Tool block goes inside a system turn
        assertContains(result, "<|turn>system")

        // CRUCIALLY: standard JSON should NOT appear — that's what the
        // model wasn't trained on.
        assertFalse(result.contains("\"name\":\"calculator\""),
            "tool block must NOT use standard JSON formatting; the model wasn't trained on that")
        assertFalse(result.contains("\"type\":\"string\""),
            "type names must be uppercased (STRING, not \"string\")")
    }

    @Test
    fun multipleToolsGetSeparateToolBlocks() {
        val tool2 = ToolDefinition(
            name = "lookup",
            description = "Lookup a value",
            parameters = buildJsonObject {
                put("type", "object")
                putJsonObject("properties") {
                    putJsonObject("key") { put("type", "string") }
                }
            }
        )
        val result = Gemma4ChatTemplate().apply(listOf(userMsg), tools = listOf(sampleTool, tool2))

        // Each tool gets its own <|tool>...<tool|> block — NOT one block
        // wrapping a JSON array of all tools.
        assertEquals(2, countOf(result, "<|tool>"))
        assertEquals(2, countOf(result, "<tool|>"))
        assertContains(result, "<|tool>declaration:calculator{")
        assertContains(result, "<|tool>declaration:lookup{")
    }

    // ---- Tool-call output (model → us) parsing ----

    @Test
    fun parseToolCallWithCallBody() {
        val text = "Let me run that.\n" +
            "<|tool_call>call:calculator{expression:<|\"|>2 + 3<|\"|>}<tool_call|>"
        val calls = Gemma4ChatTemplate().parseToolCalls(text)
        assertEquals(1, calls.size)
        assertEquals("calculator", calls[0].name)
        assertEquals(JsonPrimitive("2 + 3"), calls[0].arguments["expression"])
    }

    @Test
    fun parseToolCallNoArgs() {
        val text = "<|tool_call>call:list_files{}<tool_call|>"
        val calls = Gemma4ChatTemplate().parseToolCalls(text)
        assertEquals(1, calls.size)
        assertEquals("list_files", calls[0].name)
        assertTrue(calls[0].arguments.isEmpty())
    }

    @Test
    fun parseToolCallWithMixedScalarTypes() {
        val text = "<|tool_call>call:fn{name:<|\"|>foo<|\"|>,n:42,active:true,ratio:3.14}<tool_call|>"
        val calls = Gemma4ChatTemplate().parseToolCalls(text)
        assertEquals(1, calls.size)
        val args = calls[0].arguments
        assertEquals(JsonPrimitive("foo"), args["name"])
        assertEquals(JsonPrimitive(42L), args["n"])
        assertEquals(JsonPrimitive(true), args["active"])
        assertEquals(JsonPrimitive(3.14), args["ratio"])
    }

    @Test
    fun parseToolCallWithNestedObjectAndArray() {
        val text = "<|tool_call>call:fn{point:{x:1,y:2},tags:[<|\"|>a<|\"|>,<|\"|>b<|\"|>]}<tool_call|>"
        val calls = Gemma4ChatTemplate().parseToolCalls(text)
        assertEquals(1, calls.size)
        val args = calls[0].arguments
        val point = args["point"] as JsonObject
        assertEquals(JsonPrimitive(1L), point["x"])
        assertEquals(JsonPrimitive(2L), point["y"])
        val tags = args["tags"] as JsonArray
        assertEquals(JsonPrimitive("a"), tags[0])
        assertEquals(JsonPrimitive("b"), tags[1])
    }

    @Test
    fun parseMultipleToolCalls() {
        val text = "<|tool_call>call:search{q:<|\"|>a<|\"|>}<tool_call|> then " +
            "<|tool_call>call:fetch{url:<|\"|>b<|\"|>}<tool_call|>"
        val calls = Gemma4ChatTemplate().parseToolCalls(text)
        assertEquals(2, calls.size)
        assertEquals("search", calls[0].name)
        assertEquals("fetch", calls[1].name)
    }

    @Test
    fun parsePlainTextReturnsEmpty() {
        val calls = Gemma4ChatTemplate().parseToolCalls("This is a normal response.")
        assertTrue(calls.isEmpty())
    }

    @Test
    fun parseMalformedToolCallReturnsEmpty() {
        // Missing closing brace inside the call body
        val text = "<|tool_call>call:fn{key:<|\"|>val<|\"|><tool_call|>"
        val calls = Gemma4ChatTemplate().parseToolCalls(text)
        assertTrue(calls.isEmpty(), "malformed body should yield no calls, not throw")
    }

    @Test
    fun containsToolCallDetectsHfFormat() {
        val template = Gemma4ChatTemplate()
        assertTrue(template.containsToolCall("<|tool_call>call:test{}<tool_call|>"))
        assertFalse(template.containsToolCall("Hello, world!"))
    }

    // ---- Tool-response emission (us → model on continuation) ----

    @Test
    fun toolResponseEmitsResponseBlockOnAssistantContinuation() {
        // History: assistant called calculator → tool returned 42 →
        // we re-prompt to let the model use the result.
        val callId = "call_1"
        val messages = listOf(
            ChatMessage(ChatRole.USER, "What is 17*23?"),
            ChatMessage(
                ChatRole.ASSISTANT,
                content = "",
                toolCalls = listOf(ToolCall(id = callId, name = "calculator", arguments = buildJsonObject {
                    put("expression", "17*23")
                }))
            ),
            ChatMessage(ChatRole.TOOL, content = "391", toolCallId = callId)
        )
        val result = Gemma4ChatTemplate().apply(messages, addGenerationPrompt = true)

        // tool_call goes through with HF format
        assertContains(result, "<|tool_call>call:calculator{expression:<|\"|>17*23<|\"|>}<tool_call|>")

        // Tool response uses HF response:NAME{...} body
        assertContains(result, "<|tool_response>response:calculator{value:<|\"|>391<|\"|>}<tool_response|>")

        // Tool response is NOT wrapped in a separate user turn — it's a
        // continuation of the assistant model turn.
        val toolResponseIdx = result.indexOf("<|tool_response>")
        val precedingFragment = result.substring(0, toolResponseIdx)
        // The most recent <|turn> opener before the tool_response should be `model`.
        val lastTurnOpenerIdx = precedingFragment.lastIndexOf("<|turn>")
        val openerLine = precedingFragment.substring(lastTurnOpenerIdx).lines().first()
        assertTrue(openerLine.startsWith("<|turn>model"),
            "tool_response must continue the assistant model turn, not open a new user turn (got '$openerLine')")
    }

    @Test
    fun toolResponseAcceptsJsonObjectContent() {
        val callId = "x"
        val messages = listOf(
            ChatMessage(ChatRole.USER, "q"),
            ChatMessage(
                ChatRole.ASSISTANT,
                content = "",
                toolCalls = listOf(ToolCall(id = callId, name = "fn", arguments = buildJsonObject {}))
            ),
            ChatMessage(ChatRole.TOOL, content = """{"result":391,"unit":"none"}""", toolCallId = callId)
        )
        val result = Gemma4ChatTemplate().apply(messages, addGenerationPrompt = false)

        // Object content is rendered as `response:NAME{key:value,…}` with
        // bare keys (alphabetical order via dictsort: result, unit).
        assertContains(result, "<|tool_response>response:fn{result:391,unit:<|\"|>none<|\"|>}<tool_response|>")
    }

    // ---- Thinking mode (HF channel-based) ----

    @Test
    fun enableThinkingPrependsThinkTokenInSystemTurn() {
        val result = Gemma4ChatTemplate(enableThinking = true).apply(
            listOf(userMsg),
            tools = listOf(sampleTool)
        )
        // The single (unpaired) <|think|> token signals "thinking enabled"
        // at the top of the first system turn.
        assertContains(result, "<|turn>system\n<|think|>\n")
    }

    @Test
    fun thinkingDisabledByDefault() {
        val result = Gemma4ChatTemplate().apply(listOf(userMsg), tools = listOf(sampleTool))
        assertFalse(result.contains("<|think|>"),
            "<|think|> must not be emitted when enableThinking=false (default)")
    }

    @Test
    fun parseThinkingFromChannelBlock() {
        val text = "Sure.\n<|channel>thought\nLet me work this out.<channel|>\nAnswer: 42."
        val blocks = Gemma4ChatTemplate().parseThinkingBlocks(text)
        assertEquals(1, blocks.size)
        assertEquals("Let me work this out.", blocks[0])
    }

    @Test
    fun parseMultipleThinkingChannels() {
        val text = "<|channel>thought\nfirst<channel|> mid <|channel>thought\nsecond<channel|>"
        val blocks = Gemma4ChatTemplate().parseThinkingBlocks(text)
        assertEquals(listOf("first", "second"), blocks)
    }

    @Test
    fun parseChannelOnlyPicksUpThoughtChannel() {
        // HF's strip_thinking only treats `<|channel>thought` as reasoning.
        // A hypothetical other channel type (e.g. `<|channel>analysis`) is
        // surfaced via different mechanics; our parser ignores non-thought
        // channels.
        val text = "<|channel>analysis\nshould not surface<channel|>"
        val blocks = Gemma4ChatTemplate().parseThinkingBlocks(text)
        assertTrue(blocks.isEmpty())
    }

    @Test
    fun stripThinkingRemovesChannelBlock() {
        val text = "Here.\n<|channel>thought\nhidden reasoning<channel|>\nResult: 7."
        val stripped = Gemma4ChatTemplate().stripThinking(text)
        assertFalse(stripped.contains("<|channel>"))
        assertFalse(stripped.contains("<channel|>"))
        assertFalse(stripped.contains("hidden reasoning"))
        assertContains(stripped, "Here.")
        assertContains(stripped, "Result: 7.")
    }

    @Test
    fun stripThinkingIdempotentWhenNoBlock() {
        val text = "Plain output with no channel at all."
        assertEquals(text, Gemma4ChatTemplate().stripThinking(text))
    }

    @Test
    fun stripThinkingDropsUnterminatedBlock() {
        // Generation truncation may cut a channel block mid-content.
        // The partial reasoning must NOT leak into history.
        val text = "Looking at it:\n<|channel>thought\nI should reason about"
        val stripped = Gemma4ChatTemplate().stripThinking(text)
        assertFalse(stripped.contains("I should reason about"))
        assertFalse(stripped.contains("<|channel>"))
        assertContains(stripped, "Looking at it:")
    }

    // ---- Aggregate / golden ----

    @Test
    fun allSpecialMarkersAppearLiterallyAndBalance() {
        val callId = "abc"
        val messages = listOf(
            ChatMessage(ChatRole.SYSTEM, "sys"),
            ChatMessage(ChatRole.USER, "q"),
            ChatMessage(
                ChatRole.ASSISTANT,
                content = "",
                toolCalls = listOf(ToolCall(id = callId, name = "calculator", arguments = buildJsonObject { put("expression", "1+1") }))
            ),
            ChatMessage(ChatRole.TOOL, content = "2", toolCallId = callId)
        )
        val result = Gemma4ChatTemplate().apply(messages, tools = listOf(sampleTool))

        // BOS prepended
        assertTrue(result.startsWith("<bos>"))

        // All marker families present literally
        assertContains(result, "<|turn>")
        assertContains(result, "<turn|>")
        assertContains(result, "<|tool>")
        assertContains(result, "<tool|>")
        assertContains(result, "<|tool_call>")
        assertContains(result, "<tool_call|>")
        assertContains(result, "<|tool_response>")
        assertContains(result, "<tool_response|>")

        // Marker pairs balance (modulo the trailing generation prompt for <|turn>)
        assertEquals(countOf(result, "<|turn>"), countOf(result, "<turn|>") + 1)
        assertEquals(countOf(result, "<|tool>"), countOf(result, "<tool|>"))
        assertEquals(countOf(result, "<|tool_call>"), countOf(result, "<tool_call|>"))
        assertEquals(countOf(result, "<|tool_response>"), countOf(result, "<tool_response|>"))
    }

    @Test
    fun fullConversationGoldenTest() {
        val messages = listOf(
            ChatMessage(ChatRole.SYSTEM, "You are helpful."),
            ChatMessage(ChatRole.USER, "What is 2+2?"),
            ChatMessage(ChatRole.ASSISTANT, "4"),
            ChatMessage(ChatRole.USER, "Thanks!")
        )
        val result = Gemma4ChatTemplate().apply(messages)

        val expected = "<bos>" +
            "<|turn>system\nYou are helpful.<turn|>\n" +
            "<|turn>user\nWhat is 2+2?<turn|>\n" +
            "<|turn>model\n4<turn|>\n" +
            "<|turn>user\nThanks!<turn|>\n" +
            "<|turn>model\n"

        assertEquals(expected, result)
    }

    private fun countOf(haystack: String, needle: String): Int {
        var count = 0
        var idx = 0
        while (true) {
            val next = haystack.indexOf(needle, idx)
            if (next < 0) return count
            count++
            idx = next + needle.length
        }
    }
}
