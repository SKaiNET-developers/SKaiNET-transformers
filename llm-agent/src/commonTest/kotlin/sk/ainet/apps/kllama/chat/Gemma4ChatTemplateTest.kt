package sk.ainet.apps.kllama.chat

import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.assertFalse
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonObject

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
                }
            }
        }
    )

    @Test
    fun basicFormat() {
        val template = Gemma4ChatTemplate()
        val result = template.apply(listOf(systemMsg, userMsg))

        assertContains(result, "<|turn>system")
        assertContains(result, "You are helpful.")
        assertContains(result, "<turn|>")
        assertContains(result, "<|turn>user")
        assertContains(result, "Hello!")
        assertTrue(result.endsWith("<|turn>model\n"))
    }

    @Test
    fun systemRoleUsesSystemTurn() {
        val template = Gemma4ChatTemplate()
        val result = template.apply(listOf(systemMsg), addGenerationPrompt = false)

        // Gemma 4 has native system role (unlike Gemma 2/3 which maps to user)
        assertContains(result, "<|turn>system\n")
        assertContains(result, "You are helpful.")
        assertContains(result, "<turn|>")
        assertFalse(result.contains("<|turn>user"))
    }

    @Test
    fun assistantRoleUsesModelTurn() {
        val template = Gemma4ChatTemplate()
        val assistantMsg = ChatMessage(ChatRole.ASSISTANT, "Hi there!")
        val result = template.apply(listOf(assistantMsg), addGenerationPrompt = false)

        assertContains(result, "<|turn>model")
        assertContains(result, "Hi there!")
        assertContains(result, "<turn|>")
    }

    @Test
    fun noGenerationPrompt() {
        val template = Gemma4ChatTemplate()
        val result = template.apply(listOf(userMsg), addGenerationPrompt = false)

        assertFalse(result.endsWith("<|turn>model\n"))
        assertTrue(result.endsWith("<turn|>\n"))
    }

    @Test
    fun toolDefinitionsAsToolBlock() {
        val template = Gemma4ChatTemplate()
        val result = template.apply(listOf(userMsg), tools = listOf(sampleTool))

        assertContains(result, "<|tool>")
        assertContains(result, "<tool|>")
        assertContains(result, "\"name\":\"calculator\"")
        assertContains(result, "\"description\":\"Evaluate math expressions\"")
        // Tool definitions should be in a system turn
        assertContains(result, "<|turn>system")
    }

    @Test
    fun toolResponseFormatting() {
        val template = Gemma4ChatTemplate()
        val toolMsg = ChatMessage(ChatRole.TOOL, "42", toolCallId = "calculator")
        val result = template.apply(listOf(toolMsg), addGenerationPrompt = false)

        assertContains(result, "<|turn>user")
        assertContains(result, "<|tool_response>")
        assertContains(result, "<tool_response|>")
        assertContains(result, "\"name\":\"calculator\"")
        assertContains(result, "\"result\":\"42\"")
    }

    @Test
    fun parseToolCallWithDelimiters() {
        val template = Gemma4ChatTemplate()
        val text = """Let me calculate that. <|tool_call>{"name": "calculator", "args": {"expression": "2 + 3"}}<tool_call|>"""

        val calls = template.parseToolCalls(text)
        assertEquals(1, calls.size)
        assertEquals("calculator", calls[0].name)
        assertEquals(
            "2 + 3",
            calls[0].arguments["expression"]?.toString()?.trim('"')
        )
    }

    @Test
    fun parseToolCallMissingArgs() {
        val template = Gemma4ChatTemplate()
        val text = """<|tool_call>{"name": "list_files"}<tool_call|>"""

        val calls = template.parseToolCalls(text)
        assertEquals(1, calls.size)
        assertEquals("list_files", calls[0].name)
        assertTrue(calls[0].arguments.isEmpty())
    }

    @Test
    fun parseMultipleToolCalls() {
        val template = Gemma4ChatTemplate()
        val text = """<|tool_call>{"name": "search", "args": {"q": "a"}}<tool_call|> then <|tool_call>{"name": "fetch", "args": {"url": "b"}}<tool_call|>"""

        val calls = template.parseToolCalls(text)
        assertEquals(2, calls.size)
        assertEquals("search", calls[0].name)
        assertEquals("fetch", calls[1].name)
    }

    @Test
    fun parsePlainTextReturnsEmpty() {
        val template = Gemma4ChatTemplate()
        val calls = template.parseToolCalls("This is a normal response.")
        assertTrue(calls.isEmpty())
    }

    @Test
    fun containsToolCallDetectsGemma4Format() {
        val template = Gemma4ChatTemplate()
        assertTrue(template.containsToolCall("""<|tool_call>{"name": "test"}<tool_call|>"""))
        assertFalse(template.containsToolCall("Hello, world!"))
    }

    @Test
    fun containsToolCallDetectsLegacyFormat() {
        val template = Gemma4ChatTemplate()
        assertTrue(template.containsToolCall("""{"functionCall": {"name": "test"}}"""))
    }

    @Test
    fun allSpecialMarkersAppearLiterally() {
        val template = Gemma4ChatTemplate()
        val messages = listOf(
            ChatMessage(ChatRole.SYSTEM, "sys"),
            ChatMessage(ChatRole.USER, "q"),
            ChatMessage(ChatRole.ASSISTANT, "a"),
            ChatMessage(ChatRole.TOOL, "42", toolCallId = "calculator")
        )
        val result = template.apply(messages, tools = listOf(sampleTool))

        // Every marker the Gemma 4 grammar relies on must appear literally at
        // least once — no HTML escaping, no unicode corruption, no accidental
        // splitting into multiple tokens that look similar but aren't.
        assertContains(result, "<|turn>")
        assertContains(result, "<turn|>")
        assertContains(result, "<|tool>")
        assertContains(result, "<tool|>")
        assertContains(result, "<|tool_response>")
        assertContains(result, "<tool_response|>")

        // Opener/closer must balance for each marker family.
        assertEquals(countOf(result, "<|turn>"), countOf(result, "<turn|>") + 1,
            "<|turn> count should exceed <turn|> by exactly one (the trailing generation prompt)")
        assertEquals(countOf(result, "<|tool>"), countOf(result, "<tool|>"),
            "<|tool> and <tool|> must balance")
        assertEquals(countOf(result, "<|tool_response>"), countOf(result, "<tool_response|>"),
            "<|tool_response> and <tool_response|> must balance")

        // The prompt must not leak <|tool_call>...<tool_call|> — the template
        // never emits those; they only appear in model output.
        assertFalse(result.contains("<|tool_call>"),
            "<|tool_call> must not appear in a rendered prompt; it belongs only in model output")
        assertFalse(result.contains("<tool_call|>"),
            "<tool_call|> must not appear in a rendered prompt; it belongs only in model output")
    }

    @Test
    fun toolCallRoundTripPreservesArguments() {
        val template = Gemma4ChatTemplate()

        // Simulate what the model would emit: an assistant turn containing
        // a tool_call block with JSON arguments.
        val modelOutput = "Let me run that.\n" +
            "<|tool_call>{\"name\":\"calculator\",\"args\":{\"expression\":\"3+4\",\"precision\":2}}<tool_call|>"

        val parsed = template.parseToolCalls(modelOutput)
        assertEquals(1, parsed.size)
        assertEquals("calculator", parsed[0].name)
        assertEquals(
            "3+4",
            parsed[0].arguments["expression"]?.toString()?.trim('"')
        )
        assertEquals("2", parsed[0].arguments["precision"]?.toString())
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

    @Test
    fun fullConversationGoldenTest() {
        val template = Gemma4ChatTemplate()
        val messages = listOf(
            ChatMessage(ChatRole.SYSTEM, "You are helpful."),
            ChatMessage(ChatRole.USER, "What is 2+2?"),
            ChatMessage(ChatRole.ASSISTANT, "4"),
            ChatMessage(ChatRole.USER, "Thanks!")
        )
        val result = template.apply(messages)

        val expected = "<|turn>system\nYou are helpful.<turn|>\n" +
            "<|turn>user\nWhat is 2+2?<turn|>\n" +
            "<|turn>model\n4<turn|>\n" +
            "<|turn>user\nThanks!<turn|>\n" +
            "<|turn>model\n"

        assertEquals(expected, result)
    }
}
