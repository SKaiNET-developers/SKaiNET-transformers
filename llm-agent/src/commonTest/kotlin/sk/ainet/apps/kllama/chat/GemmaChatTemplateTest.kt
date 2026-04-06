package sk.ainet.apps.kllama.chat

import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.assertFalse
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonObject

class GemmaChatTemplateTest {

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
        val template = GemmaChatTemplate()
        val result = template.apply(listOf(systemMsg, userMsg))

        assertContains(result, "<start_of_turn>user")
        assertContains(result, "You are helpful.")
        assertContains(result, "<end_of_turn>")
        assertContains(result, "Hello!")
        assertTrue(result.endsWith("<start_of_turn>model\n"))
    }

    @Test
    fun assistantRoleUsesModelTurn() {
        val template = GemmaChatTemplate()
        val assistantMsg = ChatMessage(ChatRole.ASSISTANT, "Hi there!")
        val result = template.apply(listOf(assistantMsg), addGenerationPrompt = false)

        assertContains(result, "<start_of_turn>model")
        assertContains(result, "Hi there!")
        assertContains(result, "<end_of_turn>")
    }

    @Test
    fun noGenerationPrompt() {
        val template = GemmaChatTemplate()
        val result = template.apply(listOf(userMsg), addGenerationPrompt = false)

        assertTrue(!result.endsWith("<start_of_turn>model\n"))
        assertTrue(result.endsWith("<end_of_turn>\n"))
    }

    @Test
    fun toolDefinitionsAsFunctionDeclarations() {
        val template = GemmaChatTemplate()
        val result = template.apply(listOf(userMsg), tools = listOf(sampleTool))

        assertContains(result, "\"function_declarations\"")
        assertContains(result, "\"name\":\"calculator\"")
        assertContains(result, "\"description\":\"Evaluate math expressions\"")
    }

    @Test
    fun toolRoleFormattedAsFunctionResponse() {
        val template = GemmaChatTemplate()
        val toolMsg = ChatMessage(ChatRole.TOOL, "42", toolCallId = "calculator")
        val result = template.apply(listOf(toolMsg), addGenerationPrompt = false)

        assertContains(result, "<start_of_turn>user")
        assertContains(result, "\"functionResponse\"")
        assertContains(result, "\"name\":\"calculator\"")
        assertContains(result, "\"result\":\"42\"")
    }

    // --- parseToolCalls ---

    @Test
    fun parseGemmaFunctionCall() {
        val template = GemmaChatTemplate()
        val text = """{"functionCall": {"name": "calculator", "args": {"expression": "2 + 3"}}}"""

        val calls = template.parseToolCalls(text)
        assertEquals(1, calls.size)
        assertEquals("calculator", calls[0].name)
        assertEquals(
            "2 + 3",
            calls[0].arguments["expression"]?.toString()?.trim('"')
        )
    }

    @Test
    fun parseGemmaFunctionCallMissingArgs() {
        val template = GemmaChatTemplate()
        val text = """{"functionCall": {"name": "list_files"}}"""

        val calls = template.parseToolCalls(text)
        assertEquals(1, calls.size)
        assertEquals("list_files", calls[0].name)
        assertTrue(calls[0].arguments.isEmpty())
    }

    @Test
    fun parseGemmaFunctionCallEmbeddedInText() {
        val template = GemmaChatTemplate()
        val text = """
            Let me look that up for you.
            {"functionCall": {"name": "search", "args": {"query": "weather"}}}
        """.trimIndent()

        val calls = template.parseToolCalls(text)
        assertEquals(1, calls.size)
        assertEquals("search", calls[0].name)
    }

    @Test
    fun parseGemmaPlainTextReturnsEmpty() {
        val template = GemmaChatTemplate()
        val calls = template.parseToolCalls("This is a normal response.")
        assertTrue(calls.isEmpty())
    }

    @Test
    fun containsToolCallDetectsGemmaFormat() {
        val template = GemmaChatTemplate()
        assertTrue(template.containsToolCall("""{"functionCall": {"name": "test"}}"""))
        assertFalse(template.containsToolCall("Hello, world!"))
    }
}
