package sk.ainet.apps.kllama.chat

import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonObject

class QwenChatTemplateTest {

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
        val template = QwenChatTemplate()
        val result = template.apply(listOf(systemMsg, userMsg))

        assertContains(result, "<|im_start|>system")
        assertContains(result, "You are helpful.")
        assertContains(result, "<|im_end|>")
        assertContains(result, "<|im_start|>user")
        assertContains(result, "Hello!")
        assertTrue(result.endsWith("<|im_start|>assistant\n"))
    }

    @Test
    fun noGenerationPrompt() {
        val template = QwenChatTemplate()
        val result = template.apply(listOf(userMsg), addGenerationPrompt = false)

        assertTrue(!result.endsWith("<|im_start|>assistant\n"))
        assertTrue(result.endsWith("<|im_end|>\n"))
    }

    @Test
    fun toolDefinitionsInQwenFormat() {
        val template = QwenChatTemplate()
        val result = template.apply(listOf(userMsg), tools = listOf(sampleTool))

        // Qwen preamble
        assertContains(result, "You are Qwen, created by Alibaba Cloud.")
        // Tool definitions in function schema format
        assertContains(result, "\"type\":\"function\"")
        assertContains(result, "\"function\":")
        assertContains(result, "\"name\":\"calculator\"")
        assertContains(result, "\"description\":\"Evaluate math expressions\"")
        // Wrapped in <tools> tags
        assertContains(result, "<tools>")
        assertContains(result, "</tools>")
        // Instructions for tool call format
        assertContains(result, "<tool_call>")
        assertContains(result, "</tool_call>")
    }

    @Test
    fun toolRole() {
        val template = QwenChatTemplate()
        val toolMsg = ChatMessage(ChatRole.TOOL, "42", toolCallId = "call_0")
        val result = template.apply(listOf(toolMsg), addGenerationPrompt = false)

        assertContains(result, "<|im_start|>tool")
        assertContains(result, "42")
        assertContains(result, "<|im_end|>")
    }

    @Test
    fun parseToolCallsDelegatesToDefault() {
        val template = QwenChatTemplate()
        val text = """
            <tool_call>
            {"name": "calculator", "arguments": {"expression": "2 + 3"}}
            </tool_call>
        """.trimIndent()

        val calls = template.parseToolCalls(text)
        assertEquals(1, calls.size)
        assertEquals("calculator", calls[0].name)
    }

    @Test
    fun containsToolCallDelegatesToDefault() {
        val template = QwenChatTemplate()
        assertTrue(template.containsToolCall("<tool_call>{}</tool_call>"))
        assertTrue(!template.containsToolCall("Just a normal response."))
    }

    @Test
    fun multipleToolDefinitions() {
        val template = QwenChatTemplate()
        val tools = listOf(
            sampleTool,
            ToolDefinition(
                name = "search",
                description = "Search the web",
                parameters = buildJsonObject { put("type", "object") }
            )
        )
        val result = template.apply(listOf(userMsg), tools = tools)

        assertContains(result, "\"name\":\"calculator\"")
        assertContains(result, "\"name\":\"search\"")
    }
}
