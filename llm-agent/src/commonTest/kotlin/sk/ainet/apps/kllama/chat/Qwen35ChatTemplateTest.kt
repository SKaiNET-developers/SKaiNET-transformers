package sk.ainet.apps.kllama.chat

import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonObject

class Qwen35ChatTemplateTest {

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
        val template = Qwen35ChatTemplate()
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
        val template = Qwen35ChatTemplate()
        val result = template.apply(listOf(userMsg), addGenerationPrompt = false)

        assertTrue(!result.endsWith("<|im_start|>assistant\n"))
        assertTrue(result.endsWith("<|im_end|>\n"))
    }

    @Test
    fun toolDefinitionsInQwen35Format() {
        val template = Qwen35ChatTemplate()
        val result = template.apply(listOf(userMsg), tools = listOf(sampleTool))

        // Qwen preamble
        assertContains(result, "You are Qwen, created by Alibaba Cloud.")
        // Tool definitions in function schema format
        assertContains(result, "\"type\":\"function\"")
        assertContains(result, "\"name\":\"calculator\"")
        // Wrapped in <tools> tags
        assertContains(result, "<tools>")
        assertContains(result, "</tools>")
        // Qwen3.5 canonical tool call instruction (not JSON)
        assertContains(result, "<function=function_name>")
        assertContains(result, "<parameter=param_name>value</parameter>")
    }

    @Test
    fun toolResponseWrapping() {
        val template = Qwen35ChatTemplate()
        val toolMsg = ChatMessage(ChatRole.TOOL, "42", toolCallId = "call_0")
        val result = template.apply(listOf(toolMsg), addGenerationPrompt = false)

        assertContains(result, "<|im_start|>tool")
        assertContains(result, "<tool_response>")
        assertContains(result, "42")
        assertContains(result, "</tool_response>")
        assertContains(result, "<|im_end|>")
    }

    @Test
    fun parseToolCallsDelegatesToQwen35Strategy() {
        val template = Qwen35ChatTemplate()
        val text = """
            <tool_call>
            <function=calculator>
            <parameter=expression>2 + 3</parameter>
            </function>
            </tool_call>
        """.trimIndent()

        val calls = template.parseToolCalls(text)
        assertEquals(1, calls.size)
        assertEquals("calculator", calls[0].name)
        assertEquals("2 + 3", calls[0].arguments["expression"]?.toString()?.trim('"'))
    }

    @Test
    fun containsToolCallDetectsQwen35Format() {
        val template = Qwen35ChatTemplate()
        assertTrue(template.containsToolCall("<tool_call><function=test></function></tool_call>"))
        assertTrue(!template.containsToolCall("Just a normal response."))
    }

    @Test
    fun containsToolCallDoesNotMatchHermesFormat() {
        val template = Qwen35ChatTemplate()
        // Hermes format has JSON, not <function=...>
        assertTrue(!template.containsToolCall("""<tool_call>{"name":"calc"}</tool_call>"""))
    }

    @Test
    fun multipleToolDefinitions() {
        val template = Qwen35ChatTemplate()
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

    @Test
    fun noToolsNoSystemInjection() {
        val template = Qwen35ChatTemplate()
        val result = template.apply(listOf(userMsg))

        // Should not contain the tools system message
        assertTrue(!result.contains("# Tools"))
        assertTrue(!result.contains("<tools>"))
    }
}
