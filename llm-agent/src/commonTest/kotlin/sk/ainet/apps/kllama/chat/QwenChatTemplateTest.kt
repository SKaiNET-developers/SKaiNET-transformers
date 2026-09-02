package sk.ainet.apps.kllama.chat

import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.assertFalse
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

        // No injected persona — the official template only carries the caller's own
        // system message (if any) ahead of the # Tools section.
        assertFalse(result.contains("You are Qwen, created by Alibaba Cloud."))
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
    fun toolResultsRenderAsUserToolResponseTurns() {
        // Official Qwen3 template: tool results are USER turns wrapped in <tool_response>,
        // and consecutive results merge into ONE user turn. The literal `tool` role string
        // never appears in Qwen's training data.
        val template = QwenChatTemplate()
        val result = template.apply(
            listOf(
                ChatMessage(ChatRole.TOOL, "42", toolCallId = "call_0"),
                ChatMessage(ChatRole.TOOL, "sunny", toolCallId = "call_1"),
            ),
            addGenerationPrompt = false,
        )

        assertContains(result, "<|im_start|>user\n<tool_response>\n42\n</tool_response>\n<tool_response>\nsunny\n</tool_response><|im_end|>")
        assertFalse(result.contains("<|im_start|>tool"))
    }

    @Test
    fun assistantToolCallsReplayAsToolCallBlocks() {
        val template = QwenChatTemplate()
        val call = ToolCall(id = "call_0", name = "calculator",
            arguments = buildJsonObject { put("expression", "2 + 3") })
        val result = template.apply(
            listOf(ChatMessage(ChatRole.ASSISTANT, "", toolCalls = listOf(call))),
            addGenerationPrompt = false,
        )
        assertContains(
            result,
            "<tool_call>\n{\"name\": \"calculator\", \"arguments\": {\"expression\":\"2 + 3\"}}\n</tool_call>",
        )
    }

    @Test
    fun rawToolCallXmlInContentIsNotRenderedTwice() {
        // AgentLoop persists the model's raw text (incl. the <tool_call> XML) as content
        // alongside the structured toolCalls — replay must render the call exactly once.
        val template = QwenChatTemplate()
        val call = ToolCall(id = "call_0", name = "calculator",
            arguments = buildJsonObject { put("expression", "2 + 3") })
        val raw = "<tool_call>\n{\"name\": \"calculator\", \"arguments\": {\"expression\": \"2 + 3\"}}\n</tool_call>"
        val result = template.apply(
            listOf(ChatMessage(ChatRole.ASSISTANT, raw, toolCalls = listOf(call))),
            addGenerationPrompt = false,
        )
        assertEquals(1, Regex("<tool_call>").findAll(result).count())
    }

    @Test
    fun thinkingBlocksAreParsedAndStripped() {
        val template = QwenChatTemplate()
        val text = "<think>\nLet me reason.\n</think>\n\nThe answer is 4."
        assertEquals(listOf("Let me reason."), template.parseThinkingBlocks(text))
        assertEquals("The answer is 4.", template.stripThinking(text))
        // Unterminated block (budget ran out mid-thought) must not leak as the answer.
        assertEquals("", template.stripThinking("<think>\nstill going"))
    }

    @Test
    fun nonThinkingModePrefillsEmptyThinkBlock() {
        val template = QwenChatTemplate(enableThinking = false)
        val result = template.apply(listOf(userMsg))
        assertContains(result, "<|im_start|>assistant\n<think>\n\n</think>\n\n")
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
