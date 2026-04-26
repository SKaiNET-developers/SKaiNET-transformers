package sk.ainet.apps.kllama.chat

import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonObject

class ChatTemplateTest {

    private val systemMsg = ChatMessage(ChatRole.SYSTEM, "You are helpful.")
    private val userMsg = ChatMessage(ChatRole.USER, "Hello!")

    // --- Llama 3 Template Tests ---

    @Test
    fun llama3BasicFormat() {
        val template = Llama3ChatTemplate()
        val result = template.apply(listOf(systemMsg, userMsg))

        assertContains(result, "<|begin_of_text|>")
        assertContains(result, "<|start_header_id|>system<|end_header_id|>")
        assertContains(result, "You are helpful.")
        assertContains(result, "<|eot_id|>")
        assertContains(result, "<|start_header_id|>user<|end_header_id|>")
        assertContains(result, "Hello!")
        // Should end with assistant generation prompt
        assertTrue(result.endsWith("<|start_header_id|>assistant<|end_header_id|>\n\n"))
    }

    @Test
    fun llama3NoGenerationPrompt() {
        val template = Llama3ChatTemplate()
        val result = template.apply(listOf(userMsg), addGenerationPrompt = false)

        assertTrue(!result.endsWith("<|start_header_id|>assistant<|end_header_id|>\n\n"))
        assertTrue(result.endsWith("<|eot_id|>"))
    }

    @Test
    fun llama3WithTools() {
        val template = Llama3ChatTemplate()
        val tool = ToolDefinition(
            name = "calculator",
            description = "Evaluate math",
            parameters = buildJsonObject {
                put("type", "object")
                putJsonObject("properties") {
                    putJsonObject("expression") {
                        put("type", "string")
                    }
                }
            }
        )

        val result = template.apply(listOf(userMsg), tools = listOf(tool))

        assertContains(result, "calculator")
        assertContains(result, "Evaluate math")
        assertContains(result, "\"name\"")
    }

    @Test
    fun llama3ToolRole() {
        val template = Llama3ChatTemplate()
        val toolMsg = ChatMessage(ChatRole.TOOL, "42", toolCallId = "call_0")
        val result = template.apply(listOf(toolMsg), addGenerationPrompt = false)

        assertContains(result, "<|start_header_id|>tool<|end_header_id|>")
        assertContains(result, "42")
    }

    // --- Llama 3 Tool-Format Variants ---

    @Test
    fun llama3JsonFormatInstructsBareJsonResponse() {
        val template = Llama3ChatTemplate(Llama3ToolFormat.JSON)
        val tool = ToolDefinition(
            name = "list_files",
            description = "List files",
            parameters = buildJsonObject { put("type", "object") }
        )
        val result = template.apply(listOf(userMsg), tools = listOf(tool))

        // Tool definition is embedded
        assertContains(result, "list_files")
        // System prompt instructs the bare-JSON format using "parameters"
        assertContains(result, "\"name\": <function-name>")
        assertContains(result, "\"parameters\": <arguments-object>")
        // Must NOT instruct the legacy <function=...> tag format
        assertTrue(!result.contains("<function="), "JSON format must not mention <function=...> tag")
    }

    @Test
    fun llama3FunctionTagFormatInstructsTaggedResponse() {
        val template = Llama3ChatTemplate(Llama3ToolFormat.FUNCTION_TAG)
        val tool = ToolDefinition(
            name = "list_files",
            description = "List files",
            parameters = buildJsonObject { put("type", "object") }
        )
        val result = template.apply(listOf(userMsg), tools = listOf(tool))

        assertContains(result, "list_files")
        // System prompt instructs the <function=...>...</function> format
        assertContains(result, "<function=function_name>")
        assertContains(result, "</function>")
    }

    @Test
    fun llama3MergesUserSystemPromptWithToolBlock() {
        val template = Llama3ChatTemplate()
        val tool = ToolDefinition(
            name = "calculator",
            description = "Math",
            parameters = buildJsonObject { put("type", "object") }
        )
        val userSystem = ChatMessage(ChatRole.SYSTEM, "Be terse.")
        val result = template.apply(listOf(userSystem, userMsg), tools = listOf(tool))

        // Both the tool block AND the user system message must be preserved
        assertContains(result, "calculator")
        assertContains(result, "Be terse.")
        // User system message comes after the tool block in the merged single
        // system turn (so the tool instructions take precedence visually).
        val toolIdx = result.indexOf("calculator")
        val userIdx = result.indexOf("Be terse.")
        assertTrue(toolIdx in 0 until userIdx, "tool block must precede user system text")
    }

    @Test
    fun llama3DefaultFormatIsJson() {
        // Regression guard: changing the default format silently would shift
        // production behavior across every Llama 3 deployment.
        val templateDefault = Llama3ChatTemplate()
        val templateExplicit = Llama3ChatTemplate(Llama3ToolFormat.JSON)
        val tool = ToolDefinition(
            name = "x",
            description = "y",
            parameters = buildJsonObject { put("type", "object") }
        )
        val msgs = listOf(userMsg)
        assertEquals(templateExplicit.apply(msgs, listOf(tool)), templateDefault.apply(msgs, listOf(tool)))
    }

    // --- ChatML Template Tests ---

    @Test
    fun chatMLBasicFormat() {
        val template = ChatMLTemplate()
        val result = template.apply(listOf(systemMsg, userMsg))

        assertContains(result, "<|im_start|>system")
        assertContains(result, "You are helpful.")
        assertContains(result, "<|im_end|>")
        assertContains(result, "<|im_start|>user")
        assertContains(result, "Hello!")
        // Should end with assistant generation prompt
        assertTrue(result.endsWith("<|im_start|>assistant\n"))
    }

    @Test
    fun chatMLNoGenerationPrompt() {
        val template = ChatMLTemplate()
        val result = template.apply(listOf(userMsg), addGenerationPrompt = false)

        assertTrue(!result.endsWith("<|im_start|>assistant\n"))
        assertTrue(result.endsWith("<|im_end|>\n"))
    }

    @Test
    fun chatMLWithTools() {
        val template = ChatMLTemplate()
        val tool = ToolDefinition(
            name = "search",
            description = "Search the web",
            parameters = buildJsonObject { put("type", "object") }
        )

        val result = template.apply(listOf(userMsg), tools = listOf(tool))

        assertContains(result, "search")
        assertContains(result, "Search the web")
        assertContains(result, "<tool_call>")
    }

    @Test
    fun chatMLToolRole() {
        val template = ChatMLTemplate()
        val toolMsg = ChatMessage(ChatRole.TOOL, "result data", toolCallId = "call_0")
        val result = template.apply(listOf(toolMsg), addGenerationPrompt = false)

        assertContains(result, "<|im_start|>tool")
        assertContains(result, "result data")
    }

    // --- Default parseToolCalls behavior ---

    @Test
    fun llama3DefaultParseToolCalls() {
        val template = Llama3ChatTemplate()
        val text = """{"name": "calculator", "arguments": {"expression": "1+1"}}"""
        val calls = template.parseToolCalls(text)
        assertEquals(1, calls.size)
        assertEquals("calculator", calls[0].name)
    }

    @Test
    fun chatMLDefaultParseToolCalls() {
        val template = ChatMLTemplate()
        val text = "<tool_call>{\"name\": \"search\", \"arguments\": {\"query\": \"test\"}}</tool_call>"
        val calls = template.parseToolCalls(text)
        assertEquals(1, calls.size)
        assertEquals("search", calls[0].name)
    }
}
