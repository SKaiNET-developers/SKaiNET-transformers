package sk.ainet.apps.kllama.chat

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.add
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.putJsonObject

/**
 * Golden-string gate for [Qwen25ChatTemplate] (closes SKaiNET-transformers#409). Every expected
 * string below was produced by rendering `Qwen/Qwen2.5-0.5B-Instruct`'s real `chat_template`
 * (fetched from `tokenizer_config.json` on huggingface.co, 2026-09-23 — NOT the copy embedded in
 * the GGUF, which has a llama.cpp-conversion brace-escaping artifact: `{{"name": ...}}` doubled
 * braces instead of the source's real `{"name": ...}` single braces, confirmed by diffing the two)
 * through a real Jinja2 engine, with one intentional, documented adjustment: the `tojson` filter
 * uses compact separators (`{"k":"v"}`, no spaces) instead of `json.dumps`'s spaced default
 * (`{"k": "v"}`) — matching `kotlinx.serialization.json.Json`'s own default (`prettyPrint = false`,
 * genuinely compact, verified against the library source), which is what this class and the
 * already-shipped [QwenChatTemplate] both produce for tool JSON. The two are semantically
 * identical to a model (whitespace inside JSON never changes its meaning); this is the one
 * respect in which these golden strings are not literal HF byte output.
 */
class Qwen25ChatTemplateTest {

    private val userMsg = ChatMessage(ChatRole.USER, "schalte auf ard")

    private val switchChannelTool = ToolDefinition(
        name = "TV__SWITCH_CHANNEL",
        description = "Switch the TV to a channel.",
        parameters = buildJsonObject {
            put("type", "object")
            putJsonObject("properties") {
                putJsonObject("channel") {
                    put("type", "string")
                    put("description", "channel name")
                }
            }
            putJsonArray("required") { add(JsonPrimitive("channel")) }
        },
    )

    @Test
    fun noSystemMessage_getsTheDefaultPersona_toolsPresent() {
        val result = Qwen25ChatTemplate().apply(listOf(userMsg), tools = listOf(switchChannelTool))
        assertEquals(
            "<|im_start|>system\n" +
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n" +
                "# Tools\n\n" +
                "You may call one or more functions to assist with the user query.\n\n" +
                "You are provided with function signatures within <tools></tools> XML tags:\n" +
                "<tools>\n" +
                "{\"type\":\"function\",\"function\":{\"name\":\"TV__SWITCH_CHANNEL\",\"description\":\"Switch the TV to a channel.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"channel\":{\"type\":\"string\",\"description\":\"channel name\"}},\"required\":[\"channel\"]}}}\n" +
                "</tools>\n\n" +
                "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n" +
                "<tool_call>\n" +
                "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n" +
                "</tool_call><|im_end|>\n" +
                "<|im_start|>user\nschalte auf ard<|im_end|>\n" +
                "<|im_start|>assistant\n",
            result,
        )
    }

    @Test
    fun callerSystemMessage_replacesTheDefaultPersona_toolsPresent() {
        val messages = listOf(ChatMessage(ChatRole.SYSTEM, "You control a TV."), ChatMessage(ChatRole.USER, "mach lauter"))
        val result = Qwen25ChatTemplate().apply(messages, tools = listOf(switchChannelTool))
        assertEquals(
            "<|im_start|>system\n" +
                "You control a TV.\n\n" +
                "# Tools\n\n" +
                "You may call one or more functions to assist with the user query.\n\n" +
                "You are provided with function signatures within <tools></tools> XML tags:\n" +
                "<tools>\n" +
                "{\"type\":\"function\",\"function\":{\"name\":\"TV__SWITCH_CHANNEL\",\"description\":\"Switch the TV to a channel.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"channel\":{\"type\":\"string\",\"description\":\"channel name\"}},\"required\":[\"channel\"]}}}\n" +
                "</tools>\n\n" +
                "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n" +
                "<tool_call>\n" +
                "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n" +
                "</tool_call><|im_end|>\n" +
                "<|im_start|>user\nmach lauter<|im_end|>\n" +
                "<|im_start|>assistant\n",
            result,
        )
    }

    @Test
    fun noTools_noSystem_getsJustTheDefaultPersonaTurn() {
        val result = Qwen25ChatTemplate().apply(listOf(ChatMessage(ChatRole.USER, "hi")), tools = emptyList())
        assertEquals(
            "<|im_start|>system\n" +
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n" +
                "<|im_start|>user\nhi<|im_end|>\n" +
                "<|im_start|>assistant\n",
            result,
        )
    }

    @Test
    fun multiTurn_toolCallReplayAndToolResponse() {
        val messages = listOf(
            ChatMessage(ChatRole.USER, "schalte auf ard"),
            ChatMessage(
                ChatRole.ASSISTANT, "",
                toolCalls = listOf(ToolCall(id = "1", name = "TV__SWITCH_CHANNEL", arguments = buildJsonObject { put("channel", "ARD") })),
            ),
            ChatMessage(ChatRole.TOOL, "ok"),
            ChatMessage(ChatRole.USER, "danke"),
        )
        val result = Qwen25ChatTemplate().apply(messages, tools = listOf(switchChannelTool))
        assertEquals(
            "<|im_start|>system\n" +
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n" +
                "# Tools\n\n" +
                "You may call one or more functions to assist with the user query.\n\n" +
                "You are provided with function signatures within <tools></tools> XML tags:\n" +
                "<tools>\n" +
                "{\"type\":\"function\",\"function\":{\"name\":\"TV__SWITCH_CHANNEL\",\"description\":\"Switch the TV to a channel.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"channel\":{\"type\":\"string\",\"description\":\"channel name\"}},\"required\":[\"channel\"]}}}\n" +
                "</tools>\n\n" +
                "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n" +
                "<tool_call>\n" +
                "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n" +
                "</tool_call><|im_end|>\n" +
                "<|im_start|>user\nschalte auf ard<|im_end|>\n" +
                "<|im_start|>assistant\n" +
                "<tool_call>\n{\"name\": \"TV__SWITCH_CHANNEL\", \"arguments\": {\"channel\":\"ARD\"}}\n</tool_call><|im_end|>\n" +
                "<|im_start|>user\n<tool_response>\nok\n</tool_response><|im_end|>\n" +
                "<|im_start|>user\ndanke<|im_end|>\n" +
                "<|im_start|>assistant\n",
            result,
        )
    }

    @Test
    fun noAddGenerationPrompt_stopsAfterTheLastTurn() {
        val result = Qwen25ChatTemplate().apply(listOf(userMsg), addGenerationPrompt = false)
        assertEquals(
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n" +
                "<|im_start|>user\nschalte auf ard<|im_end|>\n",
            result,
        )
    }

    @Test
    fun noThinkingModeAnywhere() {
        val template = Qwen25ChatTemplate()
        // parseThinkingBlocks/stripThinking are no-ops: Qwen2.5 never had a thinking mode.
        assertEquals(emptyList(), template.parseThinkingBlocks("<think>reasoning</think>answer"))
        assertEquals("<think>reasoning</think>answer", template.stripThinking("<think>reasoning</think>answer"))
        // The generation prompt is never followed by any <think> prefill, unlike
        // QwenChatTemplate(enableThinking = false), which pre-fills an empty <think></think> block
        // Qwen2.5 has never seen in training.
        val result = template.apply(listOf(userMsg))
        assertEquals(true, result.endsWith("<|im_start|>assistant\n"))
        assertEquals(false, result.contains("<think>"))
    }
}
