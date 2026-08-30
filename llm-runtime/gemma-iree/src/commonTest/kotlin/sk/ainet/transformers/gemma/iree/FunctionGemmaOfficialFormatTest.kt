package sk.ainet.transformers.gemma.iree

import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.putJsonObject
import kotlinx.serialization.json.add
import sk.ainet.apps.kllama.chat.ChatMessage
import sk.ainet.apps.kllama.chat.ChatRole
import sk.ainet.apps.kllama.chat.ToolDefinition
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Modelless string tests for the OFFICIAL google/functiongemma-270m-it prompt
 * and output format. The expected strings below are derived from the Jinja
 * `tokenizer.chat_template` extracted verbatim from the released GGUF (see
 * `FunctionGemmaOfficialChatTemplate` kdoc) — notably the `<escape>`-token
 * string wrapping and the property field order (`description` first, `type`
 * last). The env-gated `FunctionGemmaOfficialGgufTest` re-verifies the token
 * inventory against the GGUF itself.
 */
class FunctionGemmaOfficialFormatTest {

    private val weatherTool = ToolDefinition(
        name = "get_weather",
        description = "Get current weather for a location",
        parameters = buildJsonObject {
            put("type", "object")
            putJsonObject("properties") {
                putJsonObject("location") {
                    put("type", "string")
                    put("description", "City name")
                }
            }
            putJsonArray("required") { add("location") }
        },
    )

    @Test
    fun renders_declaration_exactly_per_release_template() {
        val decl = FunctionGemmaOfficialChatTemplate.renderDeclaration(weatherTool)
        assertEquals(
            "declaration:get_weather{" +
                "description:<escape>Get current weather for a location<escape>," +
                "parameters:{" +
                "properties:{location:{description:<escape>City name<escape>,type:<escape>STRING<escape>}}," +
                "required:[<escape>location<escape>]," +
                "type:<escape>OBJECT<escape>}}",
            decl,
        )
    }

    @Test
    fun renders_developer_turn_and_framing() {
        val template = FunctionGemmaOfficialChatTemplate()
        val prompt = template.apply(
            messages = listOf(ChatMessage(ChatRole.USER, "What's the weather in Paris?")),
            tools = listOf(weatherTool),
        )
        assertTrue(prompt.startsWith("<start_of_turn>developer\n"), prompt)
        // Preamble concatenates directly into the declaration block — no newline.
        assertTrue(
            prompt.contains("following functions<start_function_declaration>declaration:get_weather{"),
            prompt,
        )
        assertTrue(prompt.contains("<end_function_declaration><end_of_turn>\n"), prompt)
        assertTrue(prompt.contains("<start_of_turn>user\nWhat's the weather in Paris?<end_of_turn>\n"), prompt)
        assertTrue(prompt.endsWith("<start_of_turn>model\n"), prompt)
        assertFalse(prompt.contains("<bos>"), "BOS is the runtime's job, not the template's")
    }

    @Test
    fun renders_plain_chat_without_tools() {
        val template = FunctionGemmaOfficialChatTemplate()
        val prompt = template.apply(messages = listOf(ChatMessage(ChatRole.USER, "hi")))
        assertEquals("<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n", prompt)
    }

    @Test
    fun parses_official_call_with_escaped_and_bare_args() {
        val parser = FunctionGemmaOfficialToolCallParserStrategy()
        val out = "<start_function_call>call:get_weather{location:<escape>Paris<escape>,days:3}<end_function_call>"
        val calls = parser.parseCompact(out)
        assertEquals(listOf(ToolCall("get_weather", mapOf("location" to "Paris", "days" to "3"))), calls)
        assertTrue(parser.containsToolCall(out))
    }

    @Test
    fun parses_call_preceded_by_think_span() {
        val parser = FunctionGemmaOfficialToolCallParserStrategy()
        val out = "<think>user asks for weather, I should call the tool</think>" +
            "<start_function_call>call:get_weather{location:<escape>Paris<escape>}<end_function_call>"
        val calls = parser.parseCompact(out)
        assertEquals(1, calls.size)
        assertEquals("get_weather", calls[0].tool)
        assertEquals("Paris", calls[0].args["location"])
    }

    @Test
    fun strips_thinking_and_survives_truncated_end_token() {
        val template = FunctionGemmaOfficialChatTemplate()
        val out = "<think>hmm</think><start_function_call>call:noop{}"
        assertEquals("<start_function_call>call:noop{}", template.stripThinking(out))
        assertEquals(listOf("hmm"), template.parseThinkingBlocks(out))
        // Truncated output (maxTokens hit before <end_function_call>) still parses.
        assertEquals(1, template.parseToolCalls(out).size)
    }

    @Test
    fun no_call_in_plain_text() {
        val parser = FunctionGemmaOfficialToolCallParserStrategy()
        assertFalse(parser.containsToolCall("The weather in Paris is sunny."))
        assertTrue(parser.parseCompact("declaration:get_weather{}").isEmpty())
    }

    @Test
    fun agent_toolcall_bridge_carries_string_args() {
        val parser = FunctionGemmaOfficialToolCallParserStrategy()
        val calls = parser.parse(
            "<start_function_call>call:get_weather{location:<escape>Paris<escape>}<end_function_call>"
        )
        assertEquals(1, calls.size)
        assertEquals("get_weather", calls[0].name)
        assertEquals(JsonObject(mapOf("location" to JsonPrimitive("Paris"))), calls[0].arguments)
    }
}
