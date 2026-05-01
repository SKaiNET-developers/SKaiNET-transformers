package sk.ainet.apps.kllama.chat

import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonPrimitive
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Tests for [ApertusToolCallParserStrategy]. Format reference:
 * `docs/specs/apertus-chat-template.md`. The parser scans for
 * `<|tools_prefix|>...<|tools_suffix|>`, parses the inner JSON
 * array, and emits one [ToolCall] per single-key object.
 */
class ApertusToolCallParserStrategyTest {

    @Test
    fun parses_single_tool_call() {
        val text = """The answer:<|tools_prefix|>[{"calculator": {"expression": "1+1"}}]<|tools_suffix|>"""
        val calls = ApertusToolCallParserStrategy.parse(text)
        assertEquals(1, calls.size)
        assertEquals("calculator", calls[0].name)
        assertEquals("1+1", calls[0].arguments["expression"]?.jsonPrimitive?.content)
    }

    @Test
    fun parses_multiple_parallel_tool_calls() {
        val text = """<|tools_prefix|>[{"a": {"x": 1}}, {"b": {"y": "two"}}]<|tools_suffix|>"""
        val calls = ApertusToolCallParserStrategy.parse(text)
        assertEquals(2, calls.size)
        assertEquals("a", calls[0].name)
        assertEquals("b", calls[1].name)
        assertEquals(1, calls[0].arguments["x"]?.jsonPrimitive?.content?.toInt())
        assertEquals("two", calls[1].arguments["y"]?.jsonPrimitive?.content)
    }

    @Test
    fun returns_empty_when_no_markers_present() {
        val calls = ApertusToolCallParserStrategy.parse("plain text without tool calls")
        assertTrue(calls.isEmpty())
    }

    @Test
    fun returns_empty_when_payload_invalid_json() {
        val text = "<|tools_prefix|>not valid json<|tools_suffix|>"
        val calls = ApertusToolCallParserStrategy.parse(text)
        assertTrue(calls.isEmpty(), "invalid payload must not crash: $text")
    }

    @Test
    fun returns_empty_when_object_has_zero_or_multiple_keys() {
        // Two-key object — Apertus's spec says exactly one key per object.
        val text = """<|tools_prefix|>[{"a": {}, "b": {}}]<|tools_suffix|>"""
        assertTrue(ApertusToolCallParserStrategy.parse(text).isEmpty())
        // Empty object — also ignored.
        val emptyObj = """<|tools_prefix|>[{}]<|tools_suffix|>"""
        assertTrue(ApertusToolCallParserStrategy.parse(emptyObj).isEmpty())
    }

    @Test
    fun returns_empty_when_only_one_marker_present() {
        assertTrue(ApertusToolCallParserStrategy.parse("<|tools_prefix|>[{\"a\": {}}]").isEmpty())
        assertTrue(ApertusToolCallParserStrategy.parse("[{\"a\": {}}]<|tools_suffix|>").isEmpty())
    }

    @Test
    fun containsToolCall_requires_both_markers() {
        assertTrue(ApertusToolCallParserStrategy.containsToolCall("<|tools_prefix|>[]<|tools_suffix|>"))
        assertFalse(ApertusToolCallParserStrategy.containsToolCall("<|tools_prefix|>[]"))
        assertFalse(ApertusToolCallParserStrategy.containsToolCall("[]<|tools_suffix|>"))
        assertFalse(ApertusToolCallParserStrategy.containsToolCall("plain text"))
    }

    @Test
    fun assigns_distinct_ids_to_each_call() {
        val text = """<|tools_prefix|>[{"a": {}}, {"b": {}}, {"c": {}}]<|tools_suffix|>"""
        val calls = ApertusToolCallParserStrategy.parse(text)
        assertEquals(3, calls.size)
        assertEquals(setOf("apertus-tool-call-0", "apertus-tool-call-1", "apertus-tool-call-2"), calls.map { it.id }.toSet())
    }

    @Test
    fun ignores_text_outside_markers() {
        val text = "preamble<|tools_prefix|>[{\"x\": {}}]<|tools_suffix|>postscript"
        val calls = ApertusToolCallParserStrategy.parse(text)
        assertEquals(1, calls.size)
        assertEquals("x", calls[0].name)
    }

    @Test
    fun resolver_picks_apertus_for_apertus_family() {
        val md = ModelMetadata(family = "apertus", architecture = "apertus")
        val support = ToolCallingSupportResolver.resolve(md)
        assertNotNull(support)
        assertEquals("apertus", support.family)
    }

    @Test
    fun resolver_picks_apertus_for_apertus_chat_template_marker() {
        val md = ModelMetadata(
            family = null,
            architecture = null,
            chatTemplate = "...something with <|assistant_start|> in it..."
        )
        val support = ToolCallingSupportResolver.resolve(md)
        assertNotNull(support)
        assertEquals("apertus", support.family)
    }
}
