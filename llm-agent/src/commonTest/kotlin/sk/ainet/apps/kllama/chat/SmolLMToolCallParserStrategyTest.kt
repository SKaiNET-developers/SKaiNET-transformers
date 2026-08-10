package sk.ainet.apps.kllama.chat

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Unit coverage for [SmolLMToolCallParserStrategy] — the SmolLM2 tool-call
 * format (JSON array inside `<tool_call>...</tool_call>`, with a bare-object
 * fallback for models that drop the outer brackets). Runs on every target
 * with no model checkpoint, unlike the gated end-to-end smoke test.
 */
class SmolLMToolCallParserStrategyTest {

    private val strategy = SmolLMToolCallParserStrategy()

    @Test
    fun parses_official_array_form() {
        val text = """<tool_call>[{"name": "get_weather", "arguments": {"city": "Paris"}}]</tool_call>"""
        val calls = strategy.parse(text)
        assertEquals(1, calls.size)
        assertEquals("get_weather", calls[0].name)
        assertEquals("Paris", calls[0].arguments["city"]?.toString()?.trim('"'))
    }

    @Test
    fun parses_multiple_calls_in_one_array() {
        val text = """<tool_call>[
            {"name": "search", "arguments": {"q": "kotlin"}},
            {"name": "calc", "arguments": {"expr": "1+1"}}
        ]</tool_call>"""
        val calls = strategy.parse(text)
        assertEquals(2, calls.size)
        assertEquals("search", calls[0].name)
        assertEquals("calc", calls[1].name)
    }

    @Test
    fun tolerates_bare_object_fallback() {
        // Small models sometimes drop the outer array brackets.
        val text = """<tool_call>{"name": "get_time", "arguments": {}}</tool_call>"""
        val calls = strategy.parse(text)
        assertEquals(1, calls.size)
        assertEquals("get_time", calls[0].name)
        assertTrue(calls[0].arguments.isEmpty())
    }

    @Test
    fun array_form_wins_when_both_shapes_present() {
        // A well-formed array must not be shadowed by a stray object match.
        val text = """<tool_call>[{"name": "a", "arguments": {"nested": {"k": "v"}}}]</tool_call>"""
        val calls = strategy.parse(text)
        assertEquals(1, calls.size)
        assertEquals("a", calls[0].name)
        // Nested object inside arguments must survive the hand-rolled splitter.
        assertTrue(calls[0].arguments["nested"] != null)
    }

    @Test
    fun ignores_commas_inside_string_literals() {
        val text = """<tool_call>[{"name": "echo", "arguments": {"text": "a, b, c"}}]</tool_call>"""
        val calls = strategy.parse(text)
        assertEquals(1, calls.size)
        assertEquals("a, b, c", calls[0].arguments["text"]?.toString()?.trim('"'))
    }

    @Test
    fun no_tool_call_returns_empty() {
        val calls = strategy.parse("The capital of France is Paris.")
        assertTrue(calls.isEmpty())
    }

    @Test
    fun containsToolCall_detects_marker() {
        assertTrue(strategy.containsToolCall("prefix <tool_call>[]</tool_call> suffix"))
        assertFalse(strategy.containsToolCall("no markers here"))
    }
}
