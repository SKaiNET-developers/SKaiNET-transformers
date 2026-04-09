package sk.ainet.apps.kllama.chat

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.assertFalse

class GemmaToolCallParserStrategyTest {

    private val strategy = GemmaToolCallParserStrategy()

    @Test
    fun parseSingleFunctionCall() {
        val text = """{"functionCall": {"name": "list_files", "args": {"path": "/tmp"}}}"""
        val calls = strategy.parse(text)
        assertEquals(1, calls.size)
        assertEquals("list_files", calls[0].name)
        assertEquals("/tmp", calls[0].arguments["path"]?.toString()?.trim('"'))
    }

    @Test
    fun parseMultipleFunctionCalls() {
        val text = """
            {"functionCall": {"name": "search", "args": {"query": "test"}}}
            {"functionCall": {"name": "calc", "args": {"expr": "1+1"}}}
        """.trimIndent()
        val calls = strategy.parse(text)
        assertEquals(2, calls.size)
        assertEquals("search", calls[0].name)
        assertEquals("calc", calls[1].name)
    }

    @Test
    fun parseEmptyArgs() {
        val text = """{"functionCall": {"name": "get_time", "args": {}}}"""
        val calls = strategy.parse(text)
        assertEquals(1, calls.size)
        assertEquals("get_time", calls[0].name)
        assertTrue(calls[0].arguments.isEmpty())
    }

    @Test
    fun parseMissingArgsField() {
        val text = """{"functionCall": {"name": "get_time"}}"""
        val calls = strategy.parse(text)
        assertEquals(1, calls.size)
        assertEquals("get_time", calls[0].name)
        assertTrue(calls[0].arguments.isEmpty())
    }

    @Test
    fun containsToolCallDetectsFunctionCall() {
        assertTrue(strategy.containsToolCall("""{"functionCall": {"name": "test"}}"""))
    }

    @Test
    fun containsToolCallReturnsFalseForPlainText() {
        assertFalse(strategy.containsToolCall("Hello, world!"))
    }

    @Test
    fun containsToolCallReturnsFalseForHermesFormat() {
        assertFalse(strategy.containsToolCall("<tool_call>{\"name\": \"calc\"}</tool_call>"))
    }

    @Test
    fun parsePlainTextReturnsEmpty() {
        assertTrue(strategy.parse("No tool calls here.").isEmpty())
    }

    @Test
    fun parseFunctionCallEmbeddedInText() {
        val text = """
            I'll look that up for you.
            {"functionCall": {"name": "search", "args": {"q": "kotlin"}}}
            Let me check...
        """.trimIndent()
        val calls = strategy.parse(text)
        assertEquals(1, calls.size)
        assertEquals("search", calls[0].name)
    }
}
