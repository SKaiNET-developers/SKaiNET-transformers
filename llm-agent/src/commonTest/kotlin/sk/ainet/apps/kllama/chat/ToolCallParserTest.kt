package sk.ainet.apps.kllama.chat

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.assertFalse

class ToolCallParserTest {

    // --- Hermes-style parsing ---

    @Test
    fun parseHermesStyleToolCall() {
        val text = """
            Some thinking text...
            <tool_call>
            {"name": "calculator", "arguments": {"expression": "2 + 3"}}
            </tool_call>
        """.trimIndent()

        val calls = ToolCallParser.parse(text)
        assertEquals(1, calls.size)
        assertEquals("calculator", calls[0].name)
        assertEquals("2 + 3", calls[0].arguments["expression"]?.toString()?.trim('"'))
    }

    @Test
    fun parseMultipleHermesToolCalls() {
        val text = """
            <tool_call>{"name": "search", "arguments": {"query": "weather"}}</tool_call>
            <tool_call>{"name": "calculator", "arguments": {"expression": "1+1"}}</tool_call>
        """.trimIndent()

        val calls = ToolCallParser.parse(text)
        assertEquals(2, calls.size)
        assertEquals("search", calls[0].name)
        assertEquals("calculator", calls[1].name)
    }

    // --- Llama 3.1-style parsing ---

    @Test
    fun parseLlama31StyleToolCall() {
        val text = """{"name": "calculator", "arguments": {"expression": "5 * 10"}}"""

        val calls = ToolCallParser.parse(text)
        assertEquals(1, calls.size)
        assertEquals("calculator", calls[0].name)
        assertEquals("5 * 10", calls[0].arguments["expression"]?.toString()?.trim('"'))
    }

    @Test
    fun parseLlama31StyleWithId() {
        val text = """{"id": "call_42", "name": "search", "arguments": {"query": "Kotlin"}}"""

        val calls = ToolCallParser.parse(text)
        assertEquals(1, calls.size)
        assertEquals("call_42", calls[0].id)
        assertEquals("search", calls[0].name)
    }

    // --- No tool call ---

    @Test
    fun parsePlainTextReturnsEmpty() {
        val text = "This is just a normal response with no tool calls."
        val calls = ToolCallParser.parse(text)
        assertTrue(calls.isEmpty())
    }

    @Test
    fun parseEmptyStringReturnsEmpty() {
        val calls = ToolCallParser.parse("")
        assertTrue(calls.isEmpty())
    }

    @Test
    fun parseMalformedJsonReturnsEmpty() {
        val text = """{"name": "calculator", "arguments": {invalid json}}"""
        val calls = ToolCallParser.parse(text)
        assertTrue(calls.isEmpty())
    }

    // --- containsToolCall ---

    @Test
    fun containsToolCallDetectsHermesFormat() {
        assertTrue(ToolCallParser.containsToolCall("<tool_call>{}</tool_call>"))
    }

    @Test
    fun containsToolCallDetectsJsonFormat() {
        assertTrue(ToolCallParser.containsToolCall("""{"name": "calc", "arguments": {}}"""))
    }

    @Test
    fun containsToolCallReturnsFalseForPlainText() {
        assertFalse(ToolCallParser.containsToolCall("Hello, world!"))
    }

    // --- Edge cases ---

    @Test
    fun parseMissingArgumentsField() {
        val text = """{"name": "no_args"}"""
        val calls = ToolCallParser.parse(text)
        // Should still parse — missing arguments defaults to empty
        assertEquals(1, calls.size)
        assertEquals("no_args", calls[0].name)
        assertTrue(calls[0].arguments.isEmpty())
    }

    @Test
    fun parseMissingNameReturnsEmpty() {
        val text = """{"arguments": {"expression": "1+1"}}"""
        val calls = ToolCallParser.parse(text)
        assertTrue(calls.isEmpty())
    }

    // --- Llama 3.2 JSON: "parameters" key alias for "arguments" ---

    @Test
    fun parseLlama32JsonWithParametersKey() {
        // Meta's Llama 3.2 docs use "parameters" as the argument-object key.
        val text = """{"name": "list_files", "parameters": {"path": "/tmp"}}"""
        val calls = ToolCallParser.parse(text)
        assertEquals(1, calls.size)
        assertEquals("list_files", calls[0].name)
        assertEquals("/tmp", calls[0].arguments["path"]?.toString()?.trim('"'))
    }

    @Test
    fun parseLlama32JsonStripsPythonTagPrefix() {
        // Llama 3.2 sometimes emits <|python_tag|> before the JSON for built-in tools.
        // The parser should tolerate it on custom tool calls too.
        val text = """<|python_tag|>{"name": "calc", "parameters": {"x": 1}}"""
        val calls = ToolCallParser.parse(text)
        assertEquals(1, calls.size)
        assertEquals("calc", calls[0].name)
    }

    @Test
    fun parseLlama32JsonIgnoresTrailingProse() {
        // Small models often append commentary after the JSON. The parser must
        // grab the first balanced {...} and ignore the rest.
        val text = """{"name": "list_files", "parameters": {"path": "/tmp"}} I hope that helps!"""
        val calls = ToolCallParser.parse(text)
        assertEquals(1, calls.size)
        assertEquals("list_files", calls[0].name)
    }

    // --- Llama 3.1 legacy <function=...>...</function> ---

    @Test
    fun parseLlama31FunctionTagSingleCall() {
        val text = """<function=list_files>{"path": "/tmp"}</function>"""
        val calls = ToolCallParser.parse(text)
        assertEquals(1, calls.size)
        assertEquals("list_files", calls[0].name)
        assertEquals("/tmp", calls[0].arguments["path"]?.toString()?.trim('"'))
    }

    @Test
    fun parseLlama31FunctionTagMultipleCalls() {
        val text = """
            <function=search>{"q": "weather"}</function>
            <function=calc>{"expression": "1+1"}</function>
        """.trimIndent()
        val calls = ToolCallParser.parse(text)
        assertEquals(2, calls.size)
        assertEquals("search", calls[0].name)
        assertEquals("calc", calls[1].name)
    }

    @Test
    fun parseLlama31FunctionTagWithSurroundingProse() {
        val text = """Sure, let me check.
            <function=list_files>{"path": "/tmp"}</function>
            One moment."""
        val calls = ToolCallParser.parse(text)
        assertEquals(1, calls.size)
        assertEquals("list_files", calls[0].name)
    }

    @Test
    fun containsToolCallDetectsFunctionTag() {
        assertTrue(ToolCallParser.containsToolCall("""<function=x>{}</function>"""))
    }

    @Test
    fun containsToolCallDetectsParametersKey() {
        assertTrue(ToolCallParser.containsToolCall("""{"name": "x", "parameters": {}}"""))
    }

    // --- Llama 3.2 sometimes wraps its JSON in a markdown code fence ---

    @Test
    fun parseLlama32JsonInsideTripleBacktickFence() {
        val text = """
            ```
            {"name": "list_files", "parameters": {"path": "/tmp"}}
            ```
        """.trimIndent()
        val calls = ToolCallParser.parse(text)
        assertEquals(1, calls.size)
        assertEquals("list_files", calls[0].name)
        assertEquals("/tmp", calls[0].arguments["path"]?.toString()?.trim('"'))
    }

    @Test
    fun parseLlama32JsonInsideJsonTaggedFence() {
        val text = """
            ```json
            {"name": "calc", "parameters": {"x": 1}}
            ```
        """.trimIndent()
        val calls = ToolCallParser.parse(text)
        assertEquals(1, calls.size)
        assertEquals("calc", calls[0].name)
    }

    @Test
    fun containsToolCallDetectsFencedJson() {
        val text = """```
{"name": "x", "parameters": {}}
```"""
        assertTrue(ToolCallParser.containsToolCall(text))
    }
}
