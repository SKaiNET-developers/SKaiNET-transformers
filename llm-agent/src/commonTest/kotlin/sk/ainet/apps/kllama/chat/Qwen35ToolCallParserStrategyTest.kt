package sk.ainet.apps.kllama.chat

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.assertFalse

class Qwen35ToolCallParserStrategyTest {

    private val strategy = Qwen35ToolCallParserStrategy()

    @Test
    fun parseSingleFunctionCall() {
        val text = """
            <tool_call>
            <function=get_weather>
            <parameter=city>San Francisco</parameter>
            <parameter=unit>celsius</parameter>
            </function>
            </tool_call>
        """.trimIndent()

        val calls = strategy.parse(text)
        assertEquals(1, calls.size)
        assertEquals("get_weather", calls[0].name)
        assertEquals("San Francisco", calls[0].arguments["city"]?.toString()?.trim('"'))
        assertEquals("celsius", calls[0].arguments["unit"]?.toString()?.trim('"'))
    }

    @Test
    fun parseMultipleToolCalls() {
        val text = """
            <tool_call>
            <function=search>
            <parameter=query>Kotlin multiplatform</parameter>
            </function>
            </tool_call>
            <tool_call>
            <function=calculator>
            <parameter=expression>2 + 3</parameter>
            </function>
            </tool_call>
        """.trimIndent()

        val calls = strategy.parse(text)
        assertEquals(2, calls.size)
        assertEquals("search", calls[0].name)
        assertEquals("calculator", calls[1].name)
    }

    @Test
    fun parseNumericParameterValue() {
        val text = """
            <tool_call>
            <function=set_temperature>
            <parameter=value>42</parameter>
            </function>
            </tool_call>
        """.trimIndent()

        val calls = strategy.parse(text)
        assertEquals(1, calls.size)
        assertEquals(42, calls[0].arguments["value"]?.toString()?.toIntOrNull())
    }

    @Test
    fun parseBooleanParameterValue() {
        val text = """
            <tool_call>
            <function=toggle>
            <parameter=enabled>true</parameter>
            </function>
            </tool_call>
        """.trimIndent()

        val calls = strategy.parse(text)
        assertEquals(1, calls.size)
        assertEquals("true", calls[0].arguments["enabled"]?.toString())
    }

    @Test
    fun parseNoParameters() {
        val text = """
            <tool_call>
            <function=get_time>
            </function>
            </tool_call>
        """.trimIndent()

        val calls = strategy.parse(text)
        assertEquals(1, calls.size)
        assertEquals("get_time", calls[0].name)
        assertTrue(calls[0].arguments.isEmpty())
    }

    @Test
    fun parseWithSurroundingText() {
        val text = """
            I'll check the weather for you.
            <tool_call>
            <function=get_weather>
            <parameter=city>London</parameter>
            </function>
            </tool_call>
            Let me know if you need anything else.
        """.trimIndent()

        val calls = strategy.parse(text)
        assertEquals(1, calls.size)
        assertEquals("get_weather", calls[0].name)
        assertEquals("London", calls[0].arguments["city"]?.toString()?.trim('"'))
    }

    @Test
    fun containsToolCallDetectsQwen35Format() {
        val text = """
            <tool_call>
            <function=search>
            <parameter=query>test</parameter>
            </function>
            </tool_call>
        """.trimIndent()
        assertTrue(strategy.containsToolCall(text))
    }

    @Test
    fun containsToolCallReturnsFalseForHermesFormat() {
        // Hermes format uses JSON inside <tool_call>, not <function=...>
        val text = """<tool_call>{"name": "search", "arguments": {"query": "test"}}</tool_call>"""
        assertFalse(strategy.containsToolCall(text))
    }

    @Test
    fun containsToolCallReturnsFalseForPlainText() {
        assertFalse(strategy.containsToolCall("Hello, world!"))
    }

    @Test
    fun parseMalformedMissingClosingTagReturnsEmpty() {
        val text = "<tool_call><function=broken>"
        val calls = strategy.parse(text)
        assertTrue(calls.isEmpty())
    }

    @Test
    fun parseMissingFunctionHeaderReturnsEmpty() {
        val text = """
            <tool_call>
            some random content
            </tool_call>
        """.trimIndent()
        val calls = strategy.parse(text)
        assertTrue(calls.isEmpty())
    }

    @Test
    fun parseJsonObjectParameterValue() {
        val text = """
            <tool_call>
            <function=create_event>
            <parameter=details>{"title": "Meeting", "duration": 60}</parameter>
            </function>
            </tool_call>
        """.trimIndent()

        val calls = strategy.parse(text)
        assertEquals(1, calls.size)
        val details = calls[0].arguments["details"]
        assertTrue(details.toString().contains("Meeting"))
    }

    @Test
    fun parsePlainTextReturnsEmpty() {
        val calls = strategy.parse("This is just a normal response.")
        assertTrue(calls.isEmpty())
    }

    @Test
    fun parseEmptyStringReturnsEmpty() {
        val calls = strategy.parse("")
        assertTrue(calls.isEmpty())
    }
}
