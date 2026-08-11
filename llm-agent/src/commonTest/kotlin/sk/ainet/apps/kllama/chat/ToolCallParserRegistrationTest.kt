package sk.ainet.apps.kllama.chat

import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

/**
 * Tests for runtime registration of family-specific parser strategies (#40).
 */
class ToolCallParserRegistrationTest {

    /** Toy format: `%%call name {"k":"v"}%%`. */
    private class PercentCallStrategy(override val formatName: String = "percent") :
        ToolCallParserStrategy {

        private val pattern = Regex("""%%call (\w+) (\{[\s\S]*?\})%%""")

        override fun parse(text: String): List<ToolCall> =
            pattern.findAll(text).mapNotNull { m ->
                ToolCallParser.parseJsonToolCall(
                    """{"name": "${m.groupValues[1]}", "arguments": ${m.groupValues[2]}}"""
                )
            }.toList()

        override fun containsToolCall(text: String): Boolean = pattern.containsMatchIn(text)
    }

    @AfterTest
    fun cleanup() {
        ToolCallParser.unregisterStrategy("percent")
        ToolCallParser.unregisterStrategy("greedy")
    }

    @Test
    fun builtInsAreAlwaysPresent() {
        assertEquals(
            listOf("hermes", "llama3-function-tag", "llama3-json"),
            ToolCallParser.registeredFormats()
        )
    }

    @Test
    fun registeredStrategyIsUsedByParse() {
        val text = """%%call get_weather {"city": "Bratislava"}%%"""
        assertTrue(ToolCallParser.parse(text).isEmpty(), "no built-in should match the toy format")

        ToolCallParser.registerStrategy(PercentCallStrategy())
        val calls = ToolCallParser.parse(text)
        assertEquals(1, calls.size)
        assertEquals("get_weather", calls[0].name)
        assertEquals(JsonPrimitive("Bratislava"), calls[0].arguments["city"])
        assertTrue(ToolCallParser.containsToolCall(text))
    }

    @Test
    fun registeredStrategyOutranksBuiltIns() {
        // A greedy strategy that claims any text containing "call".
        val greedy = object : ToolCallParserStrategy {
            override val formatName: String = "greedy"
            override fun parse(text: String): List<ToolCall> =
                if (text.contains("<tool_call>")) {
                    listOf(ToolCall(id = "g", name = "greedy_won", arguments = JsonObject(emptyMap())))
                } else emptyList()

            override fun containsToolCall(text: String): Boolean = text.contains("<tool_call>")
        }
        ToolCallParser.registerStrategy(greedy)
        // Hermes would normally parse this; the registered strategy is prepended and wins.
        val calls = ToolCallParser.parse("""<tool_call>{"name": "real", "arguments": {}}</tool_call>""")
        assertEquals("greedy_won", calls.single().name)
        assertEquals("greedy", ToolCallParser.registeredFormats().first())
    }

    @Test
    fun sameFormatNameReplacesPreviousRegistration() {
        ToolCallParser.registerStrategy(PercentCallStrategy())
        ToolCallParser.registerStrategy(PercentCallStrategy())
        assertEquals(1, ToolCallParser.registeredFormats().count { it == "percent" })
    }

    @Test
    fun unregisterRestoresDefaultChain() {
        ToolCallParser.registerStrategy(PercentCallStrategy())
        assertTrue(ToolCallParser.unregisterStrategy("percent"))
        assertFalse(ToolCallParser.unregisterStrategy("percent"), "second removal finds nothing")
        assertEquals(
            listOf("hermes", "llama3-function-tag", "llama3-json"),
            ToolCallParser.registeredFormats()
        )
        assertTrue(ToolCallParser.parse("""%%call x {}%%""").isEmpty())
    }

    @Test
    fun unregisterCannotRemoveBuiltIns() {
        assertFalse(ToolCallParser.unregisterStrategy("hermes"))
        assertTrue("hermes" in ToolCallParser.registeredFormats())
    }
}
