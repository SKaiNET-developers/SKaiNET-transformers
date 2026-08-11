package sk.ainet.transformers.gemma.iree

import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Mirrors [CompactCodecTest] through llm-agent's [sk.ainet.apps.kllama.chat.ToolCallParserStrategy]
 * surface, plus the injectable-vocabulary path (#35/#36: a 7th tool without a library edit).
 */
class FunctionGemmaToolCallParserStrategyTest {

    private val strategy = FunctionGemmaToolCallParserStrategy()

    private fun args(vararg pairs: Pair<String, String>): JsonObject =
        JsonObject(pairs.associate { (k, v) -> k to JsonPrimitive(v) })

    @Test
    fun parsesNamedArgCall() {
        val calls = strategy.parse("""<tool_0>(state="on")<end>""")
        assertEquals(1, calls.size)
        assertEquals("set_lights", calls.single().name)
        assertEquals(args("state" to "on"), calls.single().arguments)
    }

    @Test
    fun parsesMultipleArgsAndCalls() {
        val calls = strategy.parse("""<tool_0>(color="red", state="on")<end><tool_5>(message="ok")<end>""")
        assertEquals(listOf("set_lights", "respond"), calls.map { it.name })
        assertEquals(args("color" to "red", "state" to "on"), calls[0].arguments)
        assertEquals(args("message" to "ok"), calls[1].arguments)
    }

    @Test
    fun dropsNoneTool() {
        assertEquals(emptyList(), strategy.parse("""<tool_none>()<end>"""))
    }

    @Test
    fun assignsUniqueCallIds() {
        val calls = strategy.parse("""<tool_0>(state="on")<end><tool_5>(message="ok")<end>""")
        assertEquals(2, calls.map { it.id }.distinct().size)
    }

    @Test
    fun containsToolCallDetectsCompactFormat() {
        assertTrue(strategy.containsToolCall("""<tool_0>(state="on")<end>"""))
        assertTrue(strategy.containsToolCall("""<tool_none>()<end>"""))
        assertFalse(strategy.containsToolCall("""{"functionCall": {"name": "set_lights"}}"""))
        assertFalse(strategy.containsToolCall("just some prose"))
    }

    @Test
    fun customToolMapServesSeventhToolWithoutLibraryEdit() {
        val custom = FunctionGemmaToolCallParserStrategy(
            CompactToolCodec(CompactToolCodec.DEFAULT_TOKEN_TO_NAME + ("6" to "open_gripper")),
        )
        val calls = custom.parse("""<tool_6>(width="0.5")<end><tool_0>(state="off")<end>""")
        assertEquals(listOf("open_gripper", "set_lights"), calls.map { it.name })
        assertEquals(args("width" to "0.5"), calls[0].arguments)
    }

    @Test
    fun customToolMapCanReplaceVocabularyEntirely() {
        val custom = FunctionGemmaToolCallParserStrategy(
            CompactToolCodec(mapOf("0" to "do_thing", "none" to null)),
        )
        assertEquals(listOf("do_thing"), custom.parse("""<tool_0>()<end>""").map { it.name })
        // Token 5 exists in the stock vocabulary but not in this one — dropped.
        assertEquals(emptyList(), custom.parse("""<tool_5>(message="ok")<end>"""))
    }
}
