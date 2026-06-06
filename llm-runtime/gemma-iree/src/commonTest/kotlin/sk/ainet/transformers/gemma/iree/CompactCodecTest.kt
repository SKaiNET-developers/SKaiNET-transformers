package sk.ainet.transformers.gemma.iree

import kotlin.test.Test
import kotlin.test.assertEquals

class CompactCodecTest {
    @Test
    fun parsesNamedArgCall() {
        assertEquals(
            listOf(ToolCall("set_lights", mapOf("state" to "on"))),
            CompactCodec.parse("""<tool_0>(state="on")<end>"""),
        )
    }

    @Test
    fun parsesMultipleArgsAndCalls() {
        assertEquals(
            listOf(
                ToolCall("set_lights", mapOf("color" to "red", "state" to "on")),
                ToolCall("respond", mapOf("message" to "ok")),
            ),
            CompactCodec.parse("""<tool_0>(color="red", state="on")<end><tool_5>(message="ok")<end>"""),
        )
    }

    @Test
    fun dropsNoneTool() {
        assertEquals(emptyList(), CompactCodec.parse("""<tool_none>()<end>"""))
    }
}
