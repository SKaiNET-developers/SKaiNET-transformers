package sk.ainet.apps.kgemma

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Integration: the one-line EAGER FunctionGemma facade transcribes an instruction into a tool call
 * by running gemmaNetwork() on the CPU (no iree). Skips without the GGUF or with too small a heap.
 * 12g is the module default heap — override with -PkgemmaTestMaxHeap. Run with:
 *   ./gradlew -PuseLocalSkainet=true \
 *     :llm-runtime:kgemma:jvmTest --tests "*FunctionGemmaEagerTest*"
 */
class FunctionGemmaEagerTest {
    private val gguf = FunctionGemmaFixture.gguf

    @Test
    fun eager_call_turn_the_light_on() {
        FunctionGemmaFixture.assumeRealCheckpointRunnable()
        val fg = FunctionGemma.fromGguf(gguf)
        val turn = fg.call("turn the light on")
        println("EAGER text='${turn.text}' calls=${turn.calls}")

        assertTrue(turn.calls.isNotEmpty(), "expected a tool call, got text='${turn.text}'")
        val c = turn.calls.first()
        assertEquals("set_lights", c.tool, "tool")
        assertEquals("on", c.args["state"], "state arg")
    }
}
