package sk.ainet.apps.kgemma

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Integration: the one-line EAGER FunctionGemma facade transcribes an instruction into a tool call
 * by running gemmaNetwork() on the CPU (no iree). Self-skips without the GGUF. Run with:
 *   ./gradlew -PuseLocalSkainet=true -PkgemmaTestMaxHeap=12g \
 *     :llm-runtime:kgemma:jvmTest --tests "*FunctionGemmaEagerTest*"
 */
class FunctionGemmaEagerTest {
    private val gguf = System.getenv("GEMMA_GGUF")
        ?: "/home/miso/projects/coral/SKaiNET-embedded/sl2610-function-calling/models/functiongemma-physical-ai-v10-Q5_K_M.gguf"

    @Test
    fun eager_call_turn_the_light_on() {
        if (!File(gguf).exists()) {
            println("SKIP FunctionGemmaEagerTest: GGUF not present at $gguf")
            return
        }
        val fg = FunctionGemma.fromGguf(gguf)
        val turn = fg.call("turn the light on")
        println("EAGER text='${turn.text}' calls=${turn.calls}")

        assertTrue(turn.calls.isNotEmpty(), "expected a tool call, got text='${turn.text}'")
        val c = turn.calls.first()
        assertEquals("set_lights", c.tool, "tool")
        assertEquals("on", c.args["state"], "state arg")
    }
}
