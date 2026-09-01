package sk.ainet.apps.kgemma

import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.test.Test
import sk.ainet.llm.api.ChatOptions
import sk.ainet.llm.api.ChatRequest
import sk.ainet.llm.api.Message

/**
 * THROWAWAY spike (EdgeTranslator Phase 0, see the "Gemma via the SkaiNet engine" plan) — not part
 * of the permanent suite, delete after use. Proves whether GemmaChatModel.fromSafeTensors produces
 * a real *translation* (not just a chat reply) and whether the #325 residual repetition-loop bug
 * (tool-calling prompts degrade after 3-4 tokens) shows up on a short translate-shaped prompt too.
 *
 * Self-skips when GEMMA4_E2B_SAFETENSORS_PATH is unset. GemmaChatModel.fromSafeTensors now installs
 * the 0.51 kernel packs itself (this spike's own fix, applied to production code) — without that,
 * this 5B dense FP32 model would fall to the reference kernel path, which prior measurement showed
 * is hours-scale, not a real test.
 */
class Gemma4SafeTensorsTranslateSpike {

    @Test
    fun `translate-shaped prompt via SafeTensors`() {
        val raw = System.getenv("GEMMA4_E2B_SAFETENSORS_PATH")?.trim().orEmpty()
        if (raw.isEmpty()) {
            println("[skip] GEMMA4_E2B_SAFETENSORS_PATH not set.")
            return
        }
        val path = Path.of(raw)
        if (!path.exists()) {
            println("[skip] Path does not exist: $path")
            return
        }

        val t0 = System.nanoTime()
        val model = GemmaChatModel.fromSafeTensors(
            indexPath = path.toString(),
            options = ChatOptions(temperature = 0f, maxTokens = 64),
        )
        val loadMs = (System.nanoTime() - t0) / 1_000_000
        println("[spike] load took ${loadMs}ms")

        try {
            val system = "You are a professional translator. Translate the user's message from " +
                "English to French. Reply with ONLY the translation, nothing else."
            val user = "How are you? I hope you are doing well today."

            val t1 = System.nanoTime()
            val response = model.call(
                ChatRequest(
                    messages = listOf(
                        Message.system(system),
                        Message.user(user),
                    ),
                ),
            )
            val genMs = (System.nanoTime() - t1) / 1_000_000

            println("[spike] generate took ${genMs}ms")
            println("[spike] finish=${response.generations.firstOrNull()?.finishReason}")
            println("[spike] usage=${response.usage}")
            println("[spike] text='${response.text}'")
        } finally {
            model.close()
        }
    }
}
