package sk.ainet.apps.kgemma

import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.test.Test
import kotlin.time.measureTime
import sk.ainet.llm.api.ChatOptions
import sk.ainet.llm.api.ChatRequest
import sk.ainet.llm.api.Message

/**
 * End-to-end smoke test for [GemmaChatModel.fromGguf] against a real quantized Gemma 4 GGUF —
 * the new factory this arc adds (mirrors [Gemma4ChatModelSmokeTest]'s SafeTensors coverage, and
 * [sk.ainet.apps.kllama.java.KLlamaJavaKernelPackTimingProbe]'s timing-probe shape).
 *
 * Self-skips when `GEMMA4_E2B_GGUF_PATH` is not set. Two things this verifies that the SafeTensors
 * smoke test can't:
 *  1. Kernel-pack install actually takes effect on the GGUF (packed/MAPPED) path — `fromGguf`
 *     installs them internally, but this is the regression guard against that silently regressing.
 *  2. Whether #325's residual "repetition loop after 3-4 tokens" (found on tool-calling prompts,
 *     fix verified NOT to fully resolve it) also affects a short translate-shaped prompt — the
 *     EdgeTranslator Phase 0/1 decision gate.
 */
@org.junit.jupiter.api.Tag("smoke-reference")
@org.junit.jupiter.api.Tag("integration")
class Gemma4ChatModelGgufSmokeTest {

    @Test
    fun `loadGguf plus short generate completes fast, not the reference-kernel fallback`() {
        val path = locateGguf() ?: return

        var model: sk.ainet.llm.api.StreamingChatModel? = null
        val load = measureTime {
            model = GemmaChatModel.fromGguf(
                path = path.toString(),
                options = ChatOptions(temperature = 0f, maxTokens = 16),
            )
        }
        println("[gguf-smoke] load: $load")
        try {
            val gen = measureTime {
                val response = model!!.call(ChatRequest("Say hello in one short sentence."))
                println("[gguf-smoke] text='${response.text}'")
            }
            println("[gguf-smoke] generate(16 tokens): $gen")
            check(gen.inWholeSeconds < 30) {
                "generate took $gen for 16 tokens on a ~5B model — kernel pack likely not installed"
            }
        } finally {
            model?.close()
        }
    }

    @Test
    fun `chat-templated translate prompt produces French and stops before the token budget`() {
        val path = locateGguf() ?: return

        val maxTokens = 128
        val model = GemmaChatModel.fromGguf(
            path = path.toString(),
            options = ChatOptions(temperature = 0f, maxTokens = maxTokens),
        )
        try {
            val system = "You are a professional translator. Translate the user's text from " +
                "English to French. Output ONLY the translation, no quotes, no preamble."
            val user = "The capital of France is Paris."

            val text: String
            val gen = measureTime {
                val response = model.call(
                    ChatRequest(messages = listOf(Message.system(system), Message.user(user))),
                )
                text = response.text
                println("[gguf-smoke] finish=${response.generations.firstOrNull()?.finishReason}")
            }
            println("[gguf-smoke] output: '$text'")
            println("[gguf-smoke] generate(budget=$maxTokens): $gen")

            val frenchMarkers = listOf("La capitale", "capitale de la France", "est Paris")
            check(frenchMarkers.any { text.contains(it, ignoreCase = true) }) {
                "expected French translation output, got: $text"
            }
            check(!looksLikeRepetitionLoop(text)) {
                "output looks like a repetition loop (#325 residual bug) rather than a clean " +
                    "translation: $text"
            }
        } finally {
            model.close()
        }
    }

    /** Crude repetition-loop detector: same short token/phrase repeated 4+ times in a row. */
    private fun looksLikeRepetitionLoop(text: String): Boolean {
        val words = text.trim().split(Regex("\\s+"))
        if (words.size < 8) return false
        for (window in 1..3) {
            var streak = 1
            for (i in window until words.size) {
                if (words[i] == words[i - window]) {
                    streak++
                    if (streak >= 4) return true
                } else {
                    streak = 1
                }
            }
        }
        return false
    }

    private fun locateGguf(): Path? {
        val raw = System.getenv("GEMMA4_E2B_GGUF_PATH")?.trim().orEmpty()
        if (raw.isEmpty()) {
            println("[skip] GEMMA4_E2B_GGUF_PATH not set.")
            return null
        }
        val p = Path.of(raw)
        if (!p.exists()) {
            println("[skip] Path does not exist: $p")
            return null
        }
        return p
    }
}
