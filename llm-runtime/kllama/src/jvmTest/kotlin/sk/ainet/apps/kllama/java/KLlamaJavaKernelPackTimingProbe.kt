package sk.ainet.apps.kllama.java

import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.test.Test
import kotlin.time.measureTime
import sk.ainet.apps.kllama.chat.ChatMessage
import sk.ainet.apps.kllama.chat.ChatRole
import sk.ainet.apps.kllama.chat.Llama3ChatTemplate

/**
 * Manual timing probe for #338: confirms [KLlamaJava.loadGGUF] (the exact call EdgeTranslator's
 * SkaiNetLlm.jvm.kt makes) hits the fast FfmRowMajorKernelPack path, not the decoding reference
 * kernel. Set KLLAMA_JAVA_PROBE_GGUF to a Llama/Mistral GGUF path to run; skips (not fails)
 * when unset — this isn't a CI-portable model.
 */
class KLlamaJavaKernelPackTimingProbe {

    @Test
    fun `loadGGUF plus generate completes fast, not the reference-kernel fallback`() {
        val envPath = System.getenv("KLLAMA_JAVA_PROBE_GGUF")?.trim().orEmpty()
        if (envPath.isEmpty()) {
            println("[skip] KLLAMA_JAVA_PROBE_GGUF not set")
            return
        }
        val path = Path.of(envPath)
        if (!path.exists()) {
            println("[skip] model not present at $path")
            return
        }
        var session: KLlamaSession? = null
        val load = measureTime { session = KLlamaJava.loadGGUF(path) }
        println("load: $load")
        val config = GenerationConfig.builder().maxTokens(16).temperature(0f).build()
        val gen = measureTime {
            val text = session!!.generate("The capital of France is", config)
            println("output: $text")
        }
        println("generate(16 tokens): $gen")
        session!!.close()
        check(gen.inWholeSeconds < 30) {
            "generate took $gen for 16 tokens on a 1B model — kernel pack likely not installed (reference-kernel fallback)"
        }
    }

    /**
     * Reproduces EdgeTranslator's SkaiNetLlm.jvm.kt exactly: the real translate_prompt.txt
     * system message, wrapped via [Llama3ChatTemplate] (not a raw "$system\n\n$user"
     * concatenation, which is what produced English-not-French output that never stopped).
     * Verifies both symptoms are fixed: output looks like French, and generation halts well
     * before the token budget (measured against the ~40ms/token baseline from the timing probe).
     */
    @Test
    fun `chat-templated translate prompt produces French and stops before the token budget`() {
        val envPath = System.getenv("KLLAMA_JAVA_PROBE_GGUF")?.trim().orEmpty()
        if (envPath.isEmpty()) {
            println("[skip] KLLAMA_JAVA_PROBE_GGUF not set")
            return
        }
        val path = Path.of(envPath)
        if (!path.exists()) {
            println("[skip] model not present at $path")
            return
        }
        val session = KLlamaJava.loadGGUF(path)
        val system = """
            You are a professional translator.
            Translate the user's text from English to French.
            Rules:
            - Output ONLY the translation. No quotes, no notes, no preamble, no markdown.
            - Preserve paragraph breaks, punctuation, numbers, URLs and code.
            - Do not add or omit information.
        """.trimIndent()
        val prompt = Llama3ChatTemplate().apply(
            listOf(
                ChatMessage(role = ChatRole.SYSTEM, content = system),
                ChatMessage(role = ChatRole.USER, content = "The capital of France is Paris."),
            ),
        )
        val maxTokens = 400
        val config = GenerationConfig.builder().maxTokens(maxTokens).temperature(0f).build()
        val text: String
        val gen = measureTime { text = session.generate(prompt, config) }
        session.close()
        println("output: $text")
        println("generate(budget=$maxTokens): $gen")

        val frenchMarkers = listOf("La capitale", "capitale de la France", "est Paris")
        check(frenchMarkers.any { text.contains(it, ignoreCase = true) }) {
            "expected French translation output, got: $text"
        }
        // ~40ms/token measured baseline (661ms/16 tok) — stopping naturally should land well
        // under the full budget's worth of wall time, not run out the maxTokens cap.
        check(gen.inWholeMilliseconds < maxTokens * 40 / 2) {
            "generate took $gen for a budget of $maxTokens tokens — looks like it ran to the cap instead of emitting <|eot_id|>: $text"
        }
    }
}
