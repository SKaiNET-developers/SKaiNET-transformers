package sk.ainet.apps.kllama.java

import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.test.Test
import kotlin.time.measureTime

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
}
