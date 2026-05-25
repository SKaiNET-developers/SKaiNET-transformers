package sk.ainet.apps.kgemma

import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.io.path.isDirectory
import kotlin.test.Test
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Assumptions.assumeTrue
import sk.ainet.llm.api.ChatOptions
import sk.ainet.llm.api.ChatRequest
import sk.ainet.llm.api.FinishReason

/**
 * Reference smoke test #2 — Gemma-4 E2B SafeTensors (kgemma runner).
 *
 * Locked in as part of the SKaiNET 0.25.0 bump. Exercises:
 *  - SafeTensors load path. With SKaiNET 0.25.0 the BF16 SafeTensors path
 *    becomes policy-aware via `SafeTensorsParametersLoader.withPolicy` —
 *    this test stays on the default FP32-dequant adaptive default but pins
 *    the loader integration end-to-end.
 *  - Complex architecture: sliding-window attention + per-layer KV sharing
 *    (Gemma-4 specifics that the new policy-resolution pass must not break).
 *  - `Gemma4ChatModel.fromSafeTensors` → `InferenceRuntime` → `Tokenizer` →
 *    `Gemma4ChatTemplate` → `SkaiNetChatModel` wiring.
 *
 * Tagged `@Tag("smoke-reference")` so it runs only under
 * `./gradlew test -PsmokeReference -PincludeIntegration`. Self-skips when
 * `GEMMA4_E2B_SAFETENSORS_PATH` is not set.
 */
@Tag("smoke-reference")
@Tag("integration")
class Gemma4ReferenceSmokeTest {

    @Test
    fun `Gemma-4 E2B SafeTensors produces non-empty greedy text`() {
        val indexPath = locateCheckpoint()
        assumeTrue(indexPath != null, "GEMMA4_E2B_SAFETENSORS_PATH not set or path missing.")

        val maxTokens = probeMaxTokens(default = 32)

        val model = Gemma4ChatModel.fromSafeTensors(
            indexPath = indexPath!!.toString(),
            options = ChatOptions(
                temperature = 0f,
                maxTokens = maxTokens,
            ),
        )
        try {
            val response = model.call(ChatRequest("Say hello in one short sentence."))

            println("[smoke-reference] Gemma-4 modelId=${response.modelId}")
            println("[smoke-reference] Gemma-4 finish=${response.generations.firstOrNull()?.finishReason}")
            println("[smoke-reference] Gemma-4 text='${response.text.replace("\n", "\\n")}'")

            assertTrue(response.text.isNotBlank(), "Gemma-4 chat returned blank text")
            val finish = response.generations.firstOrNull()?.finishReason
            assertNotNull(finish, "Gemma-4 missing finish reason")
            assertTrue(
                finish == FinishReason.STOP || finish == FinishReason.LENGTH ||
                    finish == FinishReason.TOOL_CALL,
                "Gemma-4 unexpected finish reason: $finish",
            )
            val usage = response.usage
            assertNotNull(usage, "Gemma-4 usage should be reported")
            assertTrue(usage.completionTokens > 0, "Gemma-4 expected at least one completion token")
        } finally {
            model.close()
        }
    }

    private fun probeMaxTokens(default: Int): Int =
        System.getenv("GEMMA4_SMOKE_MAX_TOKENS")?.trim()?.toIntOrNull()?.coerceAtLeast(1) ?: default

    private fun locateCheckpoint(): Path? {
        val raw = System.getenv("GEMMA4_E2B_SAFETENSORS_PATH")?.trim().orEmpty()
        if (raw.isEmpty()) return null
        val p = Path.of(raw)
        if (!p.exists()) return null
        return when {
            p.isDirectory() -> {
                val idx = p.resolve("model.safetensors.index.json")
                val single = p.resolve("model.safetensors")
                when {
                    idx.exists() -> idx
                    single.exists() -> single
                    else -> null
                }
            }
            else -> p
        }
    }
}
