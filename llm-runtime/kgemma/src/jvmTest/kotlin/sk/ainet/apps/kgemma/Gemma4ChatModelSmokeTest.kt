package sk.ainet.apps.kgemma

import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.io.path.isDirectory
import kotlin.test.Test
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import sk.ainet.llm.api.ChatOptions
import sk.ainet.llm.api.ChatRequest
import sk.ainet.llm.api.FinishReason

/**
 * End-to-end smoke test for [GemmaChatModel.fromSafeTensors] against a real
 * Gemma 4 SafeTensors checkpoint. Proves the v1 wiring goal: SafeTensors →
 * `InferenceRuntime` → `Tokenizer` → `Gemma4ChatTemplate` → `SkaiNetChatModel`
 * produces non-empty text.
 *
 * Self-skips when `GEMMA4_E2B_SAFETENSORS_PATH` is not set so default CI stays
 * green (full E2B FP32 is ~20 GB resident — opt-in only).
 *
 * Does NOT assert a fixed output prefix. Gemma 4 numerical parity on real
 * weights is still being validated; the assertion bar is "coherent enough"
 * (non-blank, decodable, reasonable finish reason).
 */
class Gemma4ChatModelSmokeTest {

    @Test
    fun `chat model call returns non-empty text on FP32 SafeTensors`() {
        val indexPath = locateCheckpoint() ?: return
        val maxTokens = probeMaxTokens(default = 32)

        val model = GemmaChatModel.fromSafeTensors(
            indexPath = indexPath.toString(),
            options = ChatOptions(
                temperature = 0f,
                maxTokens = maxTokens,
            ),
        )

        try {
            val response = model.call(ChatRequest("Say hello in one short sentence."))

            println("[smoke] modelId=${response.modelId}")
            println("[smoke] finish=${response.generations.firstOrNull()?.finishReason}")
            println("[smoke] usage=${response.usage}")
            println("[smoke] text='${response.text.replace("\n", "\\n")}'")

            assertTrue(response.text.isNotBlank(), "ChatModel produced blank text")
            val finish = response.generations.firstOrNull()?.finishReason
            assertNotNull(finish, "Missing finish reason")
            assertTrue(
                finish == FinishReason.STOP || finish == FinishReason.LENGTH ||
                    finish == FinishReason.TOOL_CALL,
                "Unexpected finish reason: $finish",
            )
            val usage = response.usage
            assertNotNull(usage, "Usage should be reported")
            assertTrue(usage.completionTokens > 0, "Expected at least one completion token")
        } finally {
            model.close()
        }
    }

    @Test
    fun `chat model stream yields chunks then a terminal chunk with finish reason`() {
        val indexPath = locateCheckpoint() ?: return

        val model = GemmaChatModel.fromSafeTensors(
            indexPath = indexPath.toString(),
            options = ChatOptions(
                temperature = 0f,
                maxTokens = probeMaxTokens(default = 16),
            ),
        )

        try {
            val chunks = runBlocking {
                model.stream(ChatRequest("Hi.")).toList()
            }
            println("[smoke] stream produced ${chunks.size} chunks")

            assertTrue(chunks.isNotEmpty(), "Stream produced no chunks")
            val terminal = chunks.last()
            assertNotNull(terminal.finishReason, "Terminal chunk missing finish reason")
            val anyTextChunk = chunks.any { it.delta.isNotEmpty() }
            assertTrue(anyTextChunk, "Stream had no non-empty delta chunk")
        } finally {
            model.close()
        }
    }

    /**
     * Optional override for the per-call generation budget. Lets a developer
     * run a fast probe (e.g. `GEMMA4_SMOKE_MAX_TOKENS=4`) without recompiling.
     */
    private fun probeMaxTokens(default: Int): Int =
        System.getenv("GEMMA4_SMOKE_MAX_TOKENS")?.trim()?.toIntOrNull()?.coerceAtLeast(1) ?: default

    private fun locateCheckpoint(): Path? {
        val raw = System.getenv("GEMMA4_E2B_SAFETENSORS_PATH")?.trim().orEmpty()
        if (raw.isEmpty()) {
            println("[skip] GEMMA4_E2B_SAFETENSORS_PATH not set.")
            return null
        }
        val p = Path.of(raw)
        if (!p.exists()) {
            println("[skip] Path does not exist: $p")
            return null
        }
        // Accept either the index file directly, the directory containing it,
        // or a single-file model.safetensors layout.
        val resolved = when {
            p.isDirectory() -> {
                val idx = p.resolve("model.safetensors.index.json")
                val single = p.resolve("model.safetensors")
                when {
                    idx.exists() -> idx
                    single.exists() -> single
                    else -> {
                        println("[skip] No SafeTensors checkpoint found in $p")
                        return null
                    }
                }
            }
            else -> p
        }
        return resolved
    }
}
