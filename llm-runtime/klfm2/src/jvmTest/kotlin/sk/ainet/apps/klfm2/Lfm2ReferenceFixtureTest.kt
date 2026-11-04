package sk.ainet.apps.klfm2

import kotlinx.serialization.json.*
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Assumptions.assumeTrue
import sk.ainet.models.lfm2.Lfm2Config
import sk.ainet.models.lfm2.loadLfm2Model
import java.io.File
import java.security.MessageDigest
import kotlin.test.*

/** Small pinned reference outputs; real model files are supplied explicitly and never downloaded. */
class Lfm2ReferenceFixtureTest {
    private val fixture = Json.parseToJsonElement(javaClass.getResource("/lfm25-chat.json")!!.readText()).jsonObject
    private fun messages(value: JsonElement) = value.jsonArray.map {
        val o = it.jsonObject
        Lfm2ChatMessage(o.getValue("role").jsonPrimitive.content, o.getValue("content").jsonPrimitive.content,
            o["thinking"]?.takeUnless { it == JsonNull }?.jsonPrimitive?.content)
    }
    private fun ids(value: JsonElement) = value.jsonArray.map { it.jsonPrimitive.int }.toIntArray()

    @Test fun pinnedReferenceTextCasesMatchAllTemplateBytesWithoutModelFiles() {
        val cases = fixture.getValue("cases").jsonArray
        assertEquals(38, cases.size)
        for (value in cases) {
            val case = value.jsonObject
            val text = renderLfm2TextChat(messages(case.getValue("messages")), "<|startoftext|>",
                case.getValue("add_generation_prompt").jsonPrimitive.boolean,
                case.getValue("preserve_thinking").jsonPrimitive.boolean)
            assertEquals(case.getValue("expected_rendered").jsonPrimitive.content, text, case.getValue("id").jsonPrimitive.content)
            val hash = MessageDigest.getInstance("SHA-256").digest(text.toByteArray(Charsets.UTF_8)).joinToString("") { "%02x".format(it) }
            assertEquals(case.getValue("expected_rendered_utf8_sha256").jsonPrimitive.content, hash)
        }
    }

    @Test @Tag("integration")
    fun suppliedPinnedCheckpointMatchesTokenizerAndShortGreedyReference() {
        val directoryPath = System.getenv("LFM2_MODEL_DIR")
        assumeTrue(!directoryPath.isNullOrBlank(), "Set LFM2_MODEL_DIR to an explicit local pinned checkpoint directory")
        val directory = File(directoryPath!!)
        val frontend = loadLfm2TextFrontend(directory)
        assertTrue(frontend.metadataPinned)
        val unpinned = loadLfm2TextFrontend(directory, pins = null)
        assertFalse(unpinned.metadataPinned)
        assertEquals(frontend.effectiveTokenizerSha256, unpinned.effectiveTokenizerSha256)
        assertEquals(frontend.metadataHashes.filterKeys { it != "LICENSE" }, unpinned.metadataHashes)
        for (value in fixture.getValue("cases").jsonArray) {
            val case = value.jsonObject
            val tokenIds = frontend.encodeMessages(messages(case.getValue("messages")),
                case.getValue("add_generation_prompt").jsonPrimitive.boolean, case.getValue("preserve_thinking").jsonPrimitive.boolean)
            assertContentEquals(ids(case.getValue("expected_ids")), tokenIds, case.getValue("id").jsonPrimitive.content)
            assertEquals(case.getValue("expected_decoded").jsonPrimitive.content, frontend.decode(tokenIds, skipSpecialTokens = false))
            assertEquals(case.getValue("expected_decoded_skip_special_tokens").jsonPrimitive.content,
                frontend.decode(tokenIds, policy = Lfm2DecodingPolicy.HUGGING_FACE_5_COMPAT))
        }
        val checkpoint = fixture.getValue("generation_provenance").jsonObject.getValue("checkpoint").jsonObject
        val loaded = loadLfm2Model(Lfm2Config.parse(File(directory, "config.json").readText()), File(directory, "model.safetensors"),
            contextLimit = 256, expectedCheckpointSha256 = checkpoint.getValue("sha256").jsonPrimitive.content)
        val generator = Lfm2TextGenerator(loaded.model, frontend, Lfm2DecodingPolicy.HUGGING_FACE_5_COMPAT)
        for (value in fixture.getValue("generation_traces").jsonArray) {
            val case = value.jsonObject
            val result = generator.generate(messages(case.getValue("messages")),
                Lfm2GenerationOptions(case.getValue("max_new_tokens").jsonPrimitive.int))
            assertContentEquals(ids(case.getValue("prompt_ids")), result.promptTokenIds)
            assertContentEquals(ids(case.getValue("expected_generated_ids")), result.generatedTokenIds)
            assertEquals(case.getValue("expected_decoded_completion_skip_special_tokens").jsonPrimitive.content, result.text)
            assertEquals(case.getValue("expected_stop_reason").jsonPrimitive.content, result.stopReason)
            assertEquals(case.getValue("expected_state_input_tokens").jsonPrimitive.int, result.stateInputTokens)
        }
    }
}
