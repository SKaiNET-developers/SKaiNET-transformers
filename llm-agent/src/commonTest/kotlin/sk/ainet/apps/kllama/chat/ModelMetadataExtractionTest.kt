package sk.ainet.apps.kllama.chat

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class ModelMetadataExtractionTest {

    // --- fromGgufFields ---

    @Test
    fun fullGgufFields() {
        val metadata = ModelMetadataExtraction.fromGgufFields(
            mapOf(
                "general.architecture" to "qwen2",
                "tokenizer.chat_template" to "{% if tools %}<|im_start|>...Qwen...<tool_call>",
                "tokenizer.ggml.tokens" to listOf("<|im_start|>", "<|im_end|>", "hello")
            )
        )
        assertEquals("qwen", metadata.family)
        assertEquals("qwen2", metadata.architecture)
        assertEquals("gguf", metadata.sourceFormat)
        assertTrue("<tool_call>" in metadata.tokenizerHints)
        assertTrue("<|im_start|>" in metadata.tokenizerHints)
    }

    @Test
    fun missingChatTemplateDegradesGracefully() {
        val metadata = ModelMetadataExtraction.fromGgufFields(
            mapOf("general.architecture" to "llama")
        )
        assertEquals("llama", metadata.family)
        assertEquals("llama", metadata.architecture)
        assertNull(metadata.chatTemplate)
        assertTrue(metadata.tokenizerHints.isEmpty())
    }

    @Test
    fun emptyFieldsProduceUnknownArchitecture() {
        val metadata = ModelMetadataExtraction.fromGgufFields(emptyMap())
        assertNull(metadata.family)
        assertEquals("unknown", metadata.architecture)
        assertNull(metadata.chatTemplate)
        assertEquals("gguf", metadata.sourceFormat)
    }

    @Test
    fun wrongFieldTypesAreIgnored() {
        val metadata = ModelMetadataExtraction.fromGgufFields(
            mapOf(
                "general.architecture" to 42,
                "tokenizer.chat_template" to listOf("not", "a", "string"),
                "tokenizer.ggml.tokens" to "not a collection"
            )
        )
        assertNull(metadata.family)
        assertEquals("unknown", metadata.architecture)
        assertNull(metadata.chatTemplate)
        assertTrue(metadata.tokenizerHints.isEmpty())
    }

    // --- familyFromArchitecture ---

    @Test
    fun familyMapping() {
        assertEquals("qwen", ModelMetadataExtraction.familyFromArchitecture("qwen2"))
        assertEquals("qwen", ModelMetadataExtraction.familyFromArchitecture("qwen35"))
        assertEquals("gemma", ModelMetadataExtraction.familyFromArchitecture("gemma3"))
        assertEquals("llama", ModelMetadataExtraction.familyFromArchitecture("llama"))
        // unknown architectures pass through so custom providers can match
        assertEquals("apertus", ModelMetadataExtraction.familyFromArchitecture("apertus"))
        assertNull(ModelMetadataExtraction.familyFromArchitecture(null))
    }

    @Test
    fun familyMappingIsCaseInsensitive() {
        assertEquals("qwen", ModelMetadataExtraction.familyFromArchitecture("Qwen2"))
    }

    // --- tokenizerHints ---

    @Test
    fun hintsFromChatTemplateOnly() {
        val hints = ModelMetadataExtraction.tokenizerHints(
            chatTemplate = "<|start_header_id|>system<|end_header_id|> ... <|python_tag|>",
            vocabTokens = null
        )
        assertEquals(listOf("<|python_tag|>", "<|start_header_id|>"), hints)
    }

    @Test
    fun hintsFromVocabOnly() {
        val hints = ModelMetadataExtraction.tokenizerHints(
            chatTemplate = null,
            vocabTokens = listOf("plain", "<tool_call>", "[AVAILABLE_TOOLS]")
        )
        assertEquals(listOf("<tool_call>", "[AVAILABLE_TOOLS]"), hints)
    }

    @Test
    fun noInputsNoHints() {
        assertTrue(ModelMetadataExtraction.tokenizerHints(null, null).isEmpty())
        assertTrue(ModelMetadataExtraction.tokenizerHints(null, emptyList()).isEmpty())
    }

    // --- end-to-end: extraction output drives resolver auto-detection ---

    @Test
    fun extractedQwenMetadataResolvesQwenProvider() {
        val metadata = ModelMetadataExtraction.fromGgufFields(
            mapOf("general.architecture" to "qwen2")
        )
        val provider = ToolCallingSupportResolver.resolve(metadata)
        assertEquals("qwen", provider?.family)
    }

    @Test
    fun extractedGemmaMetadataResolvesGemmaProvider() {
        val metadata = ModelMetadataExtraction.fromGgufFields(
            mapOf("general.architecture" to "gemma3")
        )
        val provider = ToolCallingSupportResolver.resolve(metadata)
        assertEquals("gemma", provider?.family)
    }
}
