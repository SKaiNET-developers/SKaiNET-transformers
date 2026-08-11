package sk.ainet.apps.kllama.chat

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class HuggingFaceMetadataExtractionTest {

    @Test
    fun chatTemplateStringFromTokenizerConfig() {
        val metadata = ModelMetadataExtraction.fromHuggingFaceConfig(
            tokenizerConfigJson = """{"chat_template": "<|im_start|>{{ messages }}<tool_call>"}"""
        )
        assertEquals("<|im_start|>{{ messages }}<tool_call>", metadata.chatTemplate)
        assertEquals("hf", metadata.sourceFormat)
        assertTrue("<tool_call>" in metadata.tokenizerHints)
    }

    @Test
    fun chatTemplateListPrefersDefaultEntry() {
        val metadata = ModelMetadataExtraction.fromHuggingFaceConfig(
            tokenizerConfigJson = """
                {"chat_template": [
                    {"name": "tool_use", "template": "tools-template"},
                    {"name": "default", "template": "default-template"}
                ]}
            """.trimIndent()
        )
        assertEquals("default-template", metadata.chatTemplate)
    }

    @Test
    fun chatTemplateListFallsBackToFirstEntry() {
        val metadata = ModelMetadataExtraction.fromHuggingFaceConfig(
            tokenizerConfigJson = """
                {"chat_template": [
                    {"name": "rag", "template": "rag-template"},
                    {"name": "tool_use", "template": "tools-template"}
                ]}
            """.trimIndent()
        )
        assertEquals("rag-template", metadata.chatTemplate)
    }

    @Test
    fun chatTemplateJsonFileUsedWhenTokenizerConfigHasNone() {
        val metadata = ModelMetadataExtraction.fromHuggingFaceConfig(
            tokenizerConfigJson = """{"bos_token": "<s>"}""",
            chatTemplateJson = """{"chat_template": "from-chat-template-json"}"""
        )
        assertEquals("from-chat-template-json", metadata.chatTemplate)
    }

    @Test
    fun tokenizerConfigTemplateWinsOverChatTemplateJson() {
        val metadata = ModelMetadataExtraction.fromHuggingFaceConfig(
            tokenizerConfigJson = """{"chat_template": "from-tokenizer-config"}""",
            chatTemplateJson = """{"chat_template": "from-chat-template-json"}"""
        )
        assertEquals("from-tokenizer-config", metadata.chatTemplate)
    }

    @Test
    fun modelTypeMapsToFamily() {
        val metadata = ModelMetadataExtraction.fromHuggingFaceConfig(
            modelConfigJson = """{"model_type": "qwen2", "hidden_size": 896}"""
        )
        assertEquals("qwen2", metadata.architecture)
        assertEquals("qwen", metadata.family)
    }

    @Test
    fun additionalSpecialTokensFeedHints() {
        val metadata = ModelMetadataExtraction.fromHuggingFaceConfig(
            tokenizerConfigJson = """
                {"additional_special_tokens": ["<tool_call>", {"content": "<|im_start|>"}]}
            """.trimIndent()
        )
        assertTrue("<tool_call>" in metadata.tokenizerHints)
        assertTrue("<|im_start|>" in metadata.tokenizerHints)
    }

    @Test
    fun malformedJsonDegradesGracefully() {
        val metadata = ModelMetadataExtraction.fromHuggingFaceConfig(
            tokenizerConfigJson = "{not json",
            chatTemplateJson = "also {not json",
            modelConfigJson = "nope["
        )
        assertNull(metadata.family)
        assertNull(metadata.architecture)
        assertNull(metadata.chatTemplate)
        assertTrue(metadata.tokenizerHints.isEmpty())
        assertEquals("hf", metadata.sourceFormat)
    }

    @Test
    fun noInputsProduceEmptyMetadata() {
        val metadata = ModelMetadataExtraction.fromHuggingFaceConfig()
        assertNull(metadata.chatTemplate)
        assertNull(metadata.family)
    }

    // --- end-to-end: HF metadata drives resolver auto-detection ---

    @Test
    fun qwenSafetensorsCheckpointResolvesQwenProvider() {
        val metadata = ModelMetadataExtraction.fromHuggingFaceConfig(
            tokenizerConfigJson = """{"chat_template": "You are Qwen...<|im_start|>"}""",
            modelConfigJson = """{"model_type": "qwen2"}"""
        )
        val provider = ToolCallingSupportResolver.resolve(metadata)
        assertEquals("qwen", provider?.family)
    }

    @Test
    fun llama3SafetensorsCheckpointResolvesLlama3Provider() {
        val metadata = ModelMetadataExtraction.fromHuggingFaceConfig(
            tokenizerConfigJson = """{"chat_template": "{{ '<|start_header_id|>' }}"}"""
        )
        val provider = ToolCallingSupportResolver.resolve(metadata)
        assertEquals("llama3", provider?.family)
    }
}
