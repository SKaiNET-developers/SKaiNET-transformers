package sk.ainet.apps.kllama.chat

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class ToolCallingSupportResolverTest {

    // --- Explicit family selection ---

    @Test
    fun explicitLlama3() {
        val provider = ToolCallingSupportResolver.resolve(explicitFamily = "llama3")
        assertNotNull(provider)
        assertIs<Llama3ToolCallingSupport>(provider)
    }

    @Test
    fun explicitQwen() {
        val provider = ToolCallingSupportResolver.resolve(explicitFamily = "qwen")
        assertNotNull(provider)
        assertIs<QwenToolCallingSupport>(provider)
    }

    @Test
    fun explicitGemma() {
        val provider = ToolCallingSupportResolver.resolve(explicitFamily = "gemma")
        assertNotNull(provider)
        assertIs<GemmaToolCallingSupport>(provider)
    }

    @Test
    fun explicitChatML() {
        val provider = ToolCallingSupportResolver.resolve(explicitFamily = "chatml")
        assertNotNull(provider)
        assertIs<ChatMLToolCallingSupport>(provider)
    }

    @Test
    fun explicitHermesAlias() {
        val provider = ToolCallingSupportResolver.resolve(explicitFamily = "hermes")
        assertNotNull(provider)
        assertIs<ChatMLToolCallingSupport>(provider)
    }

    @Test
    fun explicitCaseInsensitive() {
        val provider = ToolCallingSupportResolver.resolve(explicitFamily = "QWEN")
        assertNotNull(provider)
        assertIs<QwenToolCallingSupport>(provider)
    }

    // --- Explicit override takes precedence over metadata ---

    @Test
    fun explicitOverridesMetadata() {
        val metadata = ModelMetadata(family = "gemma")
        val provider = ToolCallingSupportResolver.resolve(metadata, explicitFamily = "qwen")
        assertNotNull(provider)
        assertIs<QwenToolCallingSupport>(provider)
    }

    // --- Auto-detection from metadata ---

    @Test
    fun autoDetectQwenByFamily() {
        val metadata = ModelMetadata(family = "qwen")
        val provider = ToolCallingSupportResolver.resolve(metadata)
        assertNotNull(provider)
        assertIs<QwenToolCallingSupport>(provider)
    }

    @Test
    fun autoDetectGemmaByFamily() {
        val metadata = ModelMetadata(family = "gemma")
        val provider = ToolCallingSupportResolver.resolve(metadata)
        assertNotNull(provider)
        assertIs<GemmaToolCallingSupport>(provider)
    }

    @Test
    fun autoDetectLlamaByFamily() {
        val metadata = ModelMetadata(family = "llama")
        val provider = ToolCallingSupportResolver.resolve(metadata)
        assertNotNull(provider)
        assertIs<Llama3ToolCallingSupport>(provider)
    }

    @Test
    fun autoDetectGemmaByChatTemplate() {
        val metadata = ModelMetadata(chatTemplate = "... <start_of_turn>user ...")
        val provider = ToolCallingSupportResolver.resolve(metadata)
        assertNotNull(provider)
        assertIs<GemmaToolCallingSupport>(provider)
    }

    @Test
    fun autoDetectLlamaByChatTemplate() {
        val metadata = ModelMetadata(chatTemplate = "... <|start_header_id|>system ...")
        val provider = ToolCallingSupportResolver.resolve(metadata)
        assertNotNull(provider)
        assertIs<Llama3ToolCallingSupport>(provider)
    }

    @Test
    fun autoDetectQwenByChatTemplate() {
        val metadata = ModelMetadata(chatTemplate = "... Qwen ... <|im_start|> ...")
        val provider = ToolCallingSupportResolver.resolve(metadata)
        assertNotNull(provider)
        assertIs<QwenToolCallingSupport>(provider)
    }

    @Test
    fun autoDetectChatMLByChatTemplate() {
        val metadata = ModelMetadata(chatTemplate = "... <|im_start|>system ...")
        val provider = ToolCallingSupportResolver.resolve(metadata)
        assertNotNull(provider)
        assertIs<ChatMLToolCallingSupport>(provider)
    }

    @Test
    fun autoDetectGemmaByArchitecture() {
        val metadata = ModelMetadata(architecture = "gemma3n")
        val provider = ToolCallingSupportResolver.resolve(metadata)
        assertNotNull(provider)
        assertIs<GemmaToolCallingSupport>(provider)
    }

    // --- No match ---

    @Test
    fun noMatchReturnsNull() {
        val metadata = ModelMetadata(family = "unknown_family_xyz")
        val provider = ToolCallingSupportResolver.resolve(metadata)
        assertNull(provider)
    }

    @Test
    fun emptyMetadataReturnsNull() {
        val provider = ToolCallingSupportResolver.resolve(ModelMetadata())
        assertNull(provider)
    }

    // --- resolveOrFallback ---

    @Test
    fun fallbackReturnsGeneric() {
        val provider = ToolCallingSupportResolver.resolveOrFallback(ModelMetadata())
        assertIs<GenericToolCallingSupport>(provider)
        assertEquals(ToolCallingMode.GENERIC, provider.toolCallingMode(ModelMetadata()))
    }

    @Test
    fun fallbackPrefersNativeWhenAvailable() {
        val metadata = ModelMetadata(family = "qwen")
        val provider = ToolCallingSupportResolver.resolveOrFallback(metadata)
        assertIs<QwenToolCallingSupport>(provider)
        assertEquals(ToolCallingMode.NATIVE, provider.toolCallingMode(metadata))
    }

    // --- Tool calling mode ---

    @Test
    fun nativeProvidersModeIsNative() {
        val metadata = ModelMetadata(family = "llama3")
        val provider = ToolCallingSupportResolver.resolve(metadata)!!
        assertEquals(ToolCallingMode.NATIVE, provider.toolCallingMode(metadata))
    }

    // --- Deterministic ordering ---

    @Test
    fun qwenWinsOverChatMLForQwenTemplate() {
        // Both Qwen and ChatML use <|im_start|>, but Qwen should win when "Qwen" is in template
        val metadata = ModelMetadata(chatTemplate = "Qwen <|im_start|>system")
        val provider = ToolCallingSupportResolver.resolve(metadata)
        assertNotNull(provider)
        assertIs<QwenToolCallingSupport>(provider)
    }
}
