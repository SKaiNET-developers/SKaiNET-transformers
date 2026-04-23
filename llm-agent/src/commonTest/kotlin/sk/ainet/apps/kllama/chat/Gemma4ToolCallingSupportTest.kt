package sk.ainet.apps.kllama.chat

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Verifies that [Gemma4ToolCallingSupport] is selected ahead of the
 * generic [GemmaToolCallingSupport] for Gemma 4 checkpoints, and that
 * older Gemma (2/3/3n) checkpoints still land on the older provider.
 *
 * Also exercises the chat-template marker discrimination:
 *   - `<|turn>` → Gemma 4 wins
 *   - `<start_of_turn>` → Gemma 2/3 wins
 */
class Gemma4ToolCallingSupportTest {

    @Test
    fun `architecture gemma4 resolves to Gemma4ToolCallingSupport`() {
        val metadata = ModelMetadata(architecture = "gemma4")
        val provider = ToolCallingSupportResolver.resolveOrFallback(metadata)
        assertEquals("gemma4", provider.family, "Expected Gemma 4 provider, got ${provider.family}")
        assertTrue(provider is Gemma4ToolCallingSupport)
    }

    @Test
    fun `family gemma4 resolves to Gemma4ToolCallingSupport`() {
        val metadata = ModelMetadata(family = "gemma4")
        val provider = ToolCallingSupportResolver.resolveOrFallback(metadata)
        assertEquals("gemma4", provider.family)
    }

    @Test
    fun `chat template with pipe-turn marker resolves to Gemma 4`() {
        val metadata = ModelMetadata(
            chatTemplate = "<|turn>system\nYou are helpful<turn|>\n<|turn>user\n"
        )
        val provider = ToolCallingSupportResolver.resolveOrFallback(metadata)
        assertEquals("gemma4", provider.family)
    }

    @Test
    fun `Gemma 2 3 chat template resolves to generic Gemma provider`() {
        val metadata = ModelMetadata(
            chatTemplate = "<start_of_turn>user\n{{message}}<end_of_turn>"
        )
        val provider = ToolCallingSupportResolver.resolveOrFallback(metadata)
        assertEquals("gemma", provider.family, "Expected generic Gemma 2/3 provider, got ${provider.family}")
        assertTrue(provider is GemmaToolCallingSupport)
    }

    @Test
    fun `architecture gemma3n still falls through to generic Gemma provider`() {
        val metadata = ModelMetadata(architecture = "gemma3n")
        val provider = ToolCallingSupportResolver.resolveOrFallback(metadata)
        // Gemma 3n uses `<start_of_turn>` like Gemma 2/3, so the older
        // provider should win. Gemma4ToolCallingSupport must not claim
        // non-gemma4 architectures.
        assertEquals("gemma", provider.family, "Gemma 3n should not be claimed by Gemma 4 provider")
    }

    @Test
    fun `Gemma 4 provider hands out Gemma4ChatTemplate`() {
        val provider = Gemma4ToolCallingSupport()
        val template = provider.createChatTemplate()
        assertTrue(
            template is Gemma4ChatTemplate,
            "Expected Gemma4ChatTemplate, got ${template::class.simpleName}"
        )
    }

    @Test
    fun `Gemma 4 provider parses native tool_call delimiters`() {
        val provider = Gemma4ToolCallingSupport()
        val modelOutput = """Sure, let me compute that.
<|tool_call>{"name": "calculator", "args": {"expression": "3+4"}}<tool_call|>
""".trimMargin()
        val calls = provider.parseToolCalls(modelOutput)
        assertEquals(1, calls.size, "Expected 1 tool call, got ${calls.size}")
        assertEquals("calculator", calls[0].name)
    }
}
