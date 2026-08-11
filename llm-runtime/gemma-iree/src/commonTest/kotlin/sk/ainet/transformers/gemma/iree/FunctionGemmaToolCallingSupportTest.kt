package sk.ainet.transformers.gemma.iree

import sk.ainet.apps.kllama.chat.ChatMessage
import sk.ainet.apps.kllama.chat.ChatRole
import sk.ainet.apps.kllama.chat.ModelMetadata
import sk.ainet.apps.kllama.chat.ToolCallingMode
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class FunctionGemmaToolCallingSupportTest {

    private val support = FunctionGemmaToolCallingSupport()

    @Test
    fun templateReproducesTheExactFunctionGemmaPrompt() {
        // The string FunctionGemma.call() hardcoded before the template existed —
        // FunctionGemmaEagerTest runs the real checkpoint on this exact prompt.
        val prompt = support.createChatTemplate().apply(
            messages = listOf(ChatMessage(ChatRole.USER, "turn the light on")),
        )
        assertEquals("<start_of_turn>user\nturn the light on<end_of_turn>\n<start_of_turn>model\n", prompt)
    }

    @Test
    fun templateWithoutGenerationPromptOmitsModelPrefix() {
        val prompt = support.createChatTemplate().apply(
            messages = listOf(ChatMessage(ChatRole.USER, "hi")),
            addGenerationPrompt = false,
        )
        assertEquals("<start_of_turn>user\nhi<end_of_turn>\n", prompt)
    }

    @Test
    fun templateRendersAssistantAsModelTurn() {
        val prompt = support.createChatTemplate().apply(
            messages = listOf(
                ChatMessage(ChatRole.USER, "turn the light on"),
                ChatMessage(ChatRole.ASSISTANT, """<tool_0>(state="on")<end>"""),
                ChatMessage(ChatRole.USER, "now off"),
            ),
        )
        assertEquals(
            "<start_of_turn>user\nturn the light on<end_of_turn>\n" +
                "<start_of_turn>model\n<tool_0>(state=\"on\")<end><end_of_turn>\n" +
                "<start_of_turn>user\nnow off<end_of_turn>\n" +
                "<start_of_turn>model\n",
            prompt,
        )
    }

    @Test
    fun parsesToolCallsThroughTheProvider() {
        val calls = support.parseToolCalls("""<tool_4>(metric="all")<end>""")
        assertEquals("get_system_status", calls.single().name)
    }

    @Test
    fun supportsMatchesFamilyAndFunctionalTokensButNotStockGemma() {
        assertTrue(support.supports(ModelMetadata(family = "functiongemma")))
        assertTrue(support.supports(ModelMetadata(tokenizerHints = listOf("<tool_0>", "<tool_none>"))))
        // Base-arch gemma3 without the functional tokens must fall through to
        // the stock Gemma provider.
        assertFalse(support.supports(ModelMetadata(family = "gemma", architecture = "gemma3")))
    }

    @Test
    fun modeIsNative() {
        assertEquals(ToolCallingMode.NATIVE, support.toolCallingMode(ModelMetadata()))
    }
}
