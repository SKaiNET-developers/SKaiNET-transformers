package sk.ainet.apps.kllama.chat

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

/** transformers#360 E3: the BitNet chat entry point — template shape and resolver selection. */
class BitNetChatSupportTest {

    @Test
    fun templateRendersTheHfCheckpointFormat() {
        val template = BitNetChatTemplate()
        val prompt = template.apply(
            listOf(
                ChatMessage(role = ChatRole.SYSTEM, content = "You are a helpful assistant."),
                ChatMessage(role = ChatRole.USER, content = "What is the capital of France?"),
            ),
            emptyList(),
            addGenerationPrompt = true,
        )
        assertEquals(
            "<|begin_of_text|>" +
                "System: You are a helpful assistant.<|eot_id|>" +
                "User: What is the capital of France?<|eot_id|>" +
                "Assistant: ",
            prompt,
        )
        assertEquals(listOf("<|eot_id|>"), template.stopTokenStrings())
    }

    @Test
    fun resolverSelectsBitNetForBitnetArchitectures() {
        for (arch in listOf("bitnet", "bitnet-25", "bitnet-b1.58")) {
            val provider = ToolCallingSupportResolver.resolve(ModelMetadata(architecture = arch))
            assertIs<BitNetChatSupport>(provider, "arch '$arch' must select the BitNet provider")
        }
        // The family id set by ModelRegistry also selects it, and tool calling stays off.
        val byFamily = ToolCallingSupportResolver.resolve(ModelMetadata(family = "bitnet"))
        assertIs<BitNetChatSupport>(byFamily)
        assertTrue(byFamily.toolCallingMode(ModelMetadata(family = "bitnet")) == ToolCallingMode.UNSUPPORTED)
    }
}
