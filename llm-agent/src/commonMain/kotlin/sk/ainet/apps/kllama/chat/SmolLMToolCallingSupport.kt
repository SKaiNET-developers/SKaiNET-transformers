package sk.ainet.apps.kllama.chat

/**
 * Tool-calling support for the SmolLM2-Instruct family.
 *
 * SmolLM2 is a Llama-architecture model that uses ChatML envelopes plus a
 * specific system-prompt recipe for tool calling. Detection signals come from
 * one of:
 *
 * - [ModelMetadata.family] equal to "smollm" or "smollm2"
 * - [ModelMetadata.architecture] containing "smol" (rare — most distributions
 *   report the underlying `llama` arch)
 * - [ModelMetadata.chatTemplate] containing the literal "SmolLM" or
 *   "named SmolLM" string baked into the official tokenizer chat template
 *
 * Must be registered before [ChatMLToolCallingSupport] in
 * [ToolCallingSupportResolver]: SmolLM2's chat template uses `<|im_start|>`,
 * which the generic ChatML provider also matches.
 */
public class SmolLMToolCallingSupport : ToolCallingSupport {

    override val family: String = "smollm"

    private val template = SmolLMChatTemplate()

    override fun supports(metadata: ModelMetadata): Boolean {
        val f = metadata.family?.lowercase()
        if (f == "smollm" || f == "smollm2") return true
        val arch = metadata.architecture?.lowercase()
        if (arch != null && arch.contains("smol")) return true
        val tpl = metadata.chatTemplate ?: return false
        return tpl.contains("SmolLM")
    }

    override fun createChatTemplate(): ChatTemplate = template

    override fun parseToolCalls(content: String): List<ToolCall> =
        template.parseToolCalls(content)

    override fun toolCallingMode(metadata: ModelMetadata): ToolCallingMode =
        ToolCallingMode.NATIVE
}
