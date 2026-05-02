package sk.ainet.apps.kllama.chat

/**
 * Tool-calling support for the Apertus family (Swiss AI / EPFL).
 *
 * Wires up [ApertusChatTemplate] for prompt formatting and
 * [ApertusToolCallParserStrategy] for parsing tool-call output. See
 * `docs/specs/apertus-chat-template.md` for the format reference.
 */
public class ApertusToolCallingSupport : ToolCallingSupport {
    override val family: String = "apertus"

    private val template: ApertusChatTemplate = ApertusChatTemplate()

    override fun supports(metadata: ModelMetadata): Boolean {
        val f = metadata.family?.lowercase()
        if (f == "apertus") return true
        val arch = metadata.architecture?.lowercase()
        if (arch == "apertus") return true
        // Disambiguating chat-template marker: Apertus is the only family
        // that uses `<|assistant_start|>` (vs llama3's <|start_header_id|>,
        // chatml's <|im_start|>, gemma's <start_of_turn>).
        val tpl = metadata.chatTemplate ?: return false
        return tpl.contains("<|assistant_start|>")
    }

    override fun createChatTemplate(): ChatTemplate = template

    override fun parseToolCalls(content: String): List<ToolCall> =
        ApertusToolCallParserStrategy.parse(content)

    override fun toolCallingMode(metadata: ModelMetadata): ToolCallingMode =
        ToolCallingMode.NATIVE
}
