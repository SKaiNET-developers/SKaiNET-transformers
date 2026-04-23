package sk.ainet.apps.kllama.chat

/**
 * Tool-calling support for the Gemma 4 family.
 *
 * Distinct from [GemmaToolCallingSupport] because Gemma 4 uses different
 * turn markers and tool delimiters:
 *   - turns: `<|turn>` / `<turn|>` (vs Gemma 2/3's `<start_of_turn>` /
 *     `<end_of_turn>`)
 *   - tool definitions: `<|tool>...<tool|>`
 *   - tool calls: `<|tool_call>...<tool_call|>`
 *   - tool responses: `<|tool_response>...<tool_response|>`
 *   - thinking mode: `<|think|>`
 *
 * Registered ahead of [GemmaToolCallingSupport] in
 * [ToolCallingSupportResolver] so Gemma 4 checkpoints take this path
 * first. Gemma 2/3 / Gemma 3n checkpoints fall through to the older
 * provider (which uses `<start_of_turn>`).
 */
public class Gemma4ToolCallingSupport : ToolCallingSupport {
    override val family: String = "gemma4"

    private val template = Gemma4ChatTemplate()

    override fun supports(metadata: ModelMetadata): Boolean {
        val f = metadata.family?.lowercase()
        if (f == "gemma4") return true
        val arch = metadata.architecture?.lowercase()
        if (arch == "gemma4") return true
        // Gemma 4's chat template uses `<|turn>` markers; older Gemma
        // variants use `<start_of_turn>`. The presence of the Gemma 4
        // marker is a reliable discriminator.
        val tpl = metadata.chatTemplate ?: return false
        return tpl.contains("<|turn>")
    }

    override fun createChatTemplate(): ChatTemplate = template

    override fun parseToolCalls(content: String): List<ToolCall> =
        template.parseToolCalls(content)

    override fun toolCallingMode(metadata: ModelMetadata): ToolCallingMode =
        ToolCallingMode.NATIVE
}
