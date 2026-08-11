package sk.ainet.transformers.gemma.iree

import sk.ainet.apps.kllama.chat.ChatMessage
import sk.ainet.apps.kllama.chat.ChatRole
import sk.ainet.apps.kllama.chat.ChatTemplate
import sk.ainet.apps.kllama.chat.ToolDefinition
import sk.ainet.apps.kllama.chat.ToolCall as AgentToolCall

/**
 * [ChatTemplate] for the FunctionGemma fine-tune (Octopus-v2 style functional
 * tokens on a Gemma 3 270M base).
 *
 * Produces exactly the prompt the fine-tune was trained on — for a single user
 * message this is the string `FunctionGemma.call()` historically hardcoded:
 *
 * ```
 * <start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n
 * ```
 *
 * Two deliberate deviations from [sk.ainet.apps.kllama.chat.GemmaChatTemplate]:
 *
 * - **No newline before `<end_of_turn>`** — the fine-tune's training data
 *   closes the turn immediately after the content.
 * - **[tools] are ignored** — FunctionGemma's tool vocabulary is baked into
 *   the checkpoint as `<tool_N>` special tokens; there is no in-prompt tool
 *   declaration block. Pass the vocabulary to [FunctionGemmaToolCallParserStrategy]
 *   (via [CompactToolCodec]) instead.
 *
 * SYSTEM and TOOL messages render as `user` turns (the Gemma convention);
 * ASSISTANT renders as `model`.
 */
public class FunctionGemmaChatTemplate(
    private val parser: FunctionGemmaToolCallParserStrategy = FunctionGemmaToolCallParserStrategy(),
) : ChatTemplate {

    override fun apply(
        messages: List<ChatMessage>,
        tools: List<ToolDefinition>,
        addGenerationPrompt: Boolean,
    ): String {
        val sb = StringBuilder()
        for (msg in messages) {
            val role = if (msg.role == ChatRole.ASSISTANT) "model" else "user"
            sb.append("<start_of_turn>").append(role).append('\n')
            sb.append(msg.content)
            sb.append("<end_of_turn>\n")
        }
        if (addGenerationPrompt) {
            sb.append("<start_of_turn>model\n")
        }
        return sb.toString()
    }

    override fun parseToolCalls(text: String): List<AgentToolCall> = parser.parse(text)

    override fun containsToolCall(text: String): Boolean = parser.containsToolCall(text)
}
