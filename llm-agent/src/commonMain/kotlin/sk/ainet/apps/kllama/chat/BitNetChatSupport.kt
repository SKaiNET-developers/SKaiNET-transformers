package sk.ainet.apps.kllama.chat

/**
 * Chat template for BitNet b1.58 (`microsoft/BitNet-b1.58-2B-4T`) — the E3 entry-point work of
 * transformers#360.
 *
 * The authoritative format is the HF checkpoint's `tokenizer_config.json` template:
 *
 * ```
 * {Role}: {content}<|eot_id|>
 * ```
 *
 * per message with the role capitalized (`System:` / `User:` / `Assistant:`), and a bare
 * `Assistant: ` generation prompt. Turns end with `<|eot_id|>` (128009) — NOT the GGUF-declared
 * `eos_token_id` 128001 (`<|end_of_text|>`), the classic Llama-3-lineage mismatch; this template
 * reports it via [stopTokenStrings] so the chat loop actually stops. (The GGUF's own embedded
 * `tokenizer.chat_template` is a conversion artifact — `Human:`/`BITNETAssistant:` with a
 * misplaced eos — and is deliberately not followed.)
 *
 * 2B4T has no native tool-calling format; tool definitions are ignored.
 */
public class BitNetChatTemplate : ChatTemplate {

    override fun apply(
        messages: List<ChatMessage>,
        tools: List<ToolDefinition>,
        addGenerationPrompt: Boolean,
    ): String = buildString {
        append("<|begin_of_text|>")
        for (m in messages) {
            val role = m.role.roleName.replaceFirstChar { it.uppercaseChar() }
            append(role).append(": ").append(m.content.trim()).append("<|eot_id|>")
        }
        if (addGenerationPrompt) append("Assistant: ")
    }

    override fun stopTokenStrings(): List<String> = listOf("<|eot_id|>")
}

/**
 * Family provider for BitNet b1.58: selects [BitNetChatTemplate] for `bitnet*` architectures.
 * Tool calling is [ToolCallingMode.UNSUPPORTED] — the family has no native tool-call format and
 * none has been validated.
 */
public class BitNetChatSupport : ToolCallingSupport {

    override val family: String get() = "bitnet"

    override fun supports(metadata: ModelMetadata): Boolean {
        if (metadata.family?.lowercase() == "bitnet") return true
        return metadata.architecture?.lowercase()?.startsWith("bitnet") == true
    }

    override fun createChatTemplate(): ChatTemplate = BitNetChatTemplate()

    override fun parseToolCalls(content: String): List<ToolCall> = emptyList()

    override fun toolCallingMode(metadata: ModelMetadata): ToolCallingMode = ToolCallingMode.UNSUPPORTED
}
