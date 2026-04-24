package sk.ainet.apps.kllama.chat

/**
 * Formats a list of chat messages (and optional tool definitions) into a single
 * prompt string that the model expects.
 *
 * Different model families use different chat formats (Llama 3, ChatML, etc.).
 */
public interface ChatTemplate {

    /**
     * Apply the template to produce a prompt string.
     *
     * @param messages Conversation history.
     * @param tools Available tool definitions (empty if tool calling is not in use).
     * @param addGenerationPrompt If true, append the assistant turn prefix so the
     *   model knows it should start generating.
     * @return The formatted prompt string.
     */
    public fun apply(
        messages: List<ChatMessage>,
        tools: List<ToolDefinition> = emptyList(),
        addGenerationPrompt: Boolean = true
    ): String

    /**
     * Parse tool calls from model output text.
     *
     * The default implementation delegates to [ToolCallParser]. Templates whose
     * models use a different tool-call output format should override this method.
     */
    public fun parseToolCalls(text: String): List<ToolCall> = ToolCallParser.parse(text)

    /**
     * Check whether [text] appears to contain a tool call.
     *
     * The default implementation delegates to [ToolCallParser].
     */
    public fun containsToolCall(text: String): Boolean = ToolCallParser.containsToolCall(text)

    /**
     * Extract "thinking" blocks from model output — reasoning text the model
     * is allowed to emit but that must not be fed back into subsequent prompts
     * or shown to the end user by default. Templates whose models support a
     * thinking mode (e.g. Gemma 4's `<|think|>...<think|>`) should override.
     *
     * @return Contents of each thinking block in order of appearance; empty if none.
     */
    public fun parseThinkingBlocks(text: String): List<String> = emptyList()

    /**
     * Return [text] with every thinking block removed. Templates without a
     * thinking mode return [text] unchanged.
     */
    public fun stripThinking(text: String): String = text
}
