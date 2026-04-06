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
}
