package sk.ainet.llm.api

import kotlin.jvm.JvmOverloads

/**
 * A chat completion request.
 *
 * @param messages Conversation history; the model continues from the last message.
 * @param options Per-request overrides; `null` means fall back to the model's [ChatModel.defaultOptions].
 * @param tools Tools the model may call (empty if tool calling is not in use).
 */
public data class ChatRequest @JvmOverloads constructor(
    public val messages: List<Message>,
    public val options: ChatOptions? = null,
    public val tools: List<ToolDefinition> = emptyList(),
) {
    public constructor(prompt: String) : this(listOf(Message.user(prompt)))
}
