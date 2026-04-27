package sk.ainet.llm.api

/** A single completion candidate. */
public data class Generation(
    public val message: Message,
    public val finishReason: FinishReason,
)

/** Synchronous chat response. */
public data class ChatResponse(
    public val generations: List<Generation>,
    public val usage: Usage? = null,
    public val modelId: String? = null,
) {
    /** Convenience: text content of the first generation, or empty string if there is none. */
    public val text: String get() = generations.firstOrNull()?.message?.content.orEmpty()
}

/**
 * One streaming increment.
 *
 * @param delta Newly generated text since the last chunk (may be empty for non-text chunks).
 * @param toolCallDelta New tool call(s) detected in this chunk (typically only on the final chunk).
 * @param finishReason Set on the terminal chunk only.
 */
public data class ChatResponseChunk(
    public val delta: String,
    public val toolCallDelta: List<ToolCall> = emptyList(),
    public val finishReason: FinishReason? = null,
    public val usage: Usage? = null,
)
