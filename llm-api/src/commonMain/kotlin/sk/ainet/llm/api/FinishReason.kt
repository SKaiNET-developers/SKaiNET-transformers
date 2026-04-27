package sk.ainet.llm.api

/** Why generation stopped. */
public enum class FinishReason {
    /** EOS token reached or stop sequence matched. */
    STOP,
    /** Hit `maxTokens` cap. */
    LENGTH,
    /** Assistant emitted a tool call and is awaiting a tool result. */
    TOOL_CALL,
    /** Generation aborted due to error. */
    ERROR,
}
