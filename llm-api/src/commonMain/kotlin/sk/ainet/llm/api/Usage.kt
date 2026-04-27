package sk.ainet.llm.api

/** Token accounting for a single request. */
public data class Usage(
    public val promptTokens: Int,
    public val completionTokens: Int,
) {
    public val totalTokens: Int get() = promptTokens + completionTokens
}
