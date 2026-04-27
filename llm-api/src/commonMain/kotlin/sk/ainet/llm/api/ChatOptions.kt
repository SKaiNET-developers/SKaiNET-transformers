package sk.ainet.llm.api

import kotlin.jvm.JvmField
import kotlin.jvm.JvmOverloads

/**
 * Per-request knobs for chat generation.
 *
 * Mirrors the knob set used by Spring AI / OpenAI / Ollama so the design is familiar.
 * `null` means "use the model's default". Concrete adapters MAY ignore knobs they
 * don't support (and should log a one-time warning when they do).
 */
public data class ChatOptions @JvmOverloads constructor(
    public val model: String? = null,
    public val temperature: Float? = null,
    public val topK: Int? = null,
    public val topP: Float? = null,
    public val maxTokens: Int? = null,
    public val stopSequences: List<String> = emptyList(),
    public val seed: Long? = null,
) {
    public companion object {
        @JvmField
        public val DEFAULTS: ChatOptions = ChatOptions(
            temperature = 0.7f,
            maxTokens = 512,
        )
    }
}

/** Per-request knobs for embedding generation. */
public data class EmbeddingOptions @JvmOverloads constructor(
    public val model: String? = null,
    /** Request a specific output dimensionality (only honored if the model supports projection). */
    public val dimensions: Int? = null,
)
