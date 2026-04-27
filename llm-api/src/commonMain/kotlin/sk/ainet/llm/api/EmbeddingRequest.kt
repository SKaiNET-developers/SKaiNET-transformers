package sk.ainet.llm.api

import kotlin.jvm.JvmOverloads

/** Embedding generation request — one or more input strings. */
public data class EmbeddingRequest @JvmOverloads constructor(
    public val inputs: List<String>,
    public val options: EmbeddingOptions? = null,
) {
    public constructor(input: String) : this(listOf(input))
}

/** A single embedding result. */
public data class Embedding(
    public val index: Int,
    public val vector: FloatArray,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is Embedding) return false
        return index == other.index && vector.contentEquals(other.vector)
    }

    override fun hashCode(): Int = 31 * index + vector.contentHashCode()
}

/** Embedding generation response. */
public data class EmbeddingResponse @JvmOverloads constructor(
    public val embeddings: List<Embedding>,
    public val usage: Usage? = null,
    public val modelId: String? = null,
)
