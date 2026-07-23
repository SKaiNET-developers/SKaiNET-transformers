package sk.ainet.llm.api

import kotlin.AutoCloseable

/**
 * Provider-neutral text-embedding SPI.
 *
 * Mirrors Spring AI's `EmbeddingModel` shape; no Spring deps.
 */
public interface EmbeddingModel : AutoCloseable {

    /** Batch embedding. The result list is the same length and order as [EmbeddingRequest.inputs]. */
    public fun call(request: EmbeddingRequest): EmbeddingResponse

    /** Convenience: embed a single string and return its raw vector. */
    public fun embed(text: String): FloatArray =
        call(EmbeddingRequest(text)).embeddings.first().vector

    /** Convenience: embed many strings, return raw vectors in input order. */
    public fun embed(texts: List<String>): List<FloatArray> =
        call(EmbeddingRequest(texts)).embeddings.sortedBy { it.index }.map { it.vector }

    /**
     * Embed a *search query*. Retrieval-tuned models embed queries and
     * documents asymmetrically (instruction prefixes — E5's `query: `, BGE's
     * `Represent this sentence for searching relevant passages: `); models
     * without that distinction inherit this default, which is plain [embed].
     */
    public fun embedQuery(text: String): FloatArray = embed(text)

    /** Embed a *document / passage* for indexing. See [embedQuery]. */
    public fun embedDocument(text: String): FloatArray = embed(text)

    /** Batch [embedDocument], vectors in input order. */
    public fun embedDocuments(texts: List<String>): List<FloatArray> = embed(texts)

    /** Output vector dimensionality. */
    public val dimensions: Int

    override fun close(): Unit = Unit
}
