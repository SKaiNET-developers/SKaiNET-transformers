package sk.ainet.llm.providers

import sk.ainet.llm.api.EmbeddingModel
import sk.ainet.llm.api.EmbeddingRequest
import sk.ainet.llm.api.EmbeddingResponse

/**
 * Per-role instruction prefixes for retrieval-tuned embedding models — the
 * asymmetry between "embed this query" and "embed this passage" that models
 * like E5 *require* and BGE recommend for short queries.
 */
public data class EmbeddingPrefixes(
    val query: String,
    val document: String,
) {
    public companion object {
        /** No prefixes — symmetric models (LEAF, MiniLM). */
        public val NONE: EmbeddingPrefixes = EmbeddingPrefixes("", "")
    }
}

/**
 * Decorates an [EmbeddingModel] with per-role prefixes: [embedQuery] prepends
 * [EmbeddingPrefixes.query], [embedDocument]/[embedDocuments] prepend
 * [EmbeddingPrefixes.document]. Untyped calls ([call], [embed]) use the
 * *document* role — indexing is the bulk path, and an unprefixed E5 embedding
 * is wrong for both roles, so defaulting to document keeps `embed(chunks)`
 * correct.
 */
public class PrefixedEmbeddingModel(
    private val delegate: EmbeddingModel,
    private val prefixes: EmbeddingPrefixes,
) : EmbeddingModel {

    override fun call(request: EmbeddingRequest): EmbeddingResponse =
        delegate.call(request.copy(inputs = request.inputs.map { prefixes.document + it }))

    override fun embedQuery(text: String): FloatArray =
        delegate.embed(prefixes.query + text)

    override fun embedDocument(text: String): FloatArray =
        delegate.embed(prefixes.document + text)

    override fun embedDocuments(texts: List<String>): List<FloatArray> =
        delegate.embed(texts.map { prefixes.document + it })

    override val dimensions: Int get() = delegate.dimensions

    override fun close(): Unit = delegate.close()
}

/**
 * Known per-model prefix profiles, keyed by Hugging Face repo-id prefix
 * (case-insensitive). [BertEmbeddingModel.fromHuggingFace] consults this
 * registry automatically; local-snapshot loads pass prefixes explicitly.
 */
public object EmbeddingModelProfiles {

    private val profiles: List<Pair<String, EmbeddingPrefixes>> = listOf(
        // E5: both roles REQUIRE prefixes (intfloat model card).
        "intfloat/multilingual-e5" to EmbeddingPrefixes(query = "query: ", document = "passage: "),
        "intfloat/e5" to EmbeddingPrefixes(query = "query: ", document = "passage: "),
        // BGE v1.5 English: query instruction recommended for short queries; passages plain.
        "BAAI/bge-small-en" to EmbeddingPrefixes(
            query = "Represent this sentence for searching relevant passages: ",
            document = "",
        ),
        "BAAI/bge-base-en" to EmbeddingPrefixes(
            query = "Represent this sentence for searching relevant passages: ",
            document = "",
        ),
        "BAAI/bge-large-en" to EmbeddingPrefixes(
            query = "Represent this sentence for searching relevant passages: ",
            document = "",
        ),
    )

    /** Prefixes for [repoId], or [EmbeddingPrefixes.NONE] when unknown. */
    public fun forRepo(repoId: String): EmbeddingPrefixes =
        profiles.firstOrNull { (prefix, _) -> repoId.startsWith(prefix, ignoreCase = true) }
            ?.second
            ?: EmbeddingPrefixes.NONE

    /** Wrap [model] with the profile for [repoId]; returns [model] unchanged when none applies. */
    public fun apply(model: EmbeddingModel, repoId: String): EmbeddingModel {
        val prefixes = forRepo(repoId)
        return if (prefixes == EmbeddingPrefixes.NONE) model else PrefixedEmbeddingModel(model, prefixes)
    }
}
