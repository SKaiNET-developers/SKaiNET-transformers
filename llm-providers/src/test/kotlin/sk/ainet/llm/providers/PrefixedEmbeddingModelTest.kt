package sk.ainet.llm.providers

import sk.ainet.llm.api.Embedding
import sk.ainet.llm.api.EmbeddingModel
import sk.ainet.llm.api.EmbeddingRequest
import sk.ainet.llm.api.EmbeddingResponse
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame

class PrefixedEmbeddingModelTest {

    /** Records every text handed to the delegate. */
    private class RecordingModel : EmbeddingModel {
        val seen = mutableListOf<String>()
        override fun call(request: EmbeddingRequest): EmbeddingResponse {
            seen += request.inputs
            return EmbeddingResponse(
                request.inputs.mapIndexed { i, _ -> Embedding(i, floatArrayOf(1f)) },
            )
        }
        override val dimensions: Int = 1
    }

    private val e5 = EmbeddingPrefixes(query = "query: ", document = "passage: ")

    @Test
    fun embedQuery_prependsQueryPrefix() {
        val delegate = RecordingModel()
        PrefixedEmbeddingModel(delegate, e5).embedQuery("what is a pangram?")
        assertEquals(listOf("query: what is a pangram?"), delegate.seen)
    }

    @Test
    fun embedDocument_andBatch_prependDocumentPrefix() {
        val delegate = RecordingModel()
        val model = PrefixedEmbeddingModel(delegate, e5)
        model.embedDocument("a")
        model.embedDocuments(listOf("b", "c"))
        assertEquals(listOf("passage: a", "passage: b", "passage: c"), delegate.seen)
    }

    @Test
    fun untypedCalls_useDocumentRole() {
        val delegate = RecordingModel()
        val model = PrefixedEmbeddingModel(delegate, e5)
        model.embed("a")
        model.call(EmbeddingRequest(listOf("b")))
        assertEquals(listOf("passage: a", "passage: b"), delegate.seen)
    }

    @Test
    fun profiles_e5AndBgeResolve_caseInsensitive() {
        assertEquals("query: ", EmbeddingModelProfiles.forRepo("intfloat/multilingual-e5-small").query)
        assertEquals("passage: ", EmbeddingModelProfiles.forRepo("INTFLOAT/Multilingual-E5-Large").document)
        val bge = EmbeddingModelProfiles.forRepo("BAAI/bge-small-en-v1.5")
        assertEquals("Represent this sentence for searching relevant passages: ", bge.query)
        assertEquals("", bge.document)
    }

    @Test
    fun profiles_unknownRepo_returnsModelUnwrapped() {
        assertEquals(EmbeddingPrefixes.NONE, EmbeddingModelProfiles.forRepo("MongoDB/mdbr-leaf-ir"))
        val delegate = RecordingModel()
        assertSame(delegate, EmbeddingModelProfiles.apply(delegate, "MongoDB/mdbr-leaf-ir"))
    }
}
