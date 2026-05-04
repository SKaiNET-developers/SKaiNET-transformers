package sk.ainet.apps.llm.tokenizer

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class UpstreamTokenizerAdapterTest {

    /** Stub of the upstream Tokenizer interface for adapter contract tests. */
    private class StubUpstreamTokenizer(
        override val vocabSize: Int,
        override val bosTokenId: Int? = null,
        override val eosTokenId: Int? = null,
        private val encodeFn: (String) -> IntArray,
        private val decodeFn: (IntArray) -> String,
    ) : sk.ainet.io.tokenizer.Tokenizer {
        override fun encode(text: String): IntArray = encodeFn(text)
        override fun decode(ids: IntArray): String = decodeFn(ids)
    }

    @Test
    fun `delegates encode and decode`() {
        val stub = StubUpstreamTokenizer(
            vocabSize = 100,
            bosTokenId = 1,
            eosTokenId = 2,
            encodeFn = { text -> intArrayOf(text.length, 99) },
            decodeFn = { ids -> "decoded(${ids.joinToString(",")})" },
        )
        val adapter = UpstreamTokenizerAdapter(stub)

        assertEquals(100, adapter.vocabSize)
        assertEquals(1, adapter.bosTokenId)
        assertEquals(2, adapter.eosTokenId)

        val encoded = adapter.encode("hello")
        assertEquals(2, encoded.size)
        assertEquals(5, encoded[0])
        assertEquals(99, encoded[1])

        assertEquals("decoded(7,8)", adapter.decode(intArrayOf(7, 8)))
    }

    @Test
    fun `single-token decode wraps in IntArray`() {
        var capturedIds: IntArray? = null
        val stub = StubUpstreamTokenizer(
            vocabSize = 10,
            encodeFn = { intArrayOf() },
            decodeFn = { ids -> capturedIds = ids; "" },
        )
        val adapter = UpstreamTokenizerAdapter(stub)

        adapter.decode(42)

        assertTrue(capturedIds != null && capturedIds!!.size == 1)
        assertEquals(42, capturedIds!![0])
    }

    @Test
    fun `null bos and eos fall back to defaults`() {
        val stub = StubUpstreamTokenizer(
            vocabSize = 10,
            bosTokenId = null,
            eosTokenId = null,
            encodeFn = { intArrayOf() },
            decodeFn = { "" },
        )
        // Defaults: bos = 1, eos = 2.
        val adapter = UpstreamTokenizerAdapter(stub)
        assertEquals(1, adapter.bosTokenId)
        assertEquals(2, adapter.eosTokenId)
    }

    @Test
    fun `null bos and eos honour custom fallbacks`() {
        val stub = StubUpstreamTokenizer(
            vocabSize = 10,
            bosTokenId = null,
            eosTokenId = null,
            encodeFn = { intArrayOf() },
            decodeFn = { "" },
        )
        val adapter = UpstreamTokenizerAdapter(stub, bosTokenIdFallback = 7, eosTokenIdFallback = 13)
        assertEquals(7, adapter.bosTokenId)
        assertEquals(13, adapter.eosTokenId)
    }
}
