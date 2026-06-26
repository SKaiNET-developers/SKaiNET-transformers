package sk.ainet.apps.llm.tokenizer

import sk.ainet.io.tokenizer.SentencePieceTokenizer
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse

/**
 * Streaming detokenization regression: a generation loop appends one decoded
 * token at a time (`response.append(tokenizer.decode(tokenId))`). Each
 * SentencePiece piece carries its own leading word-boundary space, so
 * per-token decode must NOT strip it — otherwise words run together
 * (`"the process"` → `"theprocess"`). See `SentencePieceTokenizer.decodeToken`.
 */
class SentencePieceSpecialTokensStreamingTest {

    private fun toy(): SentencePieceSpecialTokens {
        // Minimal vocab: control tokens + two ▁-prefixed word pieces.
        val tokens = listOf("<unk>", "<s>", "</s>", "▁Hello", "▁world")
        val scores = List(tokens.size) { 0.0f }
        val base = SentencePieceTokenizer(
            tokens = tokens,
            scores = scores,
            unknownTokenId = 0,
            bosTokenId = 1,
            eosTokenId = 2,
            addSpacePrefix = true,
        )
        return SentencePieceSpecialTokens(base, specialTokens = emptyMap())
    }

    @Test
    fun `streaming per-token decode preserves word spaces`() {
        val tok = toy()
        val ids = intArrayOf(3, 4) // ▁Hello, ▁world

        val streamed = buildString { for (id in ids) append(tok.decode(id)) }
        assertEquals(" Hello world", streamed)
        assertFalse(streamed.contains("Helloworld"), "words must not run together")
    }

    @Test
    fun `batch decode still strips the single leading space`() {
        val tok = toy()
        assertEquals("Hello world", tok.decode(intArrayOf(3, 4)))
    }
}
