package sk.ainet.apps.llm.tokenizer

import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.apps.llm.TokenizerType

class TokenizerStrategyTest {

    // ==================== SentencePiece Strategy Tests ====================

    @Test
    fun `SentencePieceStrategy has correct type`() {
        assertEquals(TokenizerType.SENTENCEPIECE, SentencePieceStrategy.type)
    }

    @Test
    fun `SentencePieceStrategy has correct space marker`() {
        assertEquals("\u2581", SentencePieceStrategy.spaceMarker)
    }

    @Test
    fun `SentencePieceStrategy preprocess adds space marker at start`() {
        val result = SentencePieceStrategy.preprocess("hello")
        assertEquals("\u2581hello", result)
    }

    @Test
    fun `SentencePieceStrategy preprocess replaces spaces with marker`() {
        val result = SentencePieceStrategy.preprocess("hello world")
        assertEquals("\u2581hello\u2581world", result)
    }

    @Test
    fun `SentencePieceStrategy preprocess handles multiple spaces`() {
        val result = SentencePieceStrategy.preprocess("a b c")
        assertEquals("\u2581a\u2581b\u2581c", result)
    }

    @Test
    fun `SentencePieceStrategy postprocess converts marker to space`() {
        val result = SentencePieceStrategy.postprocess("\u2581hello")
        assertEquals(" hello", result)
    }

    @Test
    fun `SentencePieceStrategy postprocess handles standalone marker`() {
        val result = SentencePieceStrategy.postprocess("\u2581")
        assertEquals(" ", result)
    }

    @Test
    fun `SentencePieceStrategy postprocess handles token without marker`() {
        val result = SentencePieceStrategy.postprocess("hello")
        assertEquals("hello", result)
    }

    // ==================== BPE Strategy Tests ====================

    @Test
    fun `BPEStrategy has correct type`() {
        assertEquals(TokenizerType.BPE, BPEStrategy.type)
    }

    @Test
    fun `BPEStrategy has correct space marker`() {
        assertEquals("\u0120", BPEStrategy.spaceMarker)
    }

    @Test
    fun `BPEStrategy preprocess does not add marker at start`() {
        val result = BPEStrategy.preprocess("hello")
        assertEquals("hello", result)
    }

    @Test
    fun `BPEStrategy preprocess replaces spaces with marker`() {
        val result = BPEStrategy.preprocess("hello world")
        assertEquals("hello\u0120world", result)
    }

    @Test
    fun `BPEStrategy preprocess handles leading space`() {
        val result = BPEStrategy.preprocess(" hello")
        assertEquals("\u0120hello", result)
    }

    @Test
    fun `BPEStrategy postprocess converts marker to space`() {
        val result = BPEStrategy.postprocess("\u0120world")
        assertEquals(" world", result)
    }

    @Test
    fun `BPEStrategy postprocess handles standalone marker`() {
        val result = BPEStrategy.postprocess("\u0120")
        assertEquals(" ", result)
    }

    @Test
    fun `BPEStrategy postprocess handles token without marker`() {
        val result = BPEStrategy.postprocess("hello")
        assertEquals("hello", result)
    }

    // ==================== BPEStrategy byte-level decode ====================

    @Test
    fun `BPEStrategy postprocess decodes newline glyph Ċ back to newline byte`() {
        // GPT-2 byte_to_unicode maps byte 0x0A (newline) to U+010A (Ċ).
        // The reverse-mapping in postprocess should restore the newline.
        val result = BPEStrategy.postprocess("Ċ")
        assertEquals("\n", result)
    }

    @Test
    fun `BPEStrategy postprocess decodes tab glyph ĉ back to tab byte`() {
        // GPT-2 byte_to_unicode maps byte 0x09 (tab) to U+0109 (ĉ).
        val result = BPEStrategy.postprocess("ĉ")
        assertEquals("\t", result)
    }

    @Test
    fun `BPEStrategy postprocess decodes mixed byte-level token preserving newlines and spaces`() {
        // The Qwen3 / GPT-2 byte-level encoding of "Hello\nworld" with a
        // leading space → "ĠHelloĊworld". postprocess should produce the
        // original byte sequence.
        val result = BPEStrategy.postprocess("ĠHelloĊworld")
        assertEquals(" Hello\nworld", result)
    }

    @Test
    fun `BPEStrategy postprocess reconstructs multi-byte UTF-8 split across bytes`() {
        // Emoji 🚀 is UTF-8 0xF0 0x9F 0x9A 0x80. GPT-2 byte_to_unicode maps:
        //   0xF0 (240, printable) → 0xF0 = 'ð'
        //   0x9F (159, not printable) → 0x121 + (offset for 159)
        //   0x9A (154, not printable) → 0x121 + (offset for 154)
        //   0x80 (128, not printable) → 0x121 + (offset for 128)
        // Postprocess should reverse each char to its byte and decode the
        // resulting 4-byte sequence as UTF-8, yielding the emoji.
        val encoded = byteArrayOf(0xF0.toByte(), 0x9F.toByte(), 0x9A.toByte(), 0x80.toByte())
            .map { byte ->
                val u = byte.toInt() and 0xFF
                when {
                    u in 33..126 || u in 161..172 || u in 174..255 -> u.toChar()
                    else -> {
                        // Compute the U+0100+offset glyph the encoder would have produced.
                        val missingBefore = (0 until u).count { b ->
                            !(b in 33..126 || b in 161..172 || b in 174..255)
                        }
                        (256 + missingBefore).toChar()
                    }
                }
            }
            .joinToString("")
        val result = BPEStrategy.postprocess(encoded)
        assertEquals("🚀", result)  // 🚀 surrogate pair
    }

    @Test
    fun `BPEStrategy postprocess falls back to identity-with-space-marker on non-byte-level input`() {
        // SentencePiece-style ▁ (U+2581) is outside the byte_to_unicode
        // alphabet; postprocess should not garble it, just pass through
        // (with the space marker still reversed if present).
        assertEquals("▁foo", BPEStrategy.postprocess("▁foo"))
        assertEquals(" foo▁bar", BPEStrategy.postprocess("Ġfoo▁bar"))
    }

    // ==================== WordPiece Strategy Tests ====================

    @Test
    fun `WordPieceStrategy has correct type`() {
        assertEquals(TokenizerType.WORDPIECE, WordPieceStrategy.type)
    }

    @Test
    fun `WordPieceStrategy has correct continuation marker`() {
        assertEquals("##", WordPieceStrategy.spaceMarker)
    }

    @Test
    fun `WordPieceStrategy preprocess returns text unchanged`() {
        val result = WordPieceStrategy.preprocess("hello world")
        assertEquals("hello world", result)
    }

    @Test
    fun `WordPieceStrategy postprocess removes continuation prefix`() {
        val result = WordPieceStrategy.postprocess("##ing")
        assertEquals("ing", result)
    }

    @Test
    fun `WordPieceStrategy postprocess keeps non-continuation token unchanged`() {
        val result = WordPieceStrategy.postprocess("hello")
        assertEquals("hello", result)
    }

    @Test
    fun `WordPieceStrategy postprocess handles hash in middle of token`() {
        val result = WordPieceStrategy.postprocess("a##b")
        assertEquals("a##b", result)
    }

    // ==================== Unknown Strategy Tests ====================

    @Test
    fun `UnknownStrategy has correct type`() {
        assertEquals(TokenizerType.UNKNOWN, UnknownStrategy.type)
    }

    @Test
    fun `UnknownStrategy defaults to SentencePiece behavior`() {
        // Should behave like SentencePiece
        assertEquals("\u2581", UnknownStrategy.spaceMarker)
        assertEquals("\u2581hello", UnknownStrategy.preprocess("hello"))
        assertEquals(" hello", UnknownStrategy.postprocess("\u2581hello"))
    }
}
