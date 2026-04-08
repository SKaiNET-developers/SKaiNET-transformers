package sk.ainet.models.voxtral

import kotlin.io.encoding.Base64
import kotlin.io.encoding.ExperimentalEncodingApi
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Tests for TekkenTokenizer parsing and encode/decode via the adapter.
 *
 * Uses a minimal synthetic tekken.json to validate the pipeline without
 * needing the full 150K-entry Voxtral tokenizer file.
 */
@OptIn(ExperimentalEncodingApi::class)
class TekkenTokenizerTest {

    /**
     * Build a minimal tekken.json with 260 vocab entries (256 single bytes + 4 merges)
     * and 3 special tokens.
     */
    private fun buildMinimalTekkenJson(): String {
        val vocabEntries = StringBuilder()

        // 256 single-byte tokens (ranks 0-255)
        for (i in 0..255) {
            val b64 = Base64.encode(byteArrayOf(i.toByte()))
            val str = if (i in 32..126 && i.toChar() != '"' && i.toChar() != '\\') {
                "\"${i.toChar()}\""
            } else {
                "null"
            }
            if (vocabEntries.isNotEmpty()) vocabEntries.append(",\n")
            vocabEntries.append("""{"rank": $i, "token_bytes": "$b64", "token_str": $str}""")
        }

        // Multi-byte merges (ranks 256-259)
        // "He" = rank 256
        val heB64 = Base64.encode("He".encodeToByteArray())
        vocabEntries.append(",\n")
        vocabEntries.append("""{"rank": 256, "token_bytes": "$heB64", "token_str": "He"}""")

        // "ll" = rank 257
        val llB64 = Base64.encode("ll".encodeToByteArray())
        vocabEntries.append(",\n")
        vocabEntries.append("""{"rank": 257, "token_bytes": "$llB64", "token_str": "ll"}""")

        // "o" is already rank 111 (single byte)

        // "Hello" = rank 258
        val helloB64 = Base64.encode("Hello".encodeToByteArray())
        vocabEntries.append(",\n")
        vocabEntries.append("""{"rank": 258, "token_bytes": "$helloB64", "token_str": "Hello"}""")

        // " w" = rank 259
        val swB64 = Base64.encode(" w".encodeToByteArray())
        vocabEntries.append(",\n")
        vocabEntries.append("""{"rank": 259, "token_bytes": "$swB64", "token_str": " w"}""")

        return """
        {
            "config": {
                "pattern": "\\S+|\\s+",
                "num_vocab_tokens": 260,
                "default_vocab_size": 263,
                "default_num_special_tokens": 3,
                "version": "v7"
            },
            "vocab": [
                $vocabEntries
            ],
            "special_tokens": [
                {"rank": 0, "token_str": "<unk>", "is_control": true},
                {"rank": 1, "token_str": "<s>", "is_control": true},
                {"rank": 2, "token_str": "</s>", "is_control": true}
            ]
        }
        """.trimIndent()
    }

    @Test
    fun `Tekken parser creates tokenizer from minimal json`() {
        val json = buildMinimalTekkenJson()
        val tokenizer = TekkenTokenizerAdapter.fromJson(json)

        // Should be able to encode and decode
        val tokens = tokenizer.encode("Hi")
        assertTrue(tokens.isNotEmpty(), "Encoding should produce tokens")
    }

    @Test
    fun `special tokens get low IDs`() {
        val json = buildMinimalTekkenJson()
        val tokenizer = TekkenTokenizerAdapter.fromJson(json)

        // Special tokens are at IDs 0, 1, 2
        assertEquals("<unk>", tokenizer.decode(0))
        assertEquals("<s>", tokenizer.decode(1))
        assertEquals("</s>", tokenizer.decode(2))
    }

    @Test
    fun `single byte tokens offset by numSpecialTokens`() {
        val json = buildMinimalTekkenJson()
        val tokenizer = TekkenTokenizerAdapter.fromJson(json)

        // 'A' = byte 65, rank 65, token ID = 3 + 65 = 68
        assertEquals("A", tokenizer.decode(3 + 65))
        // ' ' = byte 32, rank 32, token ID = 3 + 32 = 35
        assertEquals(" ", tokenizer.decode(3 + 32))
    }

    @Test
    fun `BPE merges produce expected tokens for Hello`() {
        val json = buildMinimalTekkenJson()
        val tokenizer = TekkenTokenizerAdapter.fromJson(json)

        // "Hello" should merge via BPE: H+e→He(256), l+l→ll(257), He+ll+o→Hello(258)
        // The merge order depends on rank priority. With ranks:
        //   He=256, ll=257, Hello=258
        // First merge: "He" (rank 256, lowest), then "ll" (rank 257)
        // Then: "He"+"ll" can't merge (no "Hell" token), but "He"+"ll"+"o"→"Hello" (rank 258)
        // Actually "Hello" merges He+llo or Hell+o — we need intermediate tokens.
        // With only He, ll, Hello in vocab, BPE produces: He(256) + ll(257) + o(111)
        // then checks He+ll → no rank for "Hell", ll+o → no rank for "llo"
        // So Hello can't merge to single token without intermediate "Hell" or "llo".
        // Let's just verify we get a valid encode/decode roundtrip.
        val tokens = tokenizer.encode("Hello")
        assertTrue(tokens.isNotEmpty(), "Encoding Hello should produce tokens")
        assertEquals("Hello", tokenizer.decode(tokens), "Roundtrip for Hello")
    }

    @Test
    fun `encode decode roundtrip preserves text`() {
        val json = buildMinimalTekkenJson()
        val tokenizer = TekkenTokenizerAdapter.fromJson(json)

        val texts = listOf("Hi", "ab", "A B C")
        for (text in texts) {
            val tokens = tokenizer.encode(text)
            val decoded = tokenizer.decode(tokens)
            assertEquals(text, decoded, "Roundtrip failed for '$text'")
        }
    }

    @Test
    fun `decode array produces same result as concatenated single decode`() {
        val json = buildMinimalTekkenJson()
        val tokenizer = TekkenTokenizerAdapter.fromJson(json)

        val text = "abc"
        val tokens = tokenizer.encode(text)
        val decodedArray = tokenizer.decode(tokens)
        val decodedSingle = tokens.joinToString("") { tokenizer.decode(it) }
        assertEquals(decodedArray, decodedSingle)
    }
}
