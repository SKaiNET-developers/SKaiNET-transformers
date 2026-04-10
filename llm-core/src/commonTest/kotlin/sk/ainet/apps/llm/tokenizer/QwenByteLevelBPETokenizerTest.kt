package sk.ainet.apps.llm.tokenizer

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class QwenByteLevelBPETokenizerTest {

    // --- Byte-to-Unicode mapping ---

    @Test
    fun byteToUnicodeMapHas256Entries() {
        assertEquals(256, QwenByteLevelBPETokenizer.BYTE_TO_UNICODE.size)
    }

    @Test
    fun byteToUnicodeAllUnique() {
        val chars = QwenByteLevelBPETokenizer.BYTE_TO_UNICODE.toSet()
        assertEquals(256, chars.size, "All 256 byte mappings must be unique")
    }

    @Test
    fun byteToUnicodePrintableAsciiIdentity() {
        // Printable ASCII: '!' (33) to '~' (126) map to themselves
        for (b in 33..126) {
            assertEquals(
                b.toChar(),
                QwenByteLevelBPETokenizer.BYTE_TO_UNICODE[b],
                "Byte $b should map to char '${b.toChar()}'"
            )
        }
    }

    @Test
    fun byteToUnicodeControlCharsRemapped() {
        // Byte 0 (NUL) should NOT map to char 0 — it should be remapped
        assertNotEquals('\u0000', QwenByteLevelBPETokenizer.BYTE_TO_UNICODE[0])
        // Space (32) is not in the printable range, so it should be remapped
        assertNotEquals(' ', QwenByteLevelBPETokenizer.BYTE_TO_UNICODE[32])
    }

    @Test
    fun unicodeToByteRoundTrip() {
        for (b in 0..255) {
            val unicode = QwenByteLevelBPETokenizer.BYTE_TO_UNICODE[b]
            val byteBack = QwenByteLevelBPETokenizer.UNICODE_TO_BYTE[unicode]
            assertEquals(b.toByte(), byteBack, "Round-trip failed for byte $b")
        }
    }

    // --- Tokenizer with minimal vocab ---

    /**
     * Build a minimal tokenizer for testing basic functionality.
     */
    private fun buildMinimalTokenizer(
        extraVocab: Map<String, Int> = emptyMap(),
        extraMerges: List<String> = emptyList(),
        specialTokens: Map<String, Int> = emptyMap(),
        pretokenizeRegex: String? = null
    ): QwenByteLevelBPETokenizer {
        // Start with byte-level base vocab (256 single-char tokens)
        val vocabMap = mutableMapOf<String, Int>()
        for (b in 0..255) {
            val ch = QwenByteLevelBPETokenizer.BYTE_TO_UNICODE[b].toString()
            vocabMap[ch] = b
        }
        vocabMap.putAll(extraVocab)
        vocabMap.putAll(specialTokens)

        val vocabSize = vocabMap.values.maxOrNull()?.let { it + 1 } ?: 256
        val vocab = MutableList(vocabSize) { "" }
        vocabMap.forEach { (token, id) -> if (id < vocabSize) vocab[id] = token }

        val mergeRanks = mutableMapOf<String, Int>()
        extraMerges.forEachIndexed { idx, merge ->
            mergeRanks[merge] = idx
        }

        return QwenByteLevelBPETokenizer(
            vocab = vocab,
            tokenToId = vocabMap,
            mergeRanks = mergeRanks,
            specialTokens = specialTokens,
            pretokenizeRegex = pretokenizeRegex?.let { Regex(it) },
            eosTokenId = specialTokens.values.firstOrNull() ?: 0,
            addEosToken = false,
            addBosToken = false,
            bosTokenId = 0
        )
    }

    @Test
    fun encodeEmptyString() {
        val tokenizer = buildMinimalTokenizer()
        val tokens = tokenizer.encode("")
        assertTrue(tokens.isEmpty())
    }

    @Test
    fun encodeSingleAsciiChar() {
        val tokenizer = buildMinimalTokenizer()
        val tokens = tokenizer.encode("A")
        assertEquals(1, tokens.size)
        // 'A' is byte 65, which maps to 'A' in byte-to-unicode, stored at index 65
        assertEquals(65, tokens[0])
    }

    @Test
    fun decodeRoundTripSimpleAscii() {
        val tokenizer = buildMinimalTokenizer()
        val text = "Hello"
        val encoded = tokenizer.encode(text)
        val decoded = tokenizer.decode(encoded)
        assertEquals(text, decoded)
    }

    @Test
    fun specialTokensAtomic() {
        val specialTokens = mapOf("<|im_start|>" to 256, "<|im_end|>" to 257)
        val tokenizer = buildMinimalTokenizer(specialTokens = specialTokens)

        val tokens = tokenizer.encode("<|im_start|>hello<|im_end|>")
        // Should be: special_256, h, e, l, l, o, special_257
        assertEquals(256, tokens.first())
        assertEquals(257, tokens.last())
        assertEquals(7, tokens.size) // 1 + 5 + 1
    }

    @Test
    fun specialTokenDecodePreservesMarkers() {
        val specialTokens = mapOf("<|im_start|>" to 256, "<|im_end|>" to 257)
        val tokenizer = buildMinimalTokenizer(specialTokens = specialTokens)

        val text = "<|im_start|>Hi<|im_end|>"
        val encoded = tokenizer.encode(text)
        val decoded = tokenizer.decode(encoded)
        assertEquals(text, decoded)
    }

    @Test
    fun bpeMergesApplied() {
        // Create merge for "He" (byte-level chars for 'H' and 'e')
        val hChar = QwenByteLevelBPETokenizer.BYTE_TO_UNICODE['H'.code].toString()
        val eChar = QwenByteLevelBPETokenizer.BYTE_TO_UNICODE['e'.code].toString()
        val merged = hChar + eChar

        val tokenizer = buildMinimalTokenizer(
            extraVocab = mapOf(merged to 256),
            extraMerges = listOf(merged)
        )

        val tokens = tokenizer.encode("He")
        // Should merge "H" + "e" into single token 256
        assertEquals(1, tokens.size)
        assertEquals(256, tokens[0])
    }

    @Test
    fun pretokenizeRegexSplitsText() {
        // Simple regex that splits on spaces
        val tokenizer = buildMinimalTokenizer(
            pretokenizeRegex = "\\S+"
        )

        val tokens = tokenizer.encode("A B")
        // "A B" pretokenized to ["A", "B"] (spaces dropped by regex)
        // Each encodes as a single byte token
        assertEquals(2, tokens.size)
    }

    @Test
    fun factoryDetectsQwen2Tokenizer() {
        val config = """{"tokenizer_class": "Qwen2Tokenizer"}"""
        assertTrue(isQwen2Tokenizer(config))
    }

    @Test
    fun factoryRejectsNonQwen2Tokenizer() {
        val config = """{"tokenizer_class": "LlamaTokenizer"}"""
        assertTrue(!isQwen2Tokenizer(config))
    }
}
