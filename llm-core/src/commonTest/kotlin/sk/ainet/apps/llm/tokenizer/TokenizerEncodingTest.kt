package sk.ainet.apps.llm.tokenizer

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Tests for the actual encoding/decoding logic with mock vocabularies.
 * These tests verify the BPE merge algorithm and strategy integration.
 */
class TokenizerEncodingTest {

    /**
     * A minimal mock tokenizer for testing encoding logic.
     * Scores control merge priority - higher score = merge first.
     */
    class MockBPETokenizer(
        private val vocab: List<String>,
        private val strategy: sk.ainet.apps.llm.TokenizerStrategy,
        private val scores: FloatArray = FloatArray(vocab.size) { 0f }
    ) {
        private val tokenToId: Map<String, Int> = vocab.mapIndexed { idx, token -> token to idx }.toMap()

        fun encode(text: String): IntArray {
            if (text.isEmpty()) return intArrayOf()

            val preprocessed = strategy.preprocess(text)
            return encodeBPE(preprocessed)
        }

        private fun encodeBPE(preprocessed: String): IntArray {
            val tokens = preprocessed.map { it.toString() }.toMutableList()

            var changed = true
            while (changed && tokens.size > 1) {
                changed = false
                var bestIdx = -1
                var bestScore = Float.NEGATIVE_INFINITY
                var bestMerge = ""

                for (i in 0 until tokens.size - 1) {
                    val merge = tokens[i] + tokens[i + 1]
                    val tokenId = tokenToId[merge]
                    if (tokenId != null) {
                        val score = scores[tokenId]
                        if (score > bestScore) {
                            bestScore = score
                            bestIdx = i
                            bestMerge = merge
                        }
                    }
                }

                if (bestIdx >= 0) {
                    tokens[bestIdx] = bestMerge
                    tokens.removeAt(bestIdx + 1)
                    changed = true
                }
            }

            return tokens.map { token ->
                tokenToId[token] ?: -1
            }.toIntArray()
        }

        fun decode(ids: IntArray): String {
            return ids.joinToString("") { id ->
                val token = vocab.getOrNull(id) ?: ""
                strategy.postprocess(token)
            }
        }
    }

    // ==================== SentencePiece Encoding Tests ====================

    @Test
    fun `SentencePiece preprocess adds leading space marker`() {
        val result = SentencePieceStrategy.preprocess("hi")
        assertEquals("\u2581hi", result)
    }

    @Test
    fun `SentencePiece encode builds up word through merges`() {
        // Vocab with merge chain: ▁ + h -> ▁h, then ▁h + i -> ▁hi
        val vocab = listOf(
            "\u2581",      // 0: space marker
            "h",           // 1
            "i",           // 2
            "\u2581h",     // 3: merged ▁+h
            "\u2581hi"     // 4: merged ▁h+i
        )
        // Scores: higher = merge first
        // ▁h should merge before ▁hi (we need ▁h to exist first)
        val scores = floatArrayOf(0f, 0f, 0f, 10f, 5f)
        val tokenizer = MockBPETokenizer(vocab, SentencePieceStrategy, scores)

        val ids = tokenizer.encode("hi")
        // "hi" -> preprocess -> "▁hi" -> chars ["▁", "h", "i"]
        // Step 1: merge ▁+h -> ▁h (score 10)
        // Step 2: merge ▁h+i -> ▁hi (score 5)
        assertEquals(1, ids.size)
        assertEquals(4, ids[0])  // ▁hi
    }

    @Test
    fun `SentencePiece encode with spaces between words`() {
        val vocab = listOf(
            "\u2581",      // 0: space marker
            "a",           // 1
            "b",           // 2
            "\u2581a",     // 3: merged
            "\u2581b"      // 4: merged
        )
        val scores = floatArrayOf(0f, 0f, 0f, 10f, 10f)
        val tokenizer = MockBPETokenizer(vocab, SentencePieceStrategy, scores)

        val ids = tokenizer.encode("a b")
        // "a b" -> preprocess -> "▁a▁b" -> ["▁", "a", "▁", "b"]
        // Merges: ▁+a -> ▁a, ▁+b -> ▁b
        assertEquals(2, ids.size)
        assertEquals(3, ids[0])  // ▁a
        assertEquals(4, ids[1])  // ▁b
    }

    @Test
    fun `SentencePiece decode converts markers to spaces`() {
        val vocab = listOf("\u2581hello", "\u2581world")
        val tokenizer = MockBPETokenizer(vocab, SentencePieceStrategy)

        val text = tokenizer.decode(intArrayOf(0, 1))
        assertEquals(" hello world", text)
    }

    @Test
    fun `SentencePiece partial merge when full word not in vocab`() {
        val vocab = listOf(
            "\u2581",      // 0
            "c",           // 1
            "a",           // 2
            "t",           // 3
            "\u2581c",     // 4: merged
            "at"           // 5: merged
        )
        val scores = floatArrayOf(0f, 0f, 0f, 0f, 10f, 8f)
        val tokenizer = MockBPETokenizer(vocab, SentencePieceStrategy, scores)

        val ids = tokenizer.encode("cat")
        // "cat" -> "▁cat" -> ["▁", "c", "a", "t"]
        // Step 1: ▁+c -> ▁c (score 10)
        // Step 2: a+t -> at (score 8)
        // Result: ["▁c", "at"]
        assertEquals(2, ids.size)
        assertEquals(4, ids[0])  // ▁c
        assertEquals(5, ids[1])  // at
    }

    // ==================== BPE Encoding Tests ====================

    @Test
    fun `BPE preprocess does not add leading marker`() {
        val result = BPEStrategy.preprocess("hello")
        assertEquals("hello", result)
    }

    @Test
    fun `BPE preprocess replaces spaces with G-dot marker`() {
        val result = BPEStrategy.preprocess("hi there")
        assertEquals("hi\u0120there", result)
    }

    @Test
    fun `BPE encode simple word`() {
        val vocab = listOf(
            "h",           // 0
            "i",           // 1
            "hi"           // 2: merged
        )
        val scores = floatArrayOf(0f, 0f, 10f)
        val tokenizer = MockBPETokenizer(vocab, BPEStrategy, scores)

        val ids = tokenizer.encode("hi")
        // "hi" -> preprocess -> "hi" (no change) -> ["h", "i"]
        // Merge: h+i -> hi
        assertEquals(1, ids.size)
        assertEquals(2, ids[0])  // hi
    }

    @Test
    fun `BPE encode with space marker`() {
        val vocab = listOf(
            "h",           // 0
            "i",           // 1
            "\u0120",      // 2: Ġ (space marker)
            "t",           // 3
            "hi",          // 4: merged
            "\u0120t"      // 5: merged space+t
        )
        val scores = floatArrayOf(0f, 0f, 0f, 0f, 10f, 8f)
        val tokenizer = MockBPETokenizer(vocab, BPEStrategy, scores)

        val ids = tokenizer.encode("hi t")
        // "hi t" -> "hiĠt" -> ["h", "i", "Ġ", "t"]
        // Step 1: h+i -> hi (score 10)
        // Step 2: Ġ+t -> Ġt (score 8)
        assertEquals(2, ids.size)
        assertEquals(4, ids[0])  // hi
        assertEquals(5, ids[1])  // Ġt
    }

    @Test
    fun `BPE decode converts G-dot to space`() {
        val vocab = listOf("hello", "\u0120world")
        val tokenizer = MockBPETokenizer(vocab, BPEStrategy)

        val text = tokenizer.decode(intArrayOf(0, 1))
        assertEquals("hello world", text)
    }

    // ==================== WordPiece Tests ====================

    @Test
    fun `WordPiece decode removes continuation markers`() {
        val vocab = listOf("un", "##believ", "##able")
        val tokenizer = MockBPETokenizer(vocab, WordPieceStrategy)

        val text = tokenizer.decode(intArrayOf(0, 1, 2))
        assertEquals("unbelievable", text)
    }

    @Test
    fun `WordPiece preprocess leaves text unchanged`() {
        val result = WordPieceStrategy.preprocess("hello world")
        assertEquals("hello world", result)
    }

    // ==================== Edge Cases ====================

    @Test
    fun `encode empty string returns empty array`() {
        val vocab = listOf("\u2581hello")
        val tokenizer = MockBPETokenizer(vocab, SentencePieceStrategy)

        val ids = tokenizer.encode("")
        assertTrue(ids.isEmpty())
    }

    @Test
    fun `encode unknown chars returns individual tokens or -1`() {
        val vocab = listOf("\u2581", "a")
        val tokenizer = MockBPETokenizer(vocab, SentencePieceStrategy)

        val ids = tokenizer.encode("ab")
        // "ab" -> "▁ab" -> ["▁", "a", "b"]
        // ▁ found (0), a found (1), b not found (-1)
        assertEquals(3, ids.size)
        assertEquals(0, ids[0])
        assertEquals(1, ids[1])
        assertEquals(-1, ids[2])  // unknown
    }

    @Test
    fun `BPE merge order respects scores`() {
        val vocab = listOf(
            "a",    // 0
            "b",    // 1
            "c",    // 2
            "ab",   // 3: score 5 (lower priority)
            "bc"    // 4: score 10 (higher priority)
        )
        val scores = floatArrayOf(0f, 0f, 0f, 5f, 10f)
        val tokenizer = MockBPETokenizer(vocab, BPEStrategy, scores)

        val ids = tokenizer.encode("abc")
        // "abc" -> ["a", "b", "c"]
        // bc has higher score (10) than ab (5), so bc merges first
        // Step 1: b+c -> bc (score 10)
        // Result: ["a", "bc"]
        assertEquals(2, ids.size)
        assertEquals(0, ids[0])  // a
        assertEquals(4, ids[1])  // bc
    }

    // ==================== Round-trip Tests ====================

    @Test
    fun `SentencePiece round trip`() {
        val vocab = listOf(
            "\u2581",      // 0
            "t",           // 1
            "h",           // 2
            "e",           // 3
            "\u2581t",     // 4
            "\u2581th",    // 5
            "\u2581the"    // 6
        )
        val scores = floatArrayOf(0f, 0f, 0f, 0f, 8f, 9f, 10f)
        val tokenizer = MockBPETokenizer(vocab, SentencePieceStrategy, scores)

        val ids = tokenizer.encode("the")
        val decoded = tokenizer.decode(ids)

        // Encoded as single token ▁the
        assertEquals(1, ids.size)
        assertEquals(6, ids[0])
        // Decoded with leading space (SentencePiece behavior)
        assertEquals(" the", decoded)
    }

    // ==================== Special-token (CONTROL) Encoding ====================
    //
    // GGUF marks atomic special tokens (e.g. Llama 3's <|begin_of_text|>, <|eot_id|>;
    // ChatML's <|im_start|>) with token_type == 3 (CONTROL). These must be emitted
    // as a single atomic token ID even though their multi-character string would never
    // be assembled by greedy bottom-up BPE merging from single-char starting tokens.

    @Test
    fun `BPE encode emits CONTROL tokens atomically not as char fragments`() {
        // Vocab: byte-level chars + a CONTROL special token.
        // BPE has no merge chain that could ever assemble "<|begin|>" from chars,
        // so without special handling encoding it would yield ~9 fragment IDs.
        val vocab = listOf(
            "<", "|", "b", "e", "g", "i", "n", ">",  // 0..7  single chars
            "<|begin|>",                              // 8     CONTROL
            "h", "l", "o"                             // 9..11 single chars
        )
        val scores = FloatArray(vocab.size) { 0f }
        val tokenTypes = IntArray(vocab.size) { 1 }   // 1 = NORMAL
        tokenTypes[8] = 3                             // 3 = CONTROL

        val tok = GGUFTokenizer.forTesting(
            vocab = vocab,
            scores = scores,
            bosTokenId = 8,
            eosTokenId = 8,
            unkTokenId = 0,
            strategy = BPEStrategy,
            tokenTypes = tokenTypes
        )

        val ids = tok.encode("<|begin|>hello")
        assertEquals(8, ids[0], "first token must be the atomic <|begin|> id (8), not a fragment of '<'")
        // Remaining ids should encode "hello" — exact tokenization is irrelevant here;
        // what matters is that the special token didn't get char-split.
        assertTrue(ids.size < 9, "expected ~5 ids (1 special + 'hello' chars), got ${ids.size}: ${ids.toList()}")
    }

    @Test
    fun `BPE encode finds CONTROL token in middle of plain text`() {
        val vocab = listOf(
            "h", "i",                  // 0..1
            "<|stop|>",                // 2     CONTROL
            "b", "y", "e"              // 3..5
        )
        val scores = FloatArray(vocab.size) { 0f }
        val tokenTypes = IntArray(vocab.size) { 1 }
        tokenTypes[2] = 3

        val tok = GGUFTokenizer.forTesting(
            vocab = vocab,
            scores = scores,
            bosTokenId = 2,
            eosTokenId = 2,
            unkTokenId = 0,
            strategy = BPEStrategy,
            tokenTypes = tokenTypes
        )

        val ids = tok.encode("hi<|stop|>bye")
        // Atomic stop token must appear exactly once, and only once.
        val stopOccurrences = ids.count { it == 2 }
        assertEquals(1, stopOccurrences, "atomic <|stop|> must appear exactly once, got ids=${ids.toList()}")
    }

    @Test
    fun `BPE encode handles two adjacent CONTROL tokens`() {
        val vocab = listOf(
            "<|a|>",   // 0  CONTROL
            "<|b|>",   // 1  CONTROL
            "x"        // 2
        )
        val scores = FloatArray(vocab.size) { 0f }
        val tokenTypes = intArrayOf(3, 3, 1)

        val tok = GGUFTokenizer.forTesting(
            vocab = vocab,
            scores = scores,
            bosTokenId = 0,
            eosTokenId = 1,
            unkTokenId = 2,
            strategy = BPEStrategy,
            tokenTypes = tokenTypes
        )

        val ids = tok.encode("<|a|><|b|>x")
        // Adjacent specials should yield exactly [0, 1, 2] — no char fragments between them.
        assertEquals(listOf(0, 1, 2), ids.toList())
    }

    @Test
    fun `BPE encode emits USER_DEFINED tokens atomically (Gemma 4 tool markers)`() {
        // Regression for the Gemma 4 multi-turn tool-calling case. GGUF type=4
        // (USER_DEFINED) marks visible-but-atomic tokens — `convert_hf_to_gguf.py`
        // Gemma4Model.set_vocab assigns this type to <|tool_call> / <tool_call|> /
        // <|tool_response> / <tool_response|> "so that the chat parser can read
        // them". They must round-trip through encode the same as type=3 CONTROL.
        val vocab = listOf(
            "<", "|", "t", "o", "_", "c", "a", "l", "r", "e", "s", "p", "n", ">",  // 0..13 single chars
            "<|tool_call>",      // 14  USER_DEFINED
            "<tool_call|>",      // 15  USER_DEFINED
            "<|tool_response>",  // 16  USER_DEFINED
            "<tool_response|>",  // 17  USER_DEFINED
            "x"                  // 18
        )
        val scores = FloatArray(vocab.size) { 0f }
        val tokenTypes = IntArray(vocab.size) { 1 }
        tokenTypes[14] = 4 // USER_DEFINED
        tokenTypes[15] = 4
        tokenTypes[16] = 4
        tokenTypes[17] = 4

        val tok = GGUFTokenizer.forTesting(
            vocab = vocab,
            scores = scores,
            bosTokenId = 14,
            eosTokenId = 15,
            unkTokenId = 0,
            strategy = BPEStrategy,
            tokenTypes = tokenTypes
        )

        val ids = tok.encode("<|tool_call>x<tool_call|>")
        // First and last must be the atomic IDs, not char-fragmented.
        assertEquals(14, ids.first(), "first token must be atomic <|tool_call> id (14)")
        assertEquals(15, ids.last(), "last token must be atomic <tool_call|> id (15)")
        assertTrue(ids.size <= 3, "expected ≤3 ids ([14, 18, 15]), got ${ids.toList()}")

        // Multi-turn case: tool_response in conversation history.
        val ids2 = tok.encode("<|tool_response>x<tool_response|>")
        assertEquals(16, ids2.first())
        assertEquals(17, ids2.last())
    }

    @Test
    fun `BPE encode without tokenTypes falls back to plain BPE`() {
        // No tokenTypes provided → no special handling, plain BPE on chars.
        // Guards against accidental behavior change for callers that don't pass tokenTypes.
        val vocab = listOf("h", "i")
        val scores = floatArrayOf(0f, 0f)

        val tok = GGUFTokenizer.forTesting(
            vocab = vocab,
            scores = scores,
            bosTokenId = 0,
            eosTokenId = 1,
            unkTokenId = 0,
            strategy = BPEStrategy,
            tokenTypes = null
        )

        val ids = tok.encode("hi")
        assertEquals(listOf(0, 1), ids.toList())
    }

    @Test
    fun `BPE round trip`() {
        val vocab = listOf(
            "t",           // 0
            "h",           // 1
            "e",           // 2
            "th",          // 3
            "the"          // 4
        )
        val scores = floatArrayOf(0f, 0f, 0f, 8f, 10f)
        val tokenizer = MockBPETokenizer(vocab, BPEStrategy, scores)

        val ids = tokenizer.encode("the")
        val decoded = tokenizer.decode(ids)

        assertEquals(1, ids.size)
        assertEquals(4, ids[0])
        assertEquals("the", decoded)
    }
}
