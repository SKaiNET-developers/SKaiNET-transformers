package sk.ainet.apps.kgemma

import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.io.path.isDirectory
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.apps.llm.tokenizer.GGUFTokenizer

/**
 * Fast (no-model-load) regression guard for the tokenizer fixes that
 * landed alongside the SafeTensors path:
 *
 *  - `fromTokenizerJson` populates `tokenTypes` from `added_tokens` so
 *    chat-template control tokens (`<bos>`, `<|turn>`, `<turn|>`, etc.)
 *    encode atomically rather than splitting into per-character BPE
 *    pieces. The 1f20c27 / 444fd17 commits.
 *  - SentencePiece marker (`▁`) auto-detection picks
 *    `SentencePieceStrategy` for Gemma-style vocabs and disables
 *    `add_dummy_prefix` so encoding "Hi" returns id 10979 (not the
 *    `▁Hi` variant 18428). 1f20c27.
 *
 * Asserts byte-equality with the HuggingFace `tokenizers` Python lib
 * reference for a chat-template-rendered single-user-turn prompt.
 *
 * Self-skips if `GEMMA4_E2B_SAFETENSORS_PATH` is unset — we only need
 * the model dir to read `tokenizer.json`, not the SafeTensors weights.
 */
class Gemma4TokenizerParityTest {

    @Test
    fun `chat template rendered prompt tokenizes to HF reference ids`() {
        val tokenizerJson = locateTokenizerJson() ?: return
        val tokenizer = GGUFTokenizer.fromTokenizerJson(Files.readString(tokenizerJson))

        // Exact rendered output of `Gemma4ChatTemplate.apply()` for a single
        // user turn with the canonical prompt. Hand-built here so the test
        // doesn't depend on the chat-template module — that way we're
        // verifying the tokenizer in isolation.
        val prompt = "<bos><|turn>user\nSay hello in one short sentence.<turn|>\n<|turn>model\n"

        // Reference IDs produced by HuggingFace `tokenizers` Python lib
        // against the same tokenizer.json (see commit message on 444fd17
        // for the diagnostic trace). Locked to byte-equality.
        val expected = intArrayOf(
            2,        // <bos>
            105,      // <|turn>
            2364,     // user
            107,      // \n
            37889,    // Say
            29104,    //  hello
            528,      //  in
            886,      //  one
            2822,     //  short
            13315,    //  sentence
            236761,   // .
            106,      // <turn|>
            107,      // \n
            105,      // <|turn>
            4368,     // model
            107,      // \n
        )

        val actual = tokenizer.encode(prompt)
        assertEquals(
            expected.toList(), actual.toList(),
            "Tokenized prompt diverged from HF reference. Got ${actual.size} tokens, " +
                "expected ${expected.size}.",
        )
    }

    @Test
    fun `single-token specials encode atomically with the right ids`() {
        val tokenizerJson = locateTokenizerJson() ?: return
        val tokenizer = GGUFTokenizer.fromTokenizerJson(Files.readString(tokenizerJson))

        val cases = listOf(
            "<bos>" to 2,
            "<eos>" to 1,
            "<pad>" to 0,
            "<|turn>" to 105,
            "<turn|>" to 106,
        )
        for ((literal, expectedId) in cases) {
            val encoded = tokenizer.encode(literal)
            assertEquals(
                listOf(expectedId), encoded.toList(),
                "Special token $literal did not encode atomically — got ${encoded.toList()}",
            )
        }
    }

    @Test
    fun `bos and eos token ids resolve from added_tokens`() {
        val tokenizerJson = locateTokenizerJson() ?: return
        val tokenizer = GGUFTokenizer.fromTokenizerJson(Files.readString(tokenizerJson))

        assertEquals(2, tokenizer.bosTokenId, "BOS id should be 2 (Gemma 4)")
        assertEquals(1, tokenizer.eosTokenId, "EOS id should be 1 (Gemma 4)")
    }

    @Test
    fun `decoded text has no SentencePiece marker leakage`() {
        val tokenizerJson = locateTokenizerJson() ?: return
        val tokenizer = GGUFTokenizer.fromTokenizerJson(Files.readString(tokenizerJson))

        // Round-trip encode → decode for a phrase with multiple internal
        // spaces. With BPE strategy + the `▁` SentencePiece marker, the
        // marker would leak into the decoded output literally. With the
        // SP-strategy auto-detect fix, decode replaces `▁` with " ".
        val original = "Hello world from Gemma."
        val ids = tokenizer.encode(original)
        val decoded = tokenizer.decode(ids)

        assertTrue(
            "▁" !in decoded,
            "Decoded text leaks the SentencePiece word-boundary marker: '$decoded'",
        )
        // Spaces should round-trip; we don't assert byte-equality because
        // tokenizers often canonicalise leading/trailing whitespace, but
        // every word should still be present.
        for (word in original.split(' ')) {
            assertTrue(
                word in decoded,
                "Word '$word' missing from decoded round-trip output: '$decoded'",
            )
        }
    }

    private fun locateTokenizerJson(): Path? {
        val raw = System.getenv("GEMMA4_E2B_SAFETENSORS_PATH")?.trim().orEmpty()
        if (raw.isEmpty()) {
            println("[skip] GEMMA4_E2B_SAFETENSORS_PATH not set.")
            return null
        }
        val p = Path.of(raw)
        if (!p.exists()) {
            println("[skip] Path does not exist: $p")
            return null
        }
        val modelDir: Path = if (p.isDirectory()) p else (p.parent ?: return null)
        val tokenizerJson = modelDir.resolve("tokenizer.json")
        if (!tokenizerJson.exists()) {
            println("[skip] tokenizer.json not found in $modelDir")
            return null
        }
        return tokenizerJson
    }
}
