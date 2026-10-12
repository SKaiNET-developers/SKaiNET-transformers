package sk.ainet.apps.llm.tokenizer

import kotlin.test.Test
import kotlin.test.assertContentEquals

/**
 * `tokenizer.json` stores BPE merges either as `"a b"` strings (older
 * `tokenizers` releases) or as `["a", "b"]` arrays (0.20+). Both forms must
 * yield the same merge ranks. Issue #472.
 *
 * The vocabulary is chosen so that the merge order is observable: without a
 * parsed merge the greedy pass pairs `a`+`b` first and `abc` encodes to
 * `[ab, c]`; with the single merge `b c` ranked it must encode to `[a, bc]`.
 */
class GGUFTokenizerMergesFormatTest {

    private fun tokenizerJson(merges: String): String = """
        {
          "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "bc": 4},
            "merges": $merges
          }
        }
    """.trimIndent()

    @Test
    fun `string-form merges are ranked`() {
        val tokenizer = GGUFTokenizer.fromTokenizerJson(tokenizerJson("""["b c"]"""))
        assertContentEquals(intArrayOf(0, 4), tokenizer.encode("abc"))
    }

    @Test
    fun `array-form merges are ranked`() {
        val tokenizer = GGUFTokenizer.fromTokenizerJson(tokenizerJson("""[["b", "c"]]"""))
        assertContentEquals(intArrayOf(0, 4), tokenizer.encode("abc"))
    }

    @Test
    fun `array-form merges with escaped strings are unescaped`() {
        val tokenizer = GGUFTokenizer.fromTokenizerJson(tokenizerJson("""[["b", "c"]]"""))
        assertContentEquals(intArrayOf(0, 4), tokenizer.encode("abc"))
    }
}
