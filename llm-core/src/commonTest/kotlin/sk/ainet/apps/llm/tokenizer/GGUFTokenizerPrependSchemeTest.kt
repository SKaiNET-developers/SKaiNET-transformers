package sk.ainet.apps.llm.tokenizer

import kotlin.test.Test
import kotlin.test.assertContentEquals

/**
 * `GGUFTokenizer.fromTokenizerJson` must honour the file's Metaspace
 * `prepend_scheme` (or a legacy `Prepend` normalizer) instead of assuming
 * that no word-boundary marker is prepended. Issue #473.
 *
 * Vocabulary: `▁`=0, `H`=1, `i`=2, `Hi`=3, `▁Hi`=4, merges `H i` then `▁ Hi`.
 * With the prefix `"Hi"` encodes to `[▁Hi]`, without it to `[Hi]`.
 */
class GGUFTokenizerPrependSchemeTest {

    private fun tokenizerJson(extraBlocks: String): String = """
        {
          $extraBlocks
          "model": {
            "type": "BPE",
            "vocab": {"▁": 0, "H": 1, "i": 2, "Hi": 3, "▁Hi": 4},
            "merges": ["H i", "▁ Hi"]
          }
        }
    """.trimIndent()

    private fun metaspace(scheme: String): String =
        """"pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "$scheme", "split": true},"""

    @Test
    fun `prepend_scheme always prepends the marker`() {
        val tokenizer = GGUFTokenizer.fromTokenizerJson(tokenizerJson(metaspace("always")))
        assertContentEquals(intArrayOf(4), tokenizer.encode("Hi"))
    }

    @Test
    fun `prepend_scheme first prepends the marker`() {
        val tokenizer = GGUFTokenizer.fromTokenizerJson(tokenizerJson(metaspace("first")))
        assertContentEquals(intArrayOf(4), tokenizer.encode("Hi"))
    }

    @Test
    fun `prepend_scheme never keeps the first word bare`() {
        val tokenizer = GGUFTokenizer.fromTokenizerJson(tokenizerJson(metaspace("never")))
        assertContentEquals(intArrayOf(3), tokenizer.encode("Hi"))
    }

    @Test
    fun `no pre_tokenizer block keeps the first word bare`() {
        val tokenizer = GGUFTokenizer.fromTokenizerJson(
            tokenizerJson(""""normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},"""),
        )
        assertContentEquals(intArrayOf(3), tokenizer.encode("Hi"))
    }

    @Test
    fun `legacy Prepend normalizer inside a Sequence prepends the marker`() {
        val normalizer = """
            "normalizer": {"type": "Sequence", "normalizers": [
              {"type": "Prepend", "prepend": "▁"},
              {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}
            ]},
        """.trimIndent()
        val tokenizer = GGUFTokenizer.fromTokenizerJson(tokenizerJson(normalizer))
        assertContentEquals(intArrayOf(4), tokenizer.encode("Hi"))
    }
}
