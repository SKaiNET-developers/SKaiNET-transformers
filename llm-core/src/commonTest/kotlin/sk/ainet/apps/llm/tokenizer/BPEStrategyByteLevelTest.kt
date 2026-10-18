package sk.ainet.apps.llm.tokenizer

import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * `BPEStrategy.preprocess` must apply the GPT-2 byte-to-unicode table to
 * every UTF-8 byte, not only turn spaces into `Ġ`. Issue #474.
 */
class BPEStrategyByteLevelTest {

    @Test
    fun `space maps to G-breve`() {
        assertEquals("Ġ", BPEStrategy.preprocess(" "))
    }

    @Test
    fun `newline and tab map to their glyphs`() {
        assertEquals("Ċ", BPEStrategy.preprocess("\n"))
        assertEquals("ĉ", BPEStrategy.preprocess("\t"))
    }

    @Test
    fun `non-ASCII characters map per UTF-8 byte`() {
        // ü = C3 BC → Ã ¼
        assertEquals("\u00c3\u00bc", BPEStrategy.preprocess("ü"))
        // ß = C3 9F; 0x9F is a control byte and is remapped to Ł (U+0141)
        assertEquals("gr\u00c3\u00bc\u00c3\u0141en", BPEStrategy.preprocess("grüßen"))
    }

    @Test
    fun `printable ASCII is unchanged`() {
        assertEquals("hello,Ġworld!", BPEStrategy.preprocess("hello, world!"))
    }

    @Test
    fun `postprocess inverts preprocess`() {
        val samples = listOf(
            "Grüße aus Köln",
            "日本語のテキスト",
            "thumbs up 👍🏽 done",
            "tabs\tand\nnewlines",
        )
        for (s in samples) {
            assertEquals(s, BPEStrategy.postprocess(BPEStrategy.preprocess(s)), "round trip of $s")
        }
    }
}
