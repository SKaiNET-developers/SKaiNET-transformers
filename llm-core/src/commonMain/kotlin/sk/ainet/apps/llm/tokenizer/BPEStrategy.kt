package sk.ainet.apps.llm.tokenizer

import sk.ainet.apps.llm.TokenizerStrategy
import sk.ainet.apps.llm.TokenizerType

/**
 * Tokenizer strategy for GPT-2 / GPT-3 / Qwen style **byte-level** BPE.
 *
 * Byte-level BPE encodes every input byte as a printable Unicode glyph
 * before the BPE merge passes. Spaces map to `Ġ` (U+0120), newlines to
 * `Ċ` (U+010A), tabs to `ĉ` (U+0109), etc.; the full table is the
 * `byte_to_unicode` mapping from OpenAI's GPT-2 reference
 * implementation. [postprocess] reverses that mapping so the consumer
 * sees real bytes back instead of the encoded glyphs (i.e. `Ċ` → `\n`,
 * `Ġ` → ` `).
 *
 * ## Why this matters
 *
 * Without the full inverse table consumers got the raw encoded glyphs
 * leaking into output text — most visibly `Ċ` instead of newlines, but
 * also `ĉ` for tabs and any multi-byte UTF-8 character split into its
 * constituent encoded-byte glyphs. The previous implementation only
 * handled the space marker, which was enough to make Latin sentences
 * look right but broke every byte that needed escaping.
 */
public object BPEStrategy : TokenizerStrategy {
    override val type: TokenizerType = TokenizerType.BPE

    /** GPT-2 BPE space marker: Ġ (U+0120). */
    override val spaceMarker: String = "Ġ"

    override fun preprocess(text: String): String {
        // GPT-2 style: space becomes Ġ prefix on following token
        // First token doesn't get a space prefix unless the text starts with space
        return text.replace(" ", spaceMarker)
    }

    /**
     * Reverse the byte-to-unicode mapping. Each character in [token]
     * decodes to a single byte; the resulting byte sequence is then
     * interpreted as UTF-8.
     *
     * Falls back to a verbatim pass (with only the space marker
     * reversed) if [token] contains a character outside the
     * byte-to-unicode alphabet — defensive guard for non-byte-level
     * inputs (e.g. SentencePiece tokens accidentally routed here).
     */
    override fun postprocess(token: String): String {
        if (token.isEmpty()) return token
        val bytes = ByteArray(token.length)
        var i = 0
        for (ch in token) {
            val b = UNICODE_TO_BYTE[ch] ?: return token.replace(spaceMarker, " ")
            bytes[i++] = b
        }
        return bytes.decodeToString()
    }

    // ---- byte_to_unicode / unicode_to_byte tables ----
    //
    // From the GPT-2 reference (`bytes_to_unicode()` in
    // `openai/gpt-2/src/encoder.py`, mirrored verbatim in HuggingFace
    // `tokenizers` and tiktoken). The 188 "printable" bytes are
    // identity-mapped; the remaining 68 bytes (0..32, 127..160, 173)
    // are remapped into the U+0100..U+0143 range so every encoded
    // byte renders as a single visible non-whitespace glyph.

    private val BYTE_TO_UNICODE: Map<Byte, Char> = buildByteToUnicode()
    private val UNICODE_TO_BYTE: Map<Char, Byte> =
        BYTE_TO_UNICODE.entries.associate { (b, c) -> c to b }

    private fun buildByteToUnicode(): Map<Byte, Char> {
        val printableRanges = listOf(33..126, 161..172, 174..255)
        val printable = HashSet<Int>().apply { printableRanges.forEach { addAll(it) } }
        val map = HashMap<Byte, Char>(256)
        for (range in printableRanges) for (b in range) map[b.toByte()] = b.toChar()
        var n = 0
        for (b in 0..255) {
            if (b !in printable) {
                map[b.toByte()] = (256 + n).toChar()
                n++
            }
        }
        return map
    }
}
