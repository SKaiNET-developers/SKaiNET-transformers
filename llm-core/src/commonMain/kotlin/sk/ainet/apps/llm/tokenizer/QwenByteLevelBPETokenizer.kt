package sk.ainet.apps.llm.tokenizer

import sk.ainet.apps.llm.Tokenizer

/**
 * Byte-level BPE tokenizer compatible with HuggingFace Qwen2Tokenizer.
 *
 * Implements the three-stage pipeline used by Qwen3.5 (and Qwen2/3):
 * 1. **Special token splitting** — find and isolate atomic special tokens
 * 2. **Regex pretokenization** — split non-special text by [pretokenizeRegex]
 * 3. **Byte-level BPE** — convert each chunk to byte-level characters (GPT-2 mapping),
 *    then apply greedy BPE merges
 *
 * This is a Kotlin Multiplatform implementation with no external dependencies.
 */
public class QwenByteLevelBPETokenizer internal constructor(
    private val vocab: List<String>,
    private val tokenToId: Map<String, Int>,
    private val mergeRanks: Map<String, Int>,
    private val specialTokens: Map<String, Int>,
    private val pretokenizeRegex: Regex?,
    override val eosTokenId: Int,
    private val addBosToken: Boolean,
    private val addEosToken: Boolean,
    override val bosTokenId: Int
) : Tokenizer {

    override val vocabSize: Int get() = vocab.size

    override fun encode(text: String): IntArray {
        if (text.isEmpty()) return intArrayOf()

        val tokens = mutableListOf<Int>()

        // Step 1: Split text around special tokens
        val segments = splitOnSpecialTokens(text)

        for (segment in segments) {
            val specialId = specialTokens[segment]
            if (specialId != null) {
                // Atomic special token
                tokens.add(specialId)
            } else {
                // Step 2: Regex pretokenization
                val chunks = pretokenize(segment)
                // Step 3: Byte-level BPE on each chunk
                for (chunk in chunks) {
                    tokens.addAll(encodeByteLevelBPE(chunk))
                }
            }
        }

        // Add BOS/EOS if configured
        val result = mutableListOf<Int>()
        if (addBosToken) result.add(bosTokenId)
        result.addAll(tokens)
        if (addEosToken) result.add(eosTokenId)

        return result.toIntArray()
    }

    override fun decode(token: Int): String {
        if (token < 0 || token >= vocab.size) return ""
        val tokenStr = vocab[token]
        // Check if it's a special token — return as-is
        if (specialTokens.containsKey(tokenStr)) return tokenStr
        // Decode byte-level characters back to bytes
        return decodeByteLevelString(tokenStr)
    }

    override fun decode(tokens: IntArray): String {
        val bytes = mutableListOf<Byte>()
        val sb = StringBuilder()

        for (token in tokens) {
            if (token < 0 || token >= vocab.size) continue
            val tokenStr = vocab[token]

            if (specialTokens.containsKey(tokenStr)) {
                // Flush accumulated bytes
                if (bytes.isNotEmpty()) {
                    sb.append(bytesToString(bytes))
                    bytes.clear()
                }
                sb.append(tokenStr)
            } else {
                // Accumulate byte-level characters
                for (ch in tokenStr) {
                    val byteVal = UNICODE_TO_BYTE[ch]
                    if (byteVal != null) {
                        bytes.add(byteVal)
                    }
                }
            }
        }

        // Flush remaining bytes
        if (bytes.isNotEmpty()) {
            sb.append(bytesToString(bytes))
        }

        return sb.toString()
    }

    /**
     * Split [text] into segments, alternating between special tokens and regular text.
     * Special tokens are returned as separate segments.
     */
    private fun splitOnSpecialTokens(text: String): List<String> {
        if (specialTokens.isEmpty()) return listOf(text)

        val result = mutableListOf<String>()
        var remaining = text

        while (remaining.isNotEmpty()) {
            // Find the earliest special token occurrence
            var earliestIdx = Int.MAX_VALUE
            var earliestToken = ""

            for (token in specialTokens.keys) {
                val idx = remaining.indexOf(token)
                if (idx >= 0 && idx < earliestIdx) {
                    earliestIdx = idx
                    earliestToken = token
                }
            }

            if (earliestIdx == Int.MAX_VALUE) {
                // No more special tokens
                result.add(remaining)
                break
            }

            // Add text before the special token
            if (earliestIdx > 0) {
                result.add(remaining.substring(0, earliestIdx))
            }
            // Add the special token
            result.add(earliestToken)
            remaining = remaining.substring(earliestIdx + earliestToken.length)
        }

        return result
    }

    /**
     * Apply regex pretokenization to split text into chunks.
     */
    private fun pretokenize(text: String): List<String> {
        val regex = pretokenizeRegex ?: return listOf(text)
        return regex.findAll(text).map { it.value }.toList().ifEmpty { listOf(text) }
    }

    /**
     * Encode a single pretokenized chunk using byte-level BPE.
     */
    private fun encodeByteLevelBPE(chunk: String): List<Int> {
        if (chunk.isEmpty()) return emptyList()

        // Convert to byte-level characters
        val byteChars = chunk.encodeToByteArray().map { BYTE_TO_UNICODE[it.toInt() and 0xFF].toString() }.toMutableList()

        // Greedy BPE merges by rank (lower rank = higher priority)
        while (byteChars.size > 1) {
            var bestIdx = -1
            var bestRank = Int.MAX_VALUE

            for (i in 0 until byteChars.size - 1) {
                val pair = byteChars[i] + byteChars[i + 1]
                val rank = mergeRanks[pair]
                if (rank != null && rank < bestRank) {
                    bestRank = rank
                    bestIdx = i
                }
            }

            if (bestIdx < 0) break

            byteChars[bestIdx] = byteChars[bestIdx] + byteChars[bestIdx + 1]
            byteChars.removeAt(bestIdx + 1)
        }

        // Convert to token IDs
        return byteChars.map { tokenToId[it] ?: 0 }
    }

    /**
     * Decode a byte-level encoded string back to regular text.
     */
    private fun decodeByteLevelString(tokenStr: String): String {
        val bytes = tokenStr.mapNotNull { UNICODE_TO_BYTE[it] }
        return bytesToString(bytes)
    }

    private fun bytesToString(bytes: List<Byte>): String {
        return bytes.toByteArray().decodeToString()
    }

    public companion object {
        /**
         * GPT-2 byte-to-unicode mapping table.
         *
         * Maps each byte (0–255) to a unique unicode character. Printable ASCII bytes
         * map to themselves; other bytes are shifted to avoid control characters.
         */
        internal val BYTE_TO_UNICODE: CharArray = buildByteToUnicodeMap()

        /**
         * Reverse mapping: unicode character back to byte value.
         */
        internal val UNICODE_TO_BYTE: Map<Char, Byte> = buildUnicodeToByteMap()

        private fun buildByteToUnicodeMap(): CharArray {
            val map = CharArray(256)
            var n = 0

            // Printable ranges: '!' (33) to '~' (126), '¡' (161) to '¬' (172), '®' (174) to 'ÿ' (255)
            val printableRanges = (33..126) + (161..172) + (174..255)
            val printableSet = printableRanges.toSet()

            for (b in 0..255) {
                if (b in printableSet) {
                    map[b] = b.toChar()
                } else {
                    // Map to characters starting at U+0100 (Ā) to avoid collisions
                    map[b] = (256 + n).toChar()
                    n++
                }
            }

            return map
        }

        private fun buildUnicodeToByteMap(): Map<Char, Byte> {
            val map = mutableMapOf<Char, Byte>()
            for (b in 0..255) {
                map[BYTE_TO_UNICODE[b]] = b.toByte()
            }
            return map
        }
    }
}
