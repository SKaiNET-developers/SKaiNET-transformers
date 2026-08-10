package sk.ainet.models.whisper

/**
 * Whisper special token IDs, computed from model type (ported from the proven
 * `sk.ainet.apps.kwhisper.WhisperSpecialTokens` in the skainet-whisper app).
 *
 * Multilingual models (nVocab >= 51865): EOT=50257, SOT=50258, languages=50259…,
 * TRANSLATE=50358, TRANSCRIBE=50359, NO_TIMESTAMPS=50363. English-only models
 * shift every ID by −1 (they reuse GPT-2's endoftext as EOT).
 */
public class WhisperSpecialTokens(public val isMultilingual: Boolean) {
    private val offset: Int = if (isMultilingual) 0 else -1

    public val eot: Int = 50257 + offset
    public val sot: Int = 50258 + offset
    public val translate: Int = 50358 + offset
    public val transcribe: Int = 50359 + offset
    public val solm: Int = 50360 + offset
    public val sop: Int = 50361 + offset
    public val noSpeech: Int = 50362 + offset
    public val noTimestamps: Int = 50363 + offset
    public val timestampBegin: Int = 50364 + offset

    private val langTokenBase: Int = 50259 + offset

    public fun languageToken(code: String): Int {
        val langOffset = LANGUAGE_CODES[code.lowercase()]
            ?: error("Unknown language code: $code. Supported: ${LANGUAGE_CODES.keys}")
        return langTokenBase + langOffset
    }

    /** The `[sot, lang, transcribe, noTimestamps]` decode prompt for [languageCode]. */
    public fun transcribePrompt(languageCode: String): IntArray =
        intArrayOf(sot, languageToken(languageCode), transcribe, noTimestamps)

    public fun isTimestamp(tokenId: Int): Boolean = tokenId >= timestampBegin

    public companion object {
        public fun forVocab(vocabSize: Int): WhisperSpecialTokens =
            WhisperSpecialTokens(isMultilingual = vocabSize >= 51865)

        /** ISO language code → offset from the language-token base (whisper order). */
        private val LANGUAGE_CODES: Map<String, Int> = mapOf(
            "en" to 0, "zh" to 1, "de" to 2, "es" to 3, "ru" to 4,
            "ko" to 5, "fr" to 6, "ja" to 7, "pt" to 8, "tr" to 9,
            "pl" to 10, "ca" to 11, "nl" to 12, "ar" to 13, "sv" to 14,
            "it" to 15, "id" to 16, "hi" to 17, "fi" to 18, "vi" to 19,
            "he" to 20, "uk" to 21, "el" to 22, "ms" to 23, "cs" to 24,
            "ro" to 25, "da" to 26, "hu" to 27, "ta" to 28, "no" to 29,
            "th" to 30, "ur" to 31, "hr" to 32, "bg" to 33, "lt" to 34,
            "la" to 35, "mi" to 36, "ml" to 37, "cy" to 38, "sk" to 39,
            "te" to 40, "fa" to 41, "lv" to 42, "bn" to 43, "sr" to 44,
            "az" to 45, "sl" to 46, "kn" to 47, "et" to 48, "mk" to 49,
            "br" to 50, "eu" to 51, "is" to 52, "hy" to 53, "ne" to 54,
            "mn" to 55, "bs" to 56, "kk" to 57, "sq" to 58, "sw" to 59,
            "gl" to 60, "mr" to 61, "pa" to 62, "si" to 63, "km" to 64,
            "sn" to 65, "yo" to 66, "so" to 67, "af" to 68, "oc" to 69,
            "ka" to 70, "be" to 71, "tg" to 72, "sd" to 73, "gu" to 74,
            "am" to 75, "yi" to 76, "lo" to 77, "uz" to 78, "fo" to 79,
            "ht" to 80, "ps" to 81, "tk" to 82, "nn" to 83, "mt" to 84,
            "sa" to 85, "lb" to 86, "my" to 87, "bo" to 88, "tl" to 89,
            "mg" to 90, "as" to 91, "tt" to 92, "haw" to 93, "ln" to 94,
            "ha" to 95, "ba" to 96, "jw" to 97, "su" to 98, "yue" to 99,
        )
    }
}
