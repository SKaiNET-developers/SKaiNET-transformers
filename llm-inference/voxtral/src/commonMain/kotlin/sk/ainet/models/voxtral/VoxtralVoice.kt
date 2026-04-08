package sk.ainet.models.voxtral

/**
 * Represents a Voxtral voice embedding for speaker conditioning.
 *
 * Voice conditioning in Voxtral works by prepending pre-computed audio frame
 * embeddings to the input sequence. These embeddings are already in the
 * backbone's hidden dimension (3072) and are fed directly into the transformer
 * as in-context conditioning — no architectural changes needed.
 *
 * The token sequence with voice conditioning:
 * ```
 * [BOS] [BEGIN_AUDIO] [voice_frame_0] ... [voice_frame_N] text_tokens [BEGIN_AUDIO]
 * ```
 *
 * @param name Voice identifier (e.g. "casual_male", "fr_female")
 * @param embeddings Voice embedding data: flat FloatArray of shape [numFrames * dim]
 * @param numFrames Number of audio frames in the embedding
 * @param dim Embedding dimension (3072 for Voxtral-4B)
 */
public data class VoxtralVoice(
    val name: String,
    val embeddings: FloatArray,
    val numFrames: Int,
    val dim: Int = 3072
) {
    init {
        require(embeddings.size == numFrames * dim) {
            "Embedding size ${embeddings.size} != numFrames($numFrames) * dim($dim)"
        }
    }

    /**
     * Get the embedding vector for a specific frame.
     * @return FloatArray of length [dim]
     */
    public fun frameEmbedding(frame: Int): FloatArray {
        require(frame in 0 until numFrames) { "Frame $frame out of range [0, $numFrames)" }
        val offset = frame * dim
        return embeddings.copyOfRange(offset, offset + dim)
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is VoxtralVoice) return false
        return name == other.name && numFrames == other.numFrames && dim == other.dim
    }

    override fun hashCode(): Int = name.hashCode()
}

/**
 * Available preset voices for Voxtral TTS.
 */
public object VoxtralVoices {

    /** All 20 preset voice names with their index. */
    public val PRESETS: Map<String, Int> = linkedMapOf(
        "casual_female" to 0,
        "casual_male" to 1,
        "cheerful_female" to 2,
        "neutral_female" to 3,
        "neutral_male" to 4,
        "pt_male" to 5,
        "pt_female" to 6,
        "nl_male" to 7,
        "nl_female" to 8,
        "it_male" to 9,
        "it_female" to 10,
        "fr_male" to 11,
        "fr_female" to 12,
        "es_male" to 13,
        "es_female" to 14,
        "de_male" to 15,
        "de_female" to 16,
        "ar_male" to 17,
        "hi_male" to 18,
        "hi_female" to 19
    )

    /** Default voice if none specified. */
    public const val DEFAULT: String = "neutral_female"

    /**
     * Get the .pt filename for a preset voice.
     */
    public fun filename(voiceName: String): String = "$voiceName.pt"

    /**
     * List all available voice names.
     */
    public fun list(): List<String> = PRESETS.keys.toList()
}
