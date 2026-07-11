package sk.ainet.models.t5

/**
 * Configuration for a T5 (encoder-decoder) transformer, matching the HuggingFace
 * `t5-base` / `sentence-transformers/gtr-t5-base` config fields.
 *
 * Defaults are the `t5-base` values. `feed_forward_proj = "relu"` (the original T5
 * FFN: `wo(relu(wi(x)))`, a single un-gated projection) — set [gated] = true for the
 * v1.1 "gated-gelu" variant (not needed for gtr-base / the vec2text checkpoints).
 */
public data class T5Config(
    val dModel: Int = 768,
    val numLayers: Int = 12,
    val numDecoderLayers: Int = 12,
    val numHeads: Int = 12,
    val dKv: Int = 64,
    val dFf: Int = 3072,
    val vocabSize: Int = 32128,
    val relativeAttentionNumBuckets: Int = 32,
    val relativeAttentionMaxDistance: Int = 128,
    val layerNormEpsilon: Double = 1e-6,
    val gated: Boolean = false,
    val eosTokenId: Int = 1,
    val padTokenId: Int = 0,
    /** T5 uses pad as the decoder start token. */
    val decoderStartTokenId: Int = 0,
    /** Max embedder input length (32 for the gtr__nq__32 checkpoints). */
    val maxSeqLength: Int = 32,
) {
    /** Inner attention dimension = numHeads * dKv (equals dModel for t5-base but not required to). */
    val innerDim: Int get() = numHeads * dKv
}
