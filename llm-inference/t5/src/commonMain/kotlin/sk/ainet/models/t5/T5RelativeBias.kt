package sk.ainet.models.t5

import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.min

/**
 * T5 learned relative-position bias.
 *
 * T5 replaces additive sinusoidal / rotary position encodings with a learned bias
 * added to the attention scores `Q @ Kᵀ` (which T5 leaves unscaled). The bias table
 * `[numBuckets, numHeads]` lives ONLY in block 0 of each stack (encoder and decoder)
 * and is shared by every layer of that stack.
 *
 * This is a faithful port of HuggingFace `T5Attention._relative_position_bucket` and
 * `compute_bias`. The bucketing is:
 * - bidirectional (encoder): half the buckets for future, half for past; sign encoded.
 * - unidirectional (decoder self-attention): only non-positive relative positions
 *   (future is masked out anyway), all buckets for the past.
 *
 * [biasTable] is the raw `relative_attention_bias.weight` tensor flattened row-major
 * as `[numBuckets * numHeads]` (row = bucket, col = head), i.e. `table[bucket*numHeads + head]`.
 */
public class T5RelativeBias(
    private val biasTable: FloatArray,
    private val numHeads: Int,
    private val numBuckets: Int,
    private val maxDistance: Int,
    private val bidirectional: Boolean,
) {

    /**
     * Compute the additive bias `[numHeads, queryLen, keyLen]` flattened row-major as a
     * FloatArray of length `numHeads * queryLen * keyLen`, indexed
     * `[h * queryLen * keyLen + q * keyLen + k]`.
     *
     * When [causal] is true (decoder self-attention), entries where `key > query` are set
     * to a large negative value so the softmax zeros out future positions — the causal
     * mask is folded into the same additive bias.
     */
    public fun compute(queryLen: Int, keyLen: Int, causal: Boolean): FloatArray {
        val out = FloatArray(numHeads * queryLen * keyLen)
        for (q in 0 until queryLen) {
            for (k in 0 until keyLen) {
                // HF: relative_position = memory_position (key) - context_position (query)
                val relativePosition = k - q
                if (causal && relativePosition > 0) {
                    for (h in 0 until numHeads) {
                        out[h * queryLen * keyLen + q * keyLen + k] = NEG_INF
                    }
                    continue
                }
                val bucket = relativePositionBucket(relativePosition)
                for (h in 0 until numHeads) {
                    out[h * queryLen * keyLen + q * keyLen + k] = biasTable[bucket * numHeads + h]
                }
            }
        }
        return out
    }

    /** Port of `T5Attention._relative_position_bucket` for a single scalar relative position. */
    private fun relativePositionBucket(relativePosition: Int): Int {
        var relPos = relativePosition
        var numBucketsLocal = numBuckets
        var bucket = 0
        if (bidirectional) {
            numBucketsLocal /= 2
            if (relPos > 0) bucket += numBucketsLocal
            relPos = abs(relPos)
        } else {
            // unidirectional: keep only non-positive relative positions
            relPos = -min(relPos, 0)
        }
        // now relPos is in the range [0, inf)
        val maxExact = numBucketsLocal / 2
        val isSmall = relPos < maxExact
        // The other half of the buckets are for logarithmically bigger bins in [maxExact, maxDistance)
        val relPosIfLarge = (maxExact + (
            ln(relPos.toDouble() / maxExact) / ln(maxDistance.toDouble() / maxExact) *
                (numBucketsLocal - maxExact)
            ).toInt()).let { min(it, numBucketsLocal - 1) }
        bucket += if (isSmall) relPos else relPosIfLarge
        return bucket
    }

    public companion object {
        // Matches the large-negative additive-mask convention used elsewhere in the codebase
        // (MultiHeadAttention.buildSlidingCausalMask uses -1e30f).
        public const val NEG_INF: Float = -1.0e30f
    }
}
