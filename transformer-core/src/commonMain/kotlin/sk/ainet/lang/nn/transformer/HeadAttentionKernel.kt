package sk.ainet.lang.nn.transformer

import kotlin.math.exp

/**
 * Scalar per-head attention kernels (SKEEP-005). One call handles one query head; heads are the
 * schedule's units and write disjoint output slices, so any number of them may run concurrently
 * as long as each task brings its own [scores] scratch. The loop and rounding order is exactly
 * the pre-schedule `fusedDecodeAttention` (decode) and the engine's `scaledDotProductAttention`
 * (prefill): a scheduled forward is bit-identical to a sequential one.
 */
internal object ScalarHeadAttentionKernel {

    /**
     * Decode: one query row `q[qOff until qOff + headDim]` against KV group [g] of [kv]. Writes
     * `out[outOff until outOff + headDim]`; uses `scores[0 until kv.length]`. Softmax as
     * `Σ e·v` then `· 1/sum` — the fused decode order.
     */
    fun decodeHead(
        q: FloatArray, qOff: Int,
        kv: KVBufferView, g: Int,
        scale: Float,
        scores: FloatArray,
        out: FloatArray, outOff: Int,
    ) {
        val headDim = kv.headDim
        val seqKV = kv.length
        val rs = kv.rowStride
        val base = g * kv.headStride
        val k = kv.keys
        val v = kv.values
        var maxV = Float.NEGATIVE_INFINITY
        for (ki in 0 until seqKV) {
            val kOff = base + ki * rs
            var dot = 0f
            for (d in 0 until headDim) dot += q[qOff + d] * k[kOff + d]
            val s = dot * scale
            scores[ki] = s
            if (s > maxV) maxV = s
        }
        var sum = 0f
        for (ki in 0 until seqKV) {
            val e = exp(scores[ki] - maxV)
            scores[ki] = e
            sum += e
        }
        val inv = if (sum > 0f) 1f / sum else 0f
        for (d in 0 until headDim) {
            var acc = 0f
            for (ki in 0 until seqKV) acc += scores[ki] * v[base + ki * rs + d]
            out[outOff + d] = acc * inv
        }
    }

    /**
     * Prefill: query rows `[qi0, qi1)` of head `h` (row `qi` at `q[qBase + qi * qRowStride]`)
     * against KV group [g]. Keys outside the causal / sliding-window band are excluded by loop
     * bounds — bit-identical to the engine's `-inf` additive mask, since `exp(-inf - max) == 0f`
     * and `0f * v` adds exactly nothing. Softmax divides in place then accumulates — the engine
     * SDPA order. Output row `qi` lands at `out[qi * outRowStride + outOff]`.
     * `absOffset` is the absolute position of query row 0 (`seqKV - seqQ` for a causal prefill).
     */
    fun prefillRows(
        q: FloatArray, qBase: Int, qRowStride: Int,
        qi0: Int, qi1: Int,
        kv: KVBufferView, g: Int,
        scale: Float,
        causal: Boolean, absOffset: Int,
        window: Int?, rightContext: Int,
        scores: FloatArray,
        out: FloatArray, outOff: Int, outRowStride: Int,
    ) {
        val headDim = kv.headDim
        val seqKV = kv.length
        val rs = kv.rowStride
        val base = g * kv.headStride
        val k = kv.keys
        val v = kv.values
        for (qi in qi0 until qi1) {
            val absQ = absOffset + qi
            var lo = 0
            var hi = seqKV                      // exclusive
            if (window != null) {
                // The band subsumes causality (and may look `rightContext` keys ahead), exactly
                // like the sliding mask that replaces SDPA's causal path on the general route.
                lo = maxOf(lo, absQ - window + 1)
                hi = minOf(hi, absQ + rightContext + 1)
            } else if (causal) {
                hi = minOf(hi, absQ + 1)
            }
            val qOff = qBase + qi * qRowStride
            val oOff = outOff + qi * outRowStride
            if (hi <= lo) {
                // The band lies entirely outside the returned keys (a sliding cache that holds
                // fewer positions than the prefill is long). The engine path adds -1e30 to every
                // score, which makes them all equal, so softmax is uniform: reproduce that
                // exactly — mean of V over all keys, engine summation order.
                val w = 1f / seqKV.toFloat()
                for (d in 0 until headDim) {
                    var sum = 0f
                    for (ki in 0 until seqKV) sum += w * v[base + ki * rs + d]
                    out[oOff + d] = sum
                }
                continue
            }
            var maxV = Float.NEGATIVE_INFINITY
            for (ki in lo until hi) {
                val kOff = base + ki * rs
                var dot = 0f
                for (d in 0 until headDim) dot += q[qOff + d] * k[kOff + d]
                val s = dot * scale
                scores[ki] = s
                if (s > maxV) maxV = s
            }
            var sumExp = 0f
            for (ki in lo until hi) {
                val e = exp(scores[ki] - maxV)
                scores[ki] = e
                sumExp += e
            }
            if (sumExp > 0f) for (ki in lo until hi) scores[ki] /= sumExp
            for (d in 0 until headDim) {
                var sum = 0f
                for (ki in lo until hi) sum += scores[ki] * v[base + ki * rs + d]
                out[oOff + d] = sum
            }
        }
    }
}
