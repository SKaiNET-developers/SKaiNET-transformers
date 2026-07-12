package sk.ainet.models.t5

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.layers.Embedding
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.div
import sk.ainet.lang.tensor.matmul
import sk.ainet.lang.tensor.mean
import sk.ainet.lang.tensor.narrow
import sk.ainet.lang.tensor.plus
import sk.ainet.lang.tensor.relu
import sk.ainet.lang.tensor.reshape
import sk.ainet.lang.tensor.softmax
import sk.ainet.lang.tensor.sqrt
import sk.ainet.lang.tensor.sum
import sk.ainet.lang.tensor.t
import sk.ainet.lang.tensor.times
import sk.ainet.lang.tensor.unsqueeze
import sk.ainet.lang.types.DType
import kotlin.math.pow
import kotlin.reflect.KClass

/**
 * Hand-coded T5 encoder-decoder runtime, written in the direct tensor-ops style of
 * [sk.ainet.models.bert.BertRuntime] (per-head attention via narrow/matmul/softmax; no
 * Module/graph composition, no KV cache). Batch size 1.
 *
 * T5 specifics vs. a vanilla transformer, all handled here:
 * - **No 1/√d attention scaling** — T5 feeds raw `Q·Kᵀ` to softmax.
 * - **Learned relative-position bias** added to the scores (block-0 table shared per stack;
 *   see [T5RelativeBias]); cross-attention has none.
 * - **T5LayerNorm = RMSNorm** (no mean subtraction, no bias).
 * - **Un-gated ReLU FFN**: `wo(relu(wi(x)))`.
 * - **Tied embeddings + `d_model^-0.5` logit scaling** before the LM head.
 *
 * The greedy decoder recomputes the full decoder stack each step (O(L²), no cache). For
 * the vec2text use (L ≤ 128, desktop JVM) this is simple and correct; a KV-cache fast path
 * is a later optimization.
 */
public class T5Runtime<T : DType>(
    private val ctx: ExecutionContext,
    private val weights: T5Weights<T>,
    private val dtype: KClass<T>,
) {
    private val cfg = weights.config
    private val eps = cfg.layerNormEpsilon

    private val embedding = Embedding<T, Float>(
        numEmbeddings = cfg.vocabSize,
        embeddingDim = cfg.dModel,
        initWeight = weights.shared,
        name = "shared",
    )

    private val encoderBias = T5RelativeBias(
        biasTable = weights.encoderRelativeBias.data.copyToFloatArray(),
        numHeads = cfg.numHeads,
        numBuckets = cfg.relativeAttentionNumBuckets,
        maxDistance = cfg.relativeAttentionMaxDistance,
        bidirectional = true,
    )

    private val decoderBias: T5RelativeBias? = weights.decoderRelativeBias?.let {
        T5RelativeBias(
            biasTable = it.data.copyToFloatArray(),
            numHeads = cfg.numHeads,
            numBuckets = cfg.relativeAttentionNumBuckets,
            maxDistance = cfg.relativeAttentionMaxDistance,
            bidirectional = false,
        )
    }

    // ---- public API -------------------------------------------------------

    /** Embed token ids to `[seqLen, dModel]` using the tied word-embedding table. */
    public fun embed(tokenIds: IntArray): Tensor<T, Float> = embedding.forward(tokenIds, ctx)

    /**
     * Run the encoder over pre-computed [inputsEmbeds] `[seqLen, dModel]` (token embeddings,
     * or vec2text pseudo-tokens) and return the encoder memory `[seqLen, dModel]`.
     */
    public fun encode(inputsEmbeds: Tensor<T, Float>): Tensor<T, Float> {
        val seqLen = inputsEmbeds.shape[0]
        val bias = biasTensor(encoderBias.compute(seqLen, seqLen, causal = false), seqLen, seqLen)
        var h = inputsEmbeds
        for (layer in weights.encoderLayers) {
            val a = attention(rmsNorm(h, layer.selfAttnNorm), null, layer.selfAttn, bias)
            h = h + a
            val f = feedForward(rmsNorm(h, layer.ffNorm), layer.ff)
            h = h + f
        }
        return rmsNorm(h, weights.encoderFinalNorm)
    }

    /** Encode token ids directly (embed → encoder). */
    public fun encodeTokens(tokenIds: IntArray): Tensor<T, Float> = encode(embed(tokenIds))

    /**
     * Greedy decode conditioned on [encoderMemory] `[memLen, dModel]`. Returns the generated
     * token ids (excluding the decoder start token, up to and including EOS or [maxLength]).
     */
    public fun generate(encoderMemory: Tensor<T, Float>, maxLength: Int = 128): IntArray {
        weights.decoderLayers ?: error("T5Runtime.generate: weights were loaded without a decoder")

        val generated = ArrayList<Int>(maxLength)
        val decoderIds = ArrayList<Int>(maxLength + 1)
        decoderIds.add(cfg.decoderStartTokenId)

        while (generated.size < maxLength) {
            val logits = decoderLastLogits(encoderMemory, decoderIds.toIntArray())
            var best = 0
            for (i in 1 until logits.size) if (logits[i] > logits[best]) best = i
            generated.add(best)
            if (best == cfg.eosTokenId) break
            decoderIds.add(best)
        }
        return generated.toIntArray()
    }

    /**
     * Beam-search decode conditioned on [encoderMemory]. Returns up to [numBeams] complete
     * sequences (each excluding the decoder start token), ordered best-first by
     * length-normalized log-probability (`score / genLen^lengthPenalty`, HF's convention;
     * `lengthPenalty = 1.0` = mean token log-prob). `numBeams <= 1` falls back to [generate].
     *
     * Like the greedy path this recomputes the full decoder per step per beam (no KV cache),
     * so cost scales ~linearly with [numBeams]; keep it small (2–8) for the vec2text sizes.
     */
    public fun generateBeam(
        encoderMemory: Tensor<T, Float>,
        numBeams: Int,
        maxLength: Int = 128,
        lengthPenalty: Double = 1.0,
    ): List<IntArray> {
        if (numBeams <= 1) return listOf(generate(encoderMemory, maxLength))
        weights.decoderLayers ?: error("T5Runtime.generateBeam: weights were loaded without a decoder")

        var beams = listOf(BeamHyp(intArrayOf(cfg.decoderStartTokenId), 0.0, finished = false))
        val finished = ArrayList<BeamHyp>()

        repeat(maxLength) {
            val candidates = ArrayList<BeamHyp>(beams.size * numBeams)
            for (b in beams) {
                val logProbs = logSoftmax(decoderLastLogits(encoderMemory, b.ids))
                for (t in topKIndices(logProbs, numBeams)) {
                    candidates.add(BeamHyp(b.ids + t, b.score + logProbs[t], finished = t == cfg.eosTokenId))
                }
            }
            candidates.sortByDescending { normalizedScore(it, lengthPenalty) }
            val active = ArrayList<BeamHyp>(numBeams)
            for (b in candidates.take(numBeams)) if (b.finished) finished.add(b) else active.add(b)
            beams = active
            if (beams.isEmpty() || finished.size >= numBeams) return@repeat
        }

        return (finished + beams)
            .sortedByDescending { normalizedScore(it, lengthPenalty) }
            .take(numBeams)
            .map { it.ids.copyOfRange(1, it.ids.size) } // drop the decoder start token
    }

    private class BeamHyp(val ids: IntArray, val score: Double, val finished: Boolean)

    private fun normalizedScore(b: BeamHyp, lengthPenalty: Double): Double {
        val genLen = (b.ids.size - 1).coerceAtLeast(1)
        return b.score / genLen.toDouble().pow(lengthPenalty)
    }

    // ---- building blocks --------------------------------------------------

    /** T5LayerNorm (RMSNorm): `x / sqrt(mean(x²) + eps) * weight`, no mean subtraction, no bias. */
    private fun rmsNorm(x: Tensor<T, Float>, weight: Tensor<T, Float>): Tensor<T, Float> {
        val squared = x * x
        val meanSq = squared.mean(dim = x.rank - 1)
        val rmsRaw = (meanSq + eps).sqrt()
        val rms = if (rmsRaw.rank < x.rank) rmsRaw.unsqueeze(rmsRaw.rank) else rmsRaw
        val normalized = x / rms
        val gain = weight.reshape(Shape(1, weight.shape[0]))
        return normalized * gain
    }

    /** Un-gated ReLU FFN: `wo(relu(wi(x)))` (weights stored `[out, in]`, applied via `x @ Wᵀ`). */
    private fun feedForward(x: Tensor<T, Float>, ff: T5FeedForwardWeights<T>): Tensor<T, Float> {
        val hidden = x.matmul(ff.wi.t()).relu()
        return hidden.matmul(ff.wo.t())
    }

    /**
     * Multi-head attention. When [memory] is null this is self-attention over [x]; otherwise
     * cross-attention (Q from [x], K/V from [memory]). [bias] `[numHeads, Lq, Lk]` (or null)
     * is added to the raw scores before softmax — T5 applies NO 1/√d scaling.
     */
    private fun attention(
        x: Tensor<T, Float>,
        memory: Tensor<T, Float>?,
        w: T5AttentionWeights<T>,
        bias: Tensor<T, Float>?,
    ): Tensor<T, Float> {
        val kvSource = memory ?: x
        val q = x.matmul(w.q.t())          // [Lq, innerDim]
        val k = kvSource.matmul(w.k.t())   // [Lk, innerDim]
        val v = kvSource.matmul(w.v.t())   // [Lk, innerDim]
        val lq = q.shape[0]
        val lk = k.shape[0]

        val heads = ArrayList<Tensor<T, Float>>(cfg.numHeads)
        for (hh in 0 until cfg.numHeads) {
            val off = hh * cfg.dKv
            val qh = q.narrow(dim = 1, start = off, length = cfg.dKv) // [Lq, dKv]
            val kh = k.narrow(dim = 1, start = off, length = cfg.dKv) // [Lk, dKv]
            val vh = v.narrow(dim = 1, start = off, length = cfg.dKv) // [Lk, dKv]
            var scores = qh.matmul(kh.t())                            // [Lq, Lk], no scaling (T5)
            if (bias != null) {
                val bh = bias.narrow(dim = 0, start = hh, length = 1).reshape(Shape(lq, lk))
                scores = scores + bh
            }
            val attn = scores.softmax(dim = 1)
            heads.add(attn.matmul(vh))                                // [Lq, dKv]
        }
        val merged = ctx.ops.concat(heads, dim = 1)                  // [Lq, innerDim]
        return merged.matmul(w.o.t())                                // [Lq, dModel]
    }

    /** Build a `[numHeads, Lq, Lk]` bias tensor from the flat FloatArray produced by [T5RelativeBias]. */
    private fun biasTensor(flat: FloatArray, lq: Int, lk: Int): Tensor<T, Float> =
        ctx.fromFloatArray(Shape(cfg.numHeads, lq, lk), dtype, flat)

    /**
     * Run the decoder stack over [decoderIds] (including the start token) attending to
     * [encoderMemory], and return the **last position's** vocab logits `[vocab]` as a
     * FloatArray (after the `d_model^-0.5` tied-embedding scaling). Shared by greedy and
     * beam decoding.
     */
    private fun decoderLastLogits(encoderMemory: Tensor<T, Float>, decoderIds: IntArray): FloatArray {
        val decLayers = weights.decoderLayers!!
        val bias = decoderBias!!
        val curLen = decoderIds.size
        var h = embed(decoderIds)
        val selfBias = biasTensor(bias.compute(curLen, curLen, causal = true), curLen, curLen)
        for (layer in decLayers) {
            val sa = attention(rmsNorm(h, layer.selfAttnNorm), null, layer.selfAttn, selfBias)
            h = h + sa
            val ca = attention(rmsNorm(h, layer.crossAttnNorm), encoderMemory, layer.crossAttn, null)
            h = h + ca
            val ff = feedForward(rmsNorm(h, layer.ffNorm), layer.ff)
            h = h + ff
        }
        val normed = rmsNorm(h, weights.decoderFinalNorm!!)
        val lastRow = normed.narrow(dim = 0, start = curLen - 1, length = 1) // [1, dModel]
        val scale = 1.0f / kotlin.math.sqrt(cfg.dModel.toFloat())
        return (lastRow * scale).matmul(weights.shared.t()).data.copyToFloatArray() // [vocab]
    }

    /** Numerically-stable log-softmax of a logits FloatArray → per-index log-probability. */
    private fun logSoftmax(logits: FloatArray): DoubleArray {
        var max = logits[0]
        for (l in logits) if (l > max) max = l
        var sum = 0.0
        for (l in logits) sum += kotlin.math.exp((l - max).toDouble())
        val logDenom = kotlin.math.ln(sum) + max
        return DoubleArray(logits.size) { logits[it] - logDenom }
    }

    /** Indices of the [k] largest entries of [scores] (unordered), via a linear top-k scan. */
    private fun topKIndices(scores: DoubleArray, k: Int): IntArray {
        val idx = IntArray(k) { -1 }
        val vals = DoubleArray(k) { Double.NEGATIVE_INFINITY }
        for (i in scores.indices) {
            var minSlot = 0
            for (j in 1 until k) if (vals[j] < vals[minSlot]) minSlot = j
            if (scores[i] > vals[minSlot]) { vals[minSlot] = scores[i]; idx[minSlot] = i }
        }
        return idx
    }
}
