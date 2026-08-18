package sk.ainet.models.gemma

import kotlin.math.max
import kotlin.math.sqrt
import sk.ainet.apps.llm.applyRopeRotation
import sk.ainet.apps.llm.softmaxInPlace
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * CPU-based attention backend for Gemma 4 with hybrid attention and proportional RoPE.
 *
 * Key differences from Gemma 3n:
 * - Proportional RoPE (p-RoPE) for global layers: applies partial_rotary_factor
 * - Global layers may use a different head dimension (global_head_dim)
 * - Full per-layer attention type list (not a repeating pattern)
 * - Last layer is always global
 *
 * @param ctx ExecutionContext for tensor creation
 * @param weights Model weights
 * @param dtype Data type for tensor operations
 * @param config Model configuration
 * @param kvCache KV cache implementation with layer sharing
 */
public class Gemma4AttentionBackend<T : DType>(
    private val ctx: ExecutionContext,
    private val weights: Gemma4RuntimeWeights<T>,
    private val dtype: KClass<T>,
    private val config: Gemma4Config,
    kvCache: Gemma4KvCache? = null
) : AttentionBackend<T> {

    private val cache: Gemma4KvCache = kvCache ?: HeapGemma4KvCache.fromConfig(config, 4096)
    private val seqLen: Int = cache.seqLen
    private val nHeads = config.numAttentionHeads
    private val nKvHeads = config.numKvHeads
    private val nHeadsPerKv = config.numHeadsPerKv
    private val slidingWindow = config.slidingWindow

    override fun attention(
        q: Tensor<T, Float>,
        k: Tensor<T, Float>,
        v: Tensor<T, Float>,
        layerIdx: Int,
        position: Int
    ): Tensor<T, Float> {
        val qBuf = q.expectFloatBuffer()
        val kBuf = k.expectFloatBuffer()
        val vBuf = v.expectFloatBuffer()

        val layerHeadDim = config.getHeadDim(layerIdx)
        val ropeBase = config.getRopeBase(layerIdx)
        val rotaryDim = config.getRotaryDim(layerIdx)

        // Apply RoPE with layer-specific parameters
        applyRopeGqa(qBuf, kBuf, position, ropeBase, layerHeadDim, rotaryDim)

        // Store to KV cache — pad to maxKvDim if layer has smaller KV dim.
        //
        // Shared-KV layers (config.isKvShared) all resolve to the SAME physical cache slot
        // (config.getCacheLayerIndex) — that's the whole point of sharing, to reuse one
        // layer's K/V across the group instead of recomputing per layer. But
        // HybridTransformerBlock has no KV-sharing awareness: it always runs a fresh Q/K/V
        // projection for every layer and passes the result here regardless. Storing
        // unconditionally meant every shared layer clobbered the one shared slot with its
        // own (architecturally unintended) K/V on every forward() call — so by the time a
        // later layer in the group attended over an earlier *position*, it read whichever
        // shared layer happened to run last for that position, not the group's own anchor
        // K/V, corrupting attention more and more as generated sequence length grew (see
        // GEMMA4_E2B_SKAINET_FINDINGS.md — immediate-<eos> collapse, worsening with length).
        // Only the group's designated owner (config.getCacheLayerIndex(layerIdx) ==
        // layerIdx — itself for non-shared layers, the first layer of the group for shared
        // ones) should ever write; every other shared layer must read-only reuse what the
        // owner already stored for the current position (the owner's block always runs
        // first in layer order, so its store always precedes any shared reader in the same
        // forward() pass).
        if (config.getCacheLayerIndex(layerIdx) == layerIdx) {
            val layerKvDim = nKvHeads * layerHeadDim
            val maxKvDim = cache.kvDim
            if (layerKvDim < maxKvDim) {
                val paddedK = FloatArray(maxKvDim)
                kBuf.copyInto(paddedK, 0, 0, layerKvDim)
                val paddedV = FloatArray(maxKvDim)
                vBuf.copyInto(paddedV, 0, 0, layerKvDim)
                cache.store(layerIdx, position, paddedK, 0, paddedV, 0)
            } else {
                cache.store(layerIdx, position, kBuf, 0, vBuf, 0)
            }
        }

        // Compute attention
        val layerType = config.getLayerType(layerIdx)
        val kvDim = nKvHeads * layerHeadDim
        val attnOutRaw = when (layerType) {
            LayerType.SLIDING -> attentionSliding(layerIdx, qBuf, position, layerHeadDim, kvDim)
            LayerType.GLOBAL -> attentionGlobal(layerIdx, qBuf, position, layerHeadDim, kvDim)
        }

        return ctx.fromFloatArray<T, Float>(Shape(1, nHeads * layerHeadDim), dtype, attnOutRaw)
    }

    override fun reset() {
        cache.reset()
    }

    /**
     * Apply RoPE with support for partial rotary factor.
     *
     * For global layers with partial_rotary_factor < 1.0, only a fraction
     * of the head dimensions get RoPE applied; the rest pass through unchanged.
     */
    private fun applyRopeGqa(
        qBuf: FloatArray,
        kBuf: FloatArray,
        pos: Int,
        ropeBase: Float,
        headDim: Int,
        rotaryDim: Int
    ) {
        val ropeReal = weights.ropeFreqReal?.expectFloatBuffer()
        val ropeImag = weights.ropeFreqImag?.expectFloatBuffer()
        val ropeStride = rotaryDim / 2

        require(rotaryDim % 2 == 0) { "RoPE requires even rotary dim; got $rotaryDim" }

        applyRopeRotation(
            qBuf, nHeads, headDim, rotaryDim, pos, ropeBase,
            ropeReal, ropeImag, ropeStride, config.ropeBaseLocal
        )
        applyRopeRotation(
            kBuf, nKvHeads, headDim, rotaryDim, pos, ropeBase,
            ropeReal, ropeImag, ropeStride, config.ropeBaseLocal
        )
    }

    /**
     * Sliding window attention for local layers.
     */
    private fun attentionSliding(
        layerIdx: Int,
        qBuf: FloatArray,
        pos: Int,
        headDim: Int,
        kvDim: Int
    ): FloatArray {
        val out = FloatArray(nHeads * headDim)
        val scale = 1f / sqrt(headDim.toDouble()).toFloat()

        val windowStart = max(0, pos - slidingWindow + 1)
        val windowSize = pos - windowStart + 1

        for (h in 0 until nHeads) {
            val qHeadOffset = h * headDim
            val kvHeadIdx = h / nHeadsPerKv
            val kvHeadOffset = kvHeadIdx * headDim
            val scores = FloatArray(windowSize)

            for (t in windowStart..pos) {
                var score = 0f
                for (i in 0 until headDim) {
                    score += qBuf[qHeadOffset + i] * cache.getKey(layerIdx, t, kvHeadOffset, i)
                }
                scores[t - windowStart] = score * scale
            }

            softmaxInPlace(scores)

            for (t in windowStart..pos) {
                val weight = scores[t - windowStart]
                for (i in 0 until headDim) {
                    out[qHeadOffset + i] += weight * cache.getValue(layerIdx, t, kvHeadOffset, i)
                }
            }
        }
        return out
    }

    /**
     * Global full-context attention for global layers.
     */
    private fun attentionGlobal(
        layerIdx: Int,
        qBuf: FloatArray,
        pos: Int,
        headDim: Int,
        kvDim: Int
    ): FloatArray {
        val out = FloatArray(nHeads * headDim)
        val scale = 1f / sqrt(headDim.toDouble()).toFloat()

        for (h in 0 until nHeads) {
            val qHeadOffset = h * headDim
            val kvHeadIdx = h / nHeadsPerKv
            val kvHeadOffset = kvHeadIdx * headDim
            val scores = FloatArray(pos + 1)

            for (t in 0..pos) {
                var score = 0f
                for (i in 0 until headDim) {
                    score += qBuf[qHeadOffset + i] * cache.getKey(layerIdx, t, kvHeadOffset, i)
                }
                scores[t] = score * scale
            }

            softmaxInPlace(scores)

            for (t in 0..pos) {
                val weight = scores[t]
                for (i in 0 until headDim) {
                    out[qHeadOffset + i] += weight * cache.getValue(layerIdx, t, kvHeadOffset, i)
                }
            }
        }
        return out
    }

    private fun Tensor<T, Float>.expectFloatBuffer(): FloatArray {
        val data = this.data
        if (data is FloatArrayTensorData<*>) return data.buffer
        return data.copyToFloatArray()
    }
}
