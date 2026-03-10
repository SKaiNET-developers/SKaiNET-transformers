package sk.ainet.models.apertus

import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt
import sk.ainet.apps.llm.KvCache
import sk.ainet.apps.llm.HeapKvCache
import sk.ainet.apps.llm.applyRopeRotation
import sk.ainet.apps.llm.softmaxInPlace
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * CPU-based attention backend for Apertus.
 *
 * Applies RoPE with high base theta (12M default), stores into KV cache,
 * and computes GQA attention with explicit loops.
 *
 * Q and K are expected to be QK-normed before being passed to this backend.
 */
public class ApertusCpuAttentionBackend<T : DType> private constructor(
    private val ctx: ExecutionContext,
    private val dtype: KClass<T>,
    private val dim: Int,
    private val seqLen: Int,
    private val nLayers: Int,
    private val nHeads: Int,
    private val nKvHeads: Int,
    private val headSize: Int,
    private val kvDim: Int,
    private val ropeDim: Int,
    private val nHeadsPerKv: Int,
    private val ropeFreqBase: Float,
    private val cache: KvCache,
    private val precomputedRopeFreqs: FloatArray?
) : ApertusAttentionBackend<T> {

    /**
     * Primary constructor from full runtime weights.
     */
    public constructor(
        ctx: ExecutionContext,
        weights: ApertusRuntimeWeights<T>,
        dtype: KClass<T>,
        kvCache: KvCache? = null,
        ropeFreqBase: Float = weights.metadata.ropeTheta
    ) : this(
        ctx = ctx,
        dtype = dtype,
        dim = weights.metadata.embeddingLength,
        seqLen = weights.metadata.contextLength,
        nLayers = weights.metadata.blockCount,
        nHeads = weights.metadata.headCount,
        nKvHeads = weights.metadata.kvHeadCount,
        headSize = weights.metadata.embeddingLength / weights.metadata.headCount,
        kvDim = weights.metadata.kvHeadCount * (weights.metadata.embeddingLength / weights.metadata.headCount),
        ropeDim = weights.metadata.ropeDimensionCount
            ?: (weights.metadata.embeddingLength / weights.metadata.headCount),
        nHeadsPerKv = weights.metadata.headCount / weights.metadata.kvHeadCount,
        ropeFreqBase = ropeFreqBase,
        cache = kvCache ?: HeapKvCache(
            weights.metadata.blockCount,
            weights.metadata.contextLength,
            weights.metadata.kvHeadCount * (weights.metadata.embeddingLength / weights.metadata.headCount)
        ),
        precomputedRopeFreqs = weights.ropeFreqs?.let { tensor ->
            val data = tensor.data
            if (data is FloatArrayTensorData<*>) data.buffer.copyOf()
            else data.copyToFloatArray()
        }
    )

    /**
     * Constructor from metadata and optional rope frequencies (for quantized runtime).
     */
    public constructor(
        ctx: ExecutionContext,
        metadata: ApertusModelMetadata,
        dtype: KClass<T>,
        ropeFreqs: FloatArray? = null,
        kvCache: KvCache? = null,
        ropeFreqBase: Float = metadata.ropeTheta
    ) : this(
        ctx = ctx,
        dtype = dtype,
        dim = metadata.embeddingLength,
        seqLen = metadata.contextLength,
        nLayers = metadata.blockCount,
        nHeads = metadata.headCount,
        nKvHeads = metadata.kvHeadCount,
        headSize = metadata.embeddingLength / metadata.headCount,
        kvDim = metadata.kvHeadCount * (metadata.embeddingLength / metadata.headCount),
        ropeDim = metadata.ropeDimensionCount
            ?: (metadata.embeddingLength / metadata.headCount),
        nHeadsPerKv = metadata.headCount / metadata.kvHeadCount,
        ropeFreqBase = ropeFreqBase,
        cache = kvCache ?: HeapKvCache(
            metadata.blockCount,
            metadata.contextLength,
            metadata.kvHeadCount * (metadata.embeddingLength / metadata.headCount)
        ),
        precomputedRopeFreqs = ropeFreqs
    )

    private val scoreBuffer = FloatArray(seqLen)

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

        applyRopeGqa(qBuf, kBuf, position)
        cache.store(layerIdx, position, kBuf, 0, vBuf, 0)

        val attnOutRaw = attentionGqa(layerIdx, qBuf, position)
        return ctx.fromFloatArray<T, Float>(Shape(1, dim), dtype, attnOutRaw)
    }

    override fun batchAttention(
        q: Tensor<T, Float>,
        k: Tensor<T, Float>,
        v: Tensor<T, Float>,
        layerIdx: Int,
        startPos: Int,
    ): Tensor<T, Float> {
        val batchSize = q.shape[0]
        val qAll = q.expectFloatBuffer()
        val kAll = k.expectFloatBuffer()
        val vAll = v.expectFloatBuffer()

        val result = FloatArray(batchSize * dim)

        for (i in 0 until batchSize) {
            val pos = startPos + i

            val qBuf = qAll.copyOfRange(i * dim, (i + 1) * dim)
            val kBuf = kAll.copyOfRange(i * kvDim, (i + 1) * kvDim)
            val vBuf = vAll.copyOfRange(i * kvDim, (i + 1) * kvDim)

            applyRopeGqa(qBuf, kBuf, pos)
            cache.store(layerIdx, pos, kBuf, 0, vBuf, 0)

            val attnOut = attentionGqa(layerIdx, qBuf, pos)
            attnOut.copyInto(result, i * dim)
        }

        return ctx.fromFloatArray<T, Float>(Shape(batchSize, dim), dtype, result)
    }

    override fun reset() {
        cache.reset()
    }

    private fun applyRopeGqa(qBuf: FloatArray, kBuf: FloatArray, pos: Int) {
        require(headSize % 2 == 0) { "RoPE requires even head size; got $headSize" }

        if (precomputedRopeFreqs != null) {
            applyRopeWithFreqs(qBuf, nHeads, headSize, pos, precomputedRopeFreqs)
            applyRopeWithFreqs(kBuf, nKvHeads, headSize, pos, precomputedRopeFreqs)
        } else {
            val ropeStride = headSize / 2
            applyRopeRotation(qBuf, nHeads, headSize, ropeDim, pos, ropeFreqBase, null, null, ropeStride)
            applyRopeRotation(kBuf, nKvHeads, headSize, ropeDim, pos, ropeFreqBase, null, null, ropeStride)
        }
    }

    /**
     * Apply RoPE using precomputed inverse frequencies.
     *
     * For each pair index `p`, the angle is `pos * freqs[p]`.
     * Rotation: out[2p] = in[2p] * cos(θ) - in[2p+1] * sin(θ)
     *           out[2p+1] = in[2p] * sin(θ) + in[2p+1] * cos(θ)
     */
    private fun applyRopeWithFreqs(
        buf: FloatArray,
        numHeads: Int,
        headSize: Int,
        pos: Int,
        freqs: FloatArray
    ) {
        val nPairs = freqs.size
        for (h in 0 until numHeads) {
            val headOffset = h * headSize
            for (pair in 0 until nPairs) {
                val angle = pos * freqs[pair]
                val fcr = cos(angle)
                val fci = sin(angle)
                val i = pair * 2
                val v0 = buf[headOffset + i]
                val v1 = buf[headOffset + i + 1]
                buf[headOffset + i] = (v0 * fcr - v1 * fci).toFloat()
                buf[headOffset + i + 1] = (v0 * fci + v1 * fcr).toFloat()
            }
        }
    }

    private fun attentionGqa(layerIdx: Int, qBuf: FloatArray, pos: Int): FloatArray {
        val out = FloatArray(dim)
        val scale = 1f / sqrt(headSize.toDouble()).toFloat()
        val scores = scoreBuffer

        for (h in 0 until nHeads) {
            val qHeadOffset = h * headSize
            val kvHeadIdx = h / nHeadsPerKv
            val kvHeadOffset = kvHeadIdx * headSize

            for (t in 0..pos) {
                var score = 0f
                for (i in 0 until headSize) {
                    score += qBuf[qHeadOffset + i] * cache.getKey(layerIdx, t, kvHeadOffset, i)
                }
                scores[t] = score * scale
            }

            softmaxInPlace(scores, pos + 1)

            for (t in 0..pos) {
                val weight = scores[t]
                for (i in 0 until headSize) {
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
