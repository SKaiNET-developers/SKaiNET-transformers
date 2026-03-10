package sk.ainet.models.apertus

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
public class ApertusCpuAttentionBackend<T : DType>(
    private val ctx: ExecutionContext,
    private val weights: ApertusRuntimeWeights<T>,
    private val dtype: KClass<T>,
    kvCache: KvCache? = null,
    private val ropeFreqBase: Float = 12000000f
) : ApertusAttentionBackend<T> {

    private val dim = weights.metadata.embeddingLength
    private val seqLen = weights.metadata.contextLength
    private val nLayers = weights.metadata.blockCount
    private val nHeads = weights.metadata.headCount
    private val nKvHeads = weights.metadata.kvHeadCount
    private val headSize = dim / nHeads
    private val kvDim = nKvHeads * headSize
    private val ropeDim = weights.metadata.ropeDimensionCount ?: headSize
    private val nHeadsPerKv = nHeads / nKvHeads

    private val cache: KvCache = kvCache ?: HeapKvCache(nLayers, seqLen, kvDim)

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
        val ropeStride = headSize / 2

        applyRopeRotation(qBuf, nHeads, headSize, ropeDim, pos, ropeFreqBase, null, null, ropeStride)
        applyRopeRotation(kBuf, nKvHeads, headSize, ropeDim, pos, ropeFreqBase, null, null, ropeStride)
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
