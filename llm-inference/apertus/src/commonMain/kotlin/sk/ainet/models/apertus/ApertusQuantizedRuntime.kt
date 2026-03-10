package sk.ainet.models.apertus

import kotlin.math.sqrt
import kotlin.random.Random
import sk.ainet.apps.llm.DecoderRuntime
import sk.ainet.context.ExecutionContext
import sk.ainet.io.gguf.dequant.DequantOps
import sk.ainet.lang.nn.layers.Embedding
import sk.ainet.lang.nn.normalization.RMSNormalization
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.matmul
import sk.ainet.lang.tensor.plus
import sk.ainet.lang.tensor.t
import sk.ainet.lang.types.FP32

/**
 * Apertus decoder runtime with **lazy dequantization**.
 *
 * Weight matrices (wq, wk, wv, wo, ffnUp, ffnDown, outputWeight) are stored
 * in their original quantized form ([QuantizedTensor]). Each layer dequantizes
 * its weights to FP32 on the fly during [runLayer], then discards the temporary.
 *
 * Memory profile for a 7B Q4_0 model:
 * - Resident: ~3.5 GB (quantized) + norms/embeddings in FP32 (~100 MB)
 * - Per-layer temporary: ~50 MB (one projection matrix at a time)
 * - vs. eager FP32: ~28 GB
 *
 * Trade-off: each token pays a dequantization cost per-layer. This is the same
 * approach used by llama.cpp and is well worth the 4-8x memory savings.
 */
public class ApertusQuantizedRuntime(
    private val ctx: ExecutionContext,
    val weights: ApertusQuantizedRuntimeWeights,
    private val attentionBackend: ApertusAttentionBackend<FP32>,
    private val eps: Float = weights.metadata.rmsNormEps,
    random: Random = Random.Default
) : DecoderRuntime<FP32>(random) {

    override val dim: Int = weights.metadata.embeddingLength
    override val seqLen: Int = weights.metadata.contextLength
    override val vocabSize: Int = weights.metadata.vocabSize
    override val nLayers: Int = weights.layers.size
    override val bosToken: Int = weights.metadata.bosTokenId

    private val nHeads = weights.metadata.headCount
    private val headDim = dim / nHeads
    private val nKvHeads = weights.metadata.kvHeadCount

    private val embedding = Embedding(
        numEmbeddings = vocabSize,
        embeddingDim = dim,
        initWeight = weights.tokenEmbedding,
        name = "token_embd"
    )

    private val outputNormLayer = RMSNormalization<FP32, Float>(
        normalizedShape = intArrayOf(dim),
        eps = eps.toDouble(),
        name = "output_norm",
        initWeight = weights.outputNorm
    )

    private val attnNorms = weights.layers.mapIndexed { i, layer ->
        RMSNormalization<FP32, Float>(
            normalizedShape = intArrayOf(dim),
            eps = eps.toDouble(),
            name = "layer_$i.attn_norm",
            initWeight = layer.attnNorm
        )
    }

    private val ffnNorms = weights.layers.mapIndexed { i, layer ->
        RMSNormalization<FP32, Float>(
            normalizedShape = intArrayOf(dim),
            eps = eps.toDouble(),
            name = "layer_$i.ffn_norm",
            initWeight = layer.ffnNorm
        )
    }

    override fun embedToken(tokenId: Int): Tensor<FP32, Float> =
        embedding.forward(intArrayOf(tokenId), ctx)

    override fun runLayer(layerIdx: Int, x: Tensor<FP32, Float>): Tensor<FP32, Float> {
        val layer = weights.layers[layerIdx]

        // 1. Attention norm
        val attnNorm = attnNorms[layerIdx].forward(x, ctx)

        // 2. QKV projections — dequant weight, matmul, discard temp
        val q = attnNorm.matmul(dequant2D(layer.wq).t())
        val k = attnNorm.matmul(dequant2D(layer.wk).t())
        val v = attnNorm.matmul(dequant2D(layer.wv).t())

        // 3. QK-norm: per-head RMSNorm on Q and K
        val qNormed = applyPerHeadRMSNorm(q, nHeads, headDim, layer.qNorm)
        val kNormed = applyPerHeadRMSNorm(k, nKvHeads, headDim, layer.kNorm)

        // 4. Attention (RoPE + KV cache + GQA)
        val attnOut = attentionBackend.attention(qNormed, kNormed, v, layerIdx, position)

        // 5. Output projection + residual
        val afterAttn = x + attnOut.matmul(dequant2D(layer.wo).t())

        // 6. FFN norm
        val ffnNorm = ffnNorms[layerIdx].forward(afterAttn, ctx)

        // 7. Ungated MLP: up → xIELU → down
        val up = ffnNorm.matmul(dequant2D(layer.ffnUp).t())
        val activated = applyXIELU(up, layer.xieluParams)
        val ffnOut = activated.matmul(dequant2D(layer.ffnDown).t())

        // 8. Residual
        return afterAttn + ffnOut
    }

    override fun outputNorm(x: Tensor<FP32, Float>): Tensor<FP32, Float> =
        outputNormLayer.forward(x, ctx)

    override fun outputProject(x: Tensor<FP32, Float>): Tensor<FP32, Float> =
        x.matmul(dequant2D(weights.outputWeight).t())

    override fun resetState() {
        attentionBackend.reset()
    }

    // ---- Dequantization ----

    /**
     * Dequantize a 2D quantized tensor to FP32.
     *
     * GGUF stores 2D tensors column-major with shape [out, in].
     * We transpose to row-major [in, out] so that `.t()` in the caller
     * gives the correct matmul orientation.
     */
    private fun dequant2D(qt: QuantizedTensor): Tensor<FP32, Float> {
        val floats = qt.dequantToFloat()
        return if (qt.shape.rank == 2) {
            val rows = qt.shape[0]
            val cols = qt.shape[1]
            val transposed = DequantOps.transposeColumnMajorToRowMajor(floats, rows, cols)
            ctx.fromFloatArray<FP32, Float>(Shape(cols, rows), FP32::class, transposed)
        } else {
            ctx.fromFloatArray<FP32, Float>(qt.shape, FP32::class, floats)
        }
    }

    // ---- Apertus-specific helpers (same as ApertusRuntime) ----

    private fun applyPerHeadRMSNorm(
        x: Tensor<FP32, Float>,
        numHeads: Int,
        headDim: Int,
        weight: Tensor<FP32, Float>
    ): Tensor<FP32, Float> {
        val buf = x.expectFloatBuffer().copyOf()
        val w = weight.expectFloatBuffer()
        val totalDim = numHeads * headDim
        val batchSize = if (x.shape.rank == 2) x.shape[0] else 1

        for (b in 0 until batchSize) {
            val batchOffset = b * totalDim
            for (h in 0 until numHeads) {
                val headOffset = batchOffset + h * headDim
                var sumSq = 0f
                for (i in 0 until headDim) {
                    val v = buf[headOffset + i]
                    sumSq += v * v
                }
                val rms = sqrt(sumSq / headDim + eps)
                for (i in 0 until headDim) {
                    buf[headOffset + i] = (buf[headOffset + i] / rms) * w[i]
                }
            }
        }
        return ctx.fromFloatArray<FP32, Float>(x.shape, FP32::class, buf)
    }

    private fun applyXIELU(x: Tensor<FP32, Float>, params: ApertusXIELUParams): Tensor<FP32, Float> {
        val buf = x.expectFloatBuffer().copyOf()
        xielu(buf, params)
        return ctx.fromFloatArray<FP32, Float>(x.shape, FP32::class, buf)
    }
}
