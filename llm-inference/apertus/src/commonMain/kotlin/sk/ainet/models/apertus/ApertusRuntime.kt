package sk.ainet.models.apertus

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.random.Random
import sk.ainet.apps.llm.DecoderRuntime
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.layers.Embedding
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.matmul
import sk.ainet.lang.tensor.plus
import sk.ainet.lang.tensor.t
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.nn.normalization.RMSNormalization
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Apertus decoder runtime with pluggable attention backend.
 *
 * Key differences from LLaMA:
 * - **xIELU activation** with per-layer learned scalar parameters (replaces SiLU)
 * - **Ungated MLP** — only up_proj + down_proj, no gate_proj
 * - **QK-norm** — per-head RMSNorm on Q and K before RoPE
 *
 * Extends [DecoderRuntime] for shared forward/generate/sample logic.
 */
@Deprecated(
    message = "Use OptimizedLLMRuntime with apertusNetwork() instead. " +
        "See docs/optimizable-LLM-NNs-DAG.md for migration guide.",
    replaceWith = ReplaceWith(
        "OptimizedLLMRuntime.create(apertusNetwork(config), tensors, resolver, ctx)",
        "sk.ainet.apps.llm.OptimizedLLMRuntime"
    )
)
public class ApertusRuntime<T : DType>(
    private val ctx: ExecutionContext,
    val weights: ApertusRuntimeWeights<T>,
    private val attentionBackend: ApertusAttentionBackend<T>,
    private val dtype: KClass<T>,
    private val eps: Float = weights.metadata.rmsNormEps,
    random: Random = Random.Default
) : DecoderRuntime<T>(random) {

    // ---- DecoderRuntime abstract properties ----
    override val dim: Int = weights.metadata.embeddingLength
    override val seqLen: Int = weights.metadata.contextLength
    override val vocabSize: Int = weights.metadata.vocabSize
    override val nLayers: Int = weights.layers.size
    override val bosToken: Int = weights.metadata.bosTokenId

    private val nHeads = weights.metadata.headCount
    private val headDim = dim / nHeads
    private val nKvHeads = weights.metadata.kvHeadCount
    private val kvDim = nKvHeads * headDim

    private val embedding = Embedding(
        numEmbeddings = vocabSize,
        embeddingDim = dim,
        initWeight = weights.tokenEmbedding,
        name = "token_embd"
    )

    private val outputNormLayer = RMSNormalization<T, Float>(
        normalizedShape = intArrayOf(dim),
        eps = eps.toDouble(),
        name = "output_norm",
        initWeight = weights.outputNorm
    )

    private val attnNorms = weights.layers.mapIndexed { i, layer ->
        RMSNormalization<T, Float>(
            normalizedShape = intArrayOf(dim),
            eps = eps.toDouble(),
            name = "layer_$i.attn_norm",
            initWeight = layer.attnNorm
        )
    }

    private val ffnNorms = weights.layers.mapIndexed { i, layer ->
        RMSNormalization<T, Float>(
            normalizedShape = intArrayOf(dim),
            eps = eps.toDouble(),
            name = "layer_$i.ffn_norm",
            initWeight = layer.ffnNorm
        )
    }

    private val outputWeightT: Tensor<T, Float> = weights.outputWeight.t()

    // ---- DecoderRuntime template methods ----

    override fun embedToken(tokenId: Int): Tensor<T, Float> =
        embedding.forward(intArrayOf(tokenId), ctx)

    override fun runLayer(layerIdx: Int, x: Tensor<T, Float>): Tensor<T, Float> {
        val layer = weights.layers[layerIdx]

        // 1. Attention norm
        val attnNorm = attnNorms[layerIdx].forward(x, ctx)

        // 2. QKV projections (transpose on the fly to avoid double-memory peak)
        val q = attnNorm.matmul(layer.wq.t())
        val k = attnNorm.matmul(layer.wk.t())
        val v = attnNorm.matmul(layer.wv.t())

        // 3. QK-norm: per-head RMSNorm on Q and K
        val qNormed = applyPerHeadRMSNorm(q, nHeads, headDim, layer.qNorm)
        val kNormed = applyPerHeadRMSNorm(k, nKvHeads, headDim, layer.kNorm)

        // 4. Attention (RoPE + KV cache + GQA) — backend receives QK-normed tensors
        val attnOut = attentionBackend.attention(qNormed, kNormed, v, layerIdx, position)

        // 5. Output projection + residual
        val afterAttn = x + attnOut.matmul(layer.wo.t())

        // 6. FFN norm
        val ffnNorm = ffnNorms[layerIdx].forward(afterAttn, ctx)

        // 7. Ungated MLP: up → xIELU → down
        val up = ffnNorm.matmul(layer.ffnUp.t())
        val activated = applyXIELU(up, layer.xieluParams)
        val ffnOut = activated.matmul(layer.ffnDown.t())

        // 8. Residual
        return afterAttn + ffnOut
    }

    override fun outputNorm(x: Tensor<T, Float>): Tensor<T, Float> =
        outputNormLayer.forward(x, ctx)

    override fun outputProject(x: Tensor<T, Float>): Tensor<T, Float> =
        x.matmul(outputWeightT)

    override fun resetState() {
        attentionBackend.reset()
    }

    // ---- Apertus-specific helpers ----

    /**
     * Apply per-head RMSNorm to Q or K tensor.
     *
     * Input shape: [batch, nHeads * headDim]
     * Weight shape: [nHeads * headDim] (per-head, so each head has `headDim` weight values)
     */
    private fun applyPerHeadRMSNorm(
        x: Tensor<T, Float>,
        numHeads: Int,
        headDim: Int,
        weight: Tensor<T, Float>
    ): Tensor<T, Float> {
        val buf = x.expectFloatBuffer().copyOf()
        val w = weight.expectFloatBuffer()
        val totalDim = numHeads * headDim

        // Handle batched input
        val batchSize = if (x.shape.rank == 2) x.shape[0] else 1

        for (b in 0 until batchSize) {
            val batchOffset = b * totalDim
            for (h in 0 until numHeads) {
                val headOffset = batchOffset + h * headDim

                // Compute RMS for this head
                var sumSq = 0f
                for (i in 0 until headDim) {
                    val v = buf[headOffset + i]
                    sumSq += v * v
                }
                val rms = sqrt(sumSq / headDim + eps)

                // Normalize and scale (weight is per-head, shared across all heads)
                for (i in 0 until headDim) {
                    buf[headOffset + i] = (buf[headOffset + i] / rms) * w[i]
                }
            }
        }

        val shape = x.shape
        return ctx.fromFloatArray<T, Float>(shape, dtype, buf)
    }

    /**
     * Apply xIELU activation element-wise.
     *
     * xIELU formula:
     * ```
     * alpha_p_eff = softplus(alpha_p)       // ln(1 + exp(stored_alpha_p))
     * alpha_n_eff = beta + softplus(alpha_n)
     * if x > 0:  alpha_p_eff * x^2 + beta * x
     * if x <= 0: (expm1(min(x, eps)) - x) * alpha_n_eff + beta * x
     * ```
     */
    private fun applyXIELU(x: Tensor<T, Float>, params: ApertusXIELUParams): Tensor<T, Float> {
        val buf = x.expectFloatBuffer().copyOf()
        xielu(buf, params)
        return ctx.fromFloatArray<T, Float>(x.shape, dtype, buf)
    }

}

// ========== xIELU Implementation ==========

/**
 * In-place xIELU activation on a float buffer.
 *
 * Public for unit testing.
 */
public fun xielu(buf: FloatArray, params: ApertusXIELUParams) {
    val alphaPEff = softplus(params.alphaP)
    val alphaNEff = params.beta + softplus(params.alphaN)

    for (i in buf.indices) {
        val x = buf[i]
        buf[i] = if (x > 0f) {
            alphaPEff * x * x + params.beta * x
        } else {
            val clamped = min(x, params.eps)
            (expm1(clamped) - x) * alphaNEff + params.beta * x
        }
    }
}

/**
 * softplus(x) = ln(1 + exp(x))
 *
 * Uses a numerically stable formulation:
 * - For large x: softplus(x) ≈ x
 * - For small x: use the exact formula
 */
public fun softplus(x: Float): Float {
    return if (x > 20f) x else ln(1f + exp(x))
}

/**
 * exp(x) - 1, avoiding catastrophic cancellation near zero.
 */
private fun expm1(x: Float): Float = kotlin.math.expm1(x.toDouble()).toFloat()
