package sk.ainet.models.gemma

import kotlin.random.Random
import sk.ainet.apps.llm.DecoderRuntime
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.layers.Embedding
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.matmul
import sk.ainet.lang.tensor.plus
import sk.ainet.lang.tensor.t
import sk.ainet.lang.tensor.times
import sk.ainet.lang.nn.normalization.RMSNormalization
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Gemma 4 decoder runtime with pluggable attention backend.
 *
 * Key differences from Gemma 3n:
 * - No AltUp or activation sparsity
 * - Per-layer varying Q/K/V dimensions (global_head_dim vs head_dim)
 * - Proportional RoPE delegated to attention backend
 * - GELU activation (same as Gemma 3n)
 *
 * Extends [DecoderRuntime] for shared forward/generate/sample logic.
 */
@Deprecated(
    message = "Use gemmaNetwork() + OptimizedLLMRuntime (DIRECT mode) via GemmaNetworkLoader. " +
        "The DSL path reproduces this runtime's output at FP32 machine precision across every " +
        "feature — 1-layer global, mixed sliding+global, and shared-KV configurations — see " +
        "GemmaRuntimeParityTest. Follows the same deprecation pattern as LlamaRuntime / ApertusRuntime.",
    level = DeprecationLevel.WARNING
)
public class Gemma4Runtime<T : DType>(
    private val ctx: ExecutionContext,
    public val weights: Gemma4RuntimeWeights<T>,
    private val attentionBackend: AttentionBackend<T>,
    private val dtype: KClass<T>,
    private val config: Gemma4Config,
    private val eps: Float = 1e-6f,
    random: Random = Random.Default
) : DecoderRuntime<T>(random) {

    override val dim: Int = config.hiddenSize
    override val seqLen: Int = config.maxPositionEmbeddings
    override val vocabSize: Int = weights.metadata.vocabSize
    override val nLayers: Int = weights.layers.size
    override val bosToken: Int = weights.metadata.bosTokenId

    private val embedding = Embedding(
        numEmbeddings = vocabSize,
        embeddingDim = dim,
        initWeight = weights.tokenEmbedding,
        name = "token_embd"
    )

    private val finalNormLayer = RMSNormalization<T, Float>(
        normalizedShape = intArrayOf(dim),
        eps = eps.toDouble(),
        name = "final_norm",
        initWeight = weights.finalNorm
    )

    private val inputLayernorms = weights.layers.mapIndexed { i, layer ->
        RMSNormalization<T, Float>(
            normalizedShape = intArrayOf(dim),
            eps = eps.toDouble(),
            name = "layer_$i.input_layernorm",
            initWeight = layer.inputLayernorm
        )
    }

    private val postAttentionLayernorms = weights.layers.mapIndexed { i, layer ->
        RMSNormalization<T, Float>(
            normalizedShape = intArrayOf(dim),
            eps = eps.toDouble(),
            name = "layer_$i.post_attention_layernorm",
            initWeight = layer.postAttentionLayernorm
        )
    }

    public val currentPosition: Int
        get() = position

    override fun embedToken(tokenId: Int): Tensor<T, Float> =
        embedding.forward(intArrayOf(tokenId), ctx)

    override fun runLayer(layerIdx: Int, x: Tensor<T, Float>): Tensor<T, Float> {
        val layer = weights.layers[layerIdx]

        // Pre-attention normalization
        val attnNorm = inputLayernorms[layerIdx].forward(x, ctx)

        // QKV projections
        val q = attnNorm.matmul(layer.wq.t())
        val k = attnNorm.matmul(layer.wk.t())
        val v = attnNorm.matmul(layer.wv.t())

        // Delegate attention (RoPE + KV cache + scoring) to backend
        val attnOut = attentionBackend.attention(q, k, v, layerIdx, position)

        // Output projection + residual
        val afterAttn = x + attnOut.matmul(layer.wo.t())

        // Pre-FFN normalization
        val ffnNorm = postAttentionLayernorms[layerIdx].forward(afterAttn, ctx)

        // FFN with GELU activation
        val gate = ffnNorm.matmul(layer.gateProj.t()).gelu()
        val up = ffnNorm.matmul(layer.upProj.t())
        val ffnOut = (gate * up).matmul(layer.downProj.t())

        return afterAttn + ffnOut
    }

    override fun outputNorm(x: Tensor<T, Float>): Tensor<T, Float> =
        finalNormLayer.forward(x, ctx)

    override fun outputProject(x: Tensor<T, Float>): Tensor<T, Float> =
        x.matmul(weights.lmHead.t())

    override fun resetState() {
        attentionBackend.reset()
    }

    /**
     * GELU (Gaussian Error Linear Unit) activation.
     * Gemma uses GELU instead of SiLU.
     */
    private fun Tensor<T, Float>.gelu(): Tensor<T, Float> {
        val buf = expectFloatBuffer()
        val out = FloatArray(buf.size)

        val sqrtTwoPi = 0.7978845608028654f
        val c = 0.044715f

        for (i in buf.indices) {
            val x = buf[i]
            val x3 = x * x * x
            val inner = sqrtTwoPi * (x + c * x3)
            val tanh = kotlin.math.tanh(inner.toDouble()).toFloat()
            out[i] = 0.5f * x * (1f + tanh)
        }

        return ctx.fromFloatArray(this.shape, dtype, out)
    }
}
