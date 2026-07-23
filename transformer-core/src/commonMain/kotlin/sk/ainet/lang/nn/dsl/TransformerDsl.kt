package sk.ainet.lang.nn.dsl

import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.layers.Embedding
import sk.ainet.lang.nn.layers.EmbeddingAdapter
import sk.ainet.lang.nn.layers.EmbeddingParams
import sk.ainet.lang.nn.normalization.RMSNormalization
import sk.ainet.lang.nn.transformer.AppendKVCache
import sk.ainet.lang.nn.transformer.KVCache
import sk.ainet.lang.nn.transformer.MultiHeadAttention
import sk.ainet.lang.nn.transformer.ResidualAdd
import sk.ainet.lang.nn.transformer.RoPE
import sk.ainet.lang.nn.transformer.RoPEMode
import sk.ainet.lang.nn.transformer.RoPEScaling
import sk.ainet.lang.nn.transformer.GeGLUFFN
import sk.ainet.lang.nn.transformer.SwiGLUFFN
import sk.ainet.lang.nn.transformer.XIELUActivation
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Transformer-specific DSL extensions for building LLM network definitions.
 *
 * These functions extend [StageImpl] and [NeuralNetworkDslImpl] to add
 * transformer layer creation (embedding, RMSNorm, multi-head attention,
 * SwiGLU FFN, residual connections).
 *
 * Usage in a `sequential {}` block:
 * ```kotlin
 * sequential<FP32, Float> {
 *     embedding(vocabSize, dim, id = "token_embd")
 *     rmsNorm(dim, id = "output_norm")
 *     dense(vocabSize, id = "output")
 * }
 * ```
 *
 * Usage with [StageImpl] for transformer layers:
 * ```kotlin
 * val stage = StageImpl<T, V>(ctx, "blk.0", T::class)
 * stage.rmsNorm(dim, id = "attn_norm")
 * stage.multiHeadAttention(dim, nHeads, nKVHeads) { rope(headDim, seqLen); kvCache(...) }
 * stage.residual()
 * ```
 */

// ============================================================================
// ATTENTION builder interface + implementation
// ============================================================================

@NetworkDsl
public interface ATTENTION<T : DType, V> : NetworkDslItem {
    public fun rope(
        headDim: Int,
        maxSeqLen: Int,
        mode: RoPEMode = RoPEMode.INTERLEAVED,
        base: Float = 10000.0f,
        scaling: RoPEScaling = RoPEScaling.NONE,
        scalingFactor: Float = 1.0f,
        partialRotaryFactor: Float = 1.0f,
        freqDenomRotaryDim: Boolean = false
    )
    /** Build a default [AppendKVCache] in place. */
    public fun kvCache(maxSeqLen: Int, nKVHeads: Int, headDim: Int)
    /** Attach a pre-built KV cache variant (e.g. [SlidingWindowKVCache] or [SharedKVCache]). */
    public fun kvCache(cache: KVCache<T, V>)
}

public class AttentionImpl<T : DType, V>(
    override val executionContext: ExecutionContext,
    private val dim: Int,
    private val nHeads: Int,
    private val nKVHeads: Int,
    private val causal: Boolean,
    private val qkNorm: Boolean,
    private val qkNormUnitOffset: Boolean = false,
    private val qkNormEps: Double = 1e-5,
    private val attentionScale: Float? = null,
    private val vNormNoScale: Boolean = false,
    private val bias: Boolean,
    private val id: String,
    private val slidingWindow: Int? = null,
    private val rightContext: Int = 0,
    private val kClass: kotlin.reflect.KClass<T>? = null,
) : ATTENTION<T, V> {

    private var ropeModule: RoPE<T, V>? = null
    private var kvCacheModule: KVCache<T, V>? = null
    private var explicitHeadDim: Int? = null

    override fun rope(
        headDim: Int,
        maxSeqLen: Int,
        mode: RoPEMode,
        base: Float,
        scaling: RoPEScaling,
        scalingFactor: Float,
        partialRotaryFactor: Float,
        freqDenomRotaryDim: Boolean
    ) {
        ropeModule = RoPE(
            headDim = headDim,
            maxSeqLen = maxSeqLen,
            base = base,
            mode = mode,
            scaling = scaling,
            scalingFactor = scalingFactor,
            partialRotaryFactor = partialRotaryFactor,
            freqDenomRotaryDim = freqDenomRotaryDim,
            name = "$id.rope"
        )
        explicitHeadDim = headDim
    }

    override fun kvCache(maxSeqLen: Int, nKVHeads: Int, headDim: Int) {
        kvCacheModule = AppendKVCache(
            maxSeqLen = maxSeqLen,
            nKVHeads = nKVHeads,
            headDim = headDim,
            name = "$id.kv_cache"
        )
    }

    override fun kvCache(cache: KVCache<T, V>) {
        kvCacheModule = cache
    }

    public fun create(): MultiHeadAttention<T, V> {
        // Pass explicit headDim when it differs from dim/nHeads (e.g. Voxtral: dim=3072, head_dim=128, nHeads=32)
        val needsExplicitHeadDim = explicitHeadDim != null && explicitHeadDim != dim / nHeads
        return MultiHeadAttention(
            dim = dim,
            nHeads = nHeads,
            nKVHeads = nKVHeads,
            causal = causal,
            qkNorm = qkNorm,
            qkNormUnitOffset = qkNormUnitOffset,
            qkNormEps = qkNormEps,
            attentionScale = attentionScale,
            vNormNoScale = vNormNoScale,
            bias = bias,
            name = id,
            rope = ropeModule,
            kvCache = kvCacheModule,
            explicitHeadDim = if (needsExplicitHeadDim) explicitHeadDim else null,
            slidingWindow = slidingWindow,
            rightContext = rightContext,
            dtype = kClass
        )
    }
}

// ============================================================================
// Extension functions on StageImpl
// ============================================================================

public fun <T : DType, V> StageImpl<T, V>.embedding(vocabSize: Int, dim: Int, id: String = "") {
    @Suppress("UNCHECKED_CAST")
    val voidWeight = sk.ainet.lang.tensor.VoidOpsTensor(
        object : sk.ainet.lang.tensor.data.TensorData<T, V> {
            override val shape = sk.ainet.lang.tensor.Shape(vocabSize, dim)
            override fun get(vararg indices: Int): V = 0.0f as V
            override fun set(vararg indices: Int, value: V) {}
        },
        kClass
    )
    val emb = Embedding<T, V>(
        numEmbeddings = vocabSize,
        embeddingDim = dim,
        initWeight = voidWeight,
        name = getDefaultName(id, "Embedding", modules.size)
    )
    modules += EmbeddingAdapter(emb)
    lastDimension = dim
}

public fun <T : DType, V> StageImpl<T, V>.rmsNorm(normalizedShape: Int, eps: Float = 1e-5f, id: String = "", unitOffset: Boolean = false) {
    modules += RMSNormalization<T, V>(
        normalizedShape = intArrayOf(normalizedShape),
        eps = eps.toDouble(),
        name = getDefaultName(id, "RMSNorm", modules.size),
        unitOffset = unitOffset,
        dtype = kClass
    )
}

public fun <T : DType, V> StageImpl<T, V>.multiHeadAttention(
    dim: Int,
    nHeads: Int,
    nKVHeads: Int = nHeads,
    causal: Boolean = true,
    qkNorm: Boolean = false,
    qkNormUnitOffset: Boolean = false,
    qkNormEps: Float = 1e-5f,
    attentionScale: Float? = null,
    vNormNoScale: Boolean = false,
    bias: Boolean = false,
    id: String = "",
    slidingWindow: Int? = null,
    rightContext: Int = 0,
    content: ATTENTION<T, V>.() -> Unit = {}
) {
    val attnName = getDefaultName(id, "MultiHeadAttention", modules.size)
    val impl = AttentionImpl<T, V>(
        executionContext = executionContext,
        dim = dim,
        nHeads = nHeads,
        nKVHeads = nKVHeads,
        causal = causal,
        qkNorm = qkNorm,
        qkNormUnitOffset = qkNormUnitOffset,
        qkNormEps = qkNormEps.toDouble(),
        attentionScale = attentionScale,
        vNormNoScale = vNormNoScale,
        bias = bias,
        id = attnName,
        slidingWindow = slidingWindow,
        rightContext = rightContext,
        kClass = kClass,
    )
    impl.content()
    modules += impl.create()
}

public fun <T : DType, V> StageImpl<T, V>.swiGluFFN(dim: Int, hiddenDim: Int, id: String = "") {
    modules += SwiGLUFFN<T, V>(
        dim = dim,
        hiddenDim = hiddenDim,
        name = getDefaultName(id, "SwiGLUFFN", modules.size)
    )
}

public fun <T : DType, V> StageImpl<T, V>.geGluFFN(dim: Int, hiddenDim: Int, id: String = "") {
    modules += GeGLUFFN<T, V>(
        dim = dim,
        hiddenDim = hiddenDim,
        name = getDefaultName(id, "GeGLUFFN", modules.size),
        dtype = kClass
    )
}

public fun <T : DType, V> StageImpl<T, V>.xielu(id: String = "") {
    modules += XIELUActivation<T, V>(
        name = getDefaultName(id, "XIELUActivation", modules.size)
    )
}

public fun <T : DType, V> StageImpl<T, V>.residual() {
    modules += ResidualAdd<T, V>(
        name = getDefaultName("", "ResidualAdd", modules.size)
    )
}

// ============================================================================
// Extension functions on NeuralNetworkDslImpl
// ============================================================================

public fun <T : DType, V> NeuralNetworkDslImpl<T, V>.embedding(vocabSize: Int, dim: Int, id: String = "") {
    // Use VoidOpsTensor placeholder to avoid allocating a full [vocabSize, dim] random tensor.
    // The actual weights will be set by WeightMapper during model loading.
    @Suppress("UNCHECKED_CAST")
    val voidWeight = sk.ainet.lang.tensor.VoidOpsTensor(
        object : sk.ainet.lang.tensor.data.TensorData<T, V> {
            override val shape = sk.ainet.lang.tensor.Shape(vocabSize, dim)
            override fun get(vararg indices: Int): V = 0.0f as V
            override fun set(vararg indices: Int, value: V) {}
        },
        kClass
    )
    val emb = Embedding<T, V>(
        numEmbeddings = vocabSize,
        embeddingDim = dim,
        initWeight = voidWeight,
        name = getDefaultName(id, "Embedding", modules.size)
    )
    modules += EmbeddingAdapter(emb)
    lastDimension = dim
}

public fun <T : DType, V> NeuralNetworkDslImpl<T, V>.rmsNorm(normalizedShape: Int, eps: Float = 1e-5f, id: String = "", unitOffset: Boolean = false) {
    modules += RMSNormalization<T, V>(
        normalizedShape = intArrayOf(normalizedShape),
        eps = eps.toDouble(),
        name = getDefaultName(id, "RMSNorm", modules.size),
        unitOffset = unitOffset,
        dtype = kClass
    )
}

public fun <T : DType, V> NeuralNetworkDslImpl<T, V>.multiHeadAttention(
    dim: Int,
    nHeads: Int,
    nKVHeads: Int = nHeads,
    causal: Boolean = true,
    qkNorm: Boolean = false,
    qkNormEps: Float = 1e-5f,
    bias: Boolean = false,
    id: String = "",
    slidingWindow: Int? = null,
    rightContext: Int = 0,
    content: ATTENTION<T, V>.() -> Unit = {}
) {
    val attnName = getDefaultName(id, "MultiHeadAttention", modules.size)
    val impl = AttentionImpl<T, V>(
        executionContext = executionContext,
        dim = dim,
        nHeads = nHeads,
        nKVHeads = nKVHeads,
        causal = causal,
        qkNorm = qkNorm,
        qkNormEps = qkNormEps.toDouble(),
        bias = bias,
        id = attnName,
        slidingWindow = slidingWindow,
        rightContext = rightContext,
        kClass = kClass,
    )
    impl.content()
    modules += impl.create()
}

public fun <T : DType, V> NeuralNetworkDslImpl<T, V>.swiGluFFN(dim: Int, hiddenDim: Int, id: String = "") {
    modules += SwiGLUFFN<T, V>(
        dim = dim,
        hiddenDim = hiddenDim,
        name = getDefaultName(id, "SwiGLUFFN", modules.size)
    )
}

public fun <T : DType, V> NeuralNetworkDslImpl<T, V>.geGluFFN(dim: Int, hiddenDim: Int, id: String = "") {
    modules += GeGLUFFN<T, V>(
        dim = dim,
        hiddenDim = hiddenDim,
        name = getDefaultName(id, "GeGLUFFN", modules.size),
        dtype = kClass
    )
}

public fun <T : DType, V> NeuralNetworkDslImpl<T, V>.xielu(id: String = "") {
    modules += XIELUActivation<T, V>(
        name = getDefaultName(id, "XIELUActivation", modules.size)
    )
}

public fun <T : DType, V> NeuralNetworkDslImpl<T, V>.residual() {
    modules += ResidualAdd<T, V>(
        name = getDefaultName("", "ResidualAdd", modules.size)
    )
}
