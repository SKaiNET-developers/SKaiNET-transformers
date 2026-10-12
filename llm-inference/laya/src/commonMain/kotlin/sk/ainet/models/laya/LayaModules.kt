package sk.ainet.models.laya

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.layers.Embedding
import sk.ainet.lang.nn.layers.EmbeddingAdapter
import sk.ainet.lang.nn.normalization.LayerNormalization
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.nn.transformer.MultiHeadAttention
import sk.ainet.lang.nn.transformer.ResidualAdd
import sk.ainet.lang.nn.transformer.RoPE
import sk.ainet.lang.nn.transformer.RoPEMode
import sk.ainet.lang.nn.transformer.TransformerBlock
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.DType
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.sqrt
import kotlin.reflect.KClass

/**
 * Exact (erf-based) GELU, the form `nn.GELU()` and ModernBERT's `hidden_activation: gelu`
 * compute (REQ-EXE-011). The engine's `gelu` op is the tanh approximation, so this runs on the
 * element data directly; [exact] = false selects the engine op to measure the difference.
 */
public class LayaGelu<T : DType, V>(
    private val dtype: KClass<T>,
    public val exact: Boolean = true,
    override val name: String = "gelu",
) : Module<T, V>() {
    override val modules: List<Module<T, V>> = emptyList()

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        if (!exact) return ctx.ops.gelu(input)
        val out = input.data.copyToFloatArray()
        for (i in out.indices) out[i] = geluErf(out[i])
        return ctx.fromFloatArray(input.shape, dtype, out)
    }

    public companion object {
        /** `0.5 · x · (1 + erf(x / √2))` in double precision. */
        public fun geluErf(x: Float): Float {
            val xd = x.toDouble()
            return (0.5 * xd * (1.0 + erf(xd / sqrt(2.0)))).toFloat()
        }

        /**
         * erf with absolute error below 1.5e-7 (Abramowitz and Stegun 7.1.26), evaluated in
         * double; the result is consumed as a float, where that error sits below the last bit.
         */
        public fun erf(x: Double): Double {
            val sign = if (x < 0) -1.0 else 1.0
            val a = abs(x)
            val t = 1.0 / (1.0 + 0.3275911 * a)
            val y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a)
            return sign * y
        }
    }
}

/** `x · Wᵀ (+ b)` with the Hugging Face weight layout `[out, in]`; parameters `name.weight`, `name.bias`. */
public class LayaLinear<T : DType, V>(
    public val inFeatures: Int,
    public val outFeatures: Int,
    public val bias: Boolean,
    dtype: KClass<T>,
    override val name: String,
) : Module<T, V>(), ModuleParameters<T, V> {
    override val params: List<ModuleParameter<T, V>> = buildList {
        add(ModuleParameter.WeightParameter("$name.weight", voidTensor(dtype, outFeatures, inFeatures)))
        if (bias) add(ModuleParameter.BiasParameter("$name.bias", voidTensor(dtype, outFeatures)))
    }
    override val modules: List<Module<T, V>> = emptyList()

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        val y = ctx.ops.matmulWeightTransposed(input, params[0].value)
        return if (bias) ctx.ops.add(y, params[1].value) else y
    }
}

/** ReLU as a module, for the head's `linear2(relu(linear1(x)))`. */
public class LayaRelu<T : DType, V>(override val name: String = "relu") : Module<T, V>() {
    override val modules: List<Module<T, V>> = emptyList()
    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> = ctx.ops.relu(input)
}

/**
 * ModernBERT's GeGLU feed-forward with the exact GELU: `Wo · (gelu(Wi_input · x) * (Wi_gate · x))`.
 * Parameters follow the engine's `GeGLUFFN` naming so one resolver serves both:
 * `gate_proj` is the activated half of the fused `Wi`, `up_proj` the multiplier half.
 */
public class LayaGeGLU<T : DType, V>(
    public val dim: Int,
    public val hiddenDim: Int,
    dtype: KClass<T>,
    exactGelu: Boolean,
    override val name: String = "mlp",
) : Module<T, V>(), ModuleParameters<T, V> {
    private val gelu = LayaGelu<T, V>(dtype, exactGelu, "$name.act")
    override val params: List<ModuleParameter<T, V>> = listOf(
        ModuleParameter.WeightParameter("$name.gate_proj.weight", voidTensor(dtype, hiddenDim, dim)),
        ModuleParameter.WeightParameter("$name.up_proj.weight", voidTensor(dtype, hiddenDim, dim)),
        ModuleParameter.WeightParameter("$name.down_proj.weight", voidTensor(dtype, dim, hiddenDim)),
    )
    override val modules: List<Module<T, V>> = emptyList()

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        val ops = ctx.ops
        val activated = gelu.forward(ops.matmulWeightTransposed(input, params[0].value), ctx)
        val gate = ops.matmulWeightTransposed(input, params[1].value)
        return ops.matmulWeightTransposed(ops.multiply(activated, gate), params[2].value)
    }
}

/** Runs its children in order; the container the encoder and head stacks are built from. */
public class LayaSequential<T : DType, V>(
    override val modules: List<Module<T, V>>,
    override val name: String,
) : Module<T, V>() {
    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        var x = input
        for (m in modules) x = m.forward(x, ctx)
        return x
    }
}

/**
 * Builders for the module tree of a Laya checkpoint. Module paths (`encoder/layers.3/attn`,
 * `head/layers.0/self_attn`, `scorer/0`) are what [LayaSafeTensorsNameResolver] maps to tensor names.
 */
public object LayaNetwork {

    /** The ModernBERT / mmBERT encoder: embeddings, pre-norm layers with alternating attention, final norm. */
    public fun <T : DType, V> encoder(cfg: LayaEncoderConfig, maxSeqLen: Int, dtype: KClass<T>, exactGelu: Boolean): Module<T, V> {
        val d = cfg.hiddenSize
        val modules = ArrayList<Module<T, V>>()
        modules += LayaSequential(
            listOf(
                EmbeddingAdapter(Embedding(numEmbeddings = cfg.vocabSize, embeddingDim = d, initWeight = voidTensor(dtype, cfg.vocabSize, d), name = "tok_embeddings")),
                LayerNormalization(intArrayOf(d), eps = cfg.normEps, name = "norm", dtype = dtype),
            ),
            name = "embeddings",
        )
        val kinds = cfg.layerKinds
        for (i in 0 until cfg.numHiddenLayers) {
            val kind = kinds[i]
            val attention = MultiHeadAttention<T, V>(
                dim = d,
                nHeads = cfg.numAttentionHeads,
                causal = false,
                bias = cfg.attentionBias,
                name = "attn",
                slidingWindow = if (kind == LayaLayerKind.SLIDING) cfg.localHalfWindow + 1 else null,
                rightContext = if (kind == LayaLayerKind.SLIDING) cfg.localHalfWindow else 0,
                dtype = dtype,
            )
            attention.rope = RoPE(headDim = cfg.headDim, maxSeqLen = maxSeqLen, base = cfg.ropeTheta(kind).toFloat(), mode = RoPEMode.SPLIT_HALF)
            val layer = ArrayList<Module<T, V>>()
            // ModernBERT skips the attention norm on layer 0: the embedding norm precedes it.
            if (i > 0) layer += LayerNormalization(intArrayOf(d), eps = cfg.normEps, name = "attn_norm", dtype = dtype)
            layer += attention
            layer += ResidualAdd()
            layer += LayerNormalization(intArrayOf(d), eps = cfg.normEps, name = "mlp_norm", dtype = dtype)
            layer += LayaGeGLU(d, cfg.intermediateSize, dtype, exactGelu, name = "mlp")
            layer += ResidualAdd()
            modules += TransformerBlock(layer, name = "layers.$i")
        }
        modules += LayerNormalization(intArrayOf(d), eps = cfg.normEps, name = "final_norm", dtype = dtype)
        return LayaSequential(modules, name = "encoder")
    }

    /**
     * The decision head: `head_layers` PyTorch `TransformerEncoderLayer` blocks with
     * `norm_first = true`, biased attention without positional encoding, and a ReLU feed-forward.
     */
    public fun <T : DType, V> head(hiddenSize: Int, headLayers: Int, dtype: KClass<T>): Module<T, V> {
        val nHeads = maxOf(1, hiddenSize / 64)
        val blocks = ArrayList<Module<T, V>>()
        for (i in 0 until headLayers) {
            val attention = MultiHeadAttention<T, V>(dim = hiddenSize, nHeads = nHeads, causal = false, bias = true, name = "self_attn", dtype = dtype)
            blocks += TransformerBlock(
                listOf(
                    LayerNormalization(intArrayOf(hiddenSize), eps = 1e-5, name = "norm1", dtype = dtype),
                    attention,
                    ResidualAdd(),
                    LayerNormalization(intArrayOf(hiddenSize), eps = 1e-5, name = "norm2", dtype = dtype),
                    LayaSequential(
                        listOf(
                            LayaLinear(hiddenSize, 4 * hiddenSize, bias = true, dtype = dtype, name = "linear1"),
                            LayaRelu(),
                            LayaLinear(4 * hiddenSize, hiddenSize, bias = true, dtype = dtype, name = "linear2"),
                        ),
                        name = "ffn",
                    ),
                    ResidualAdd(),
                ),
                name = "layers.$i",
            )
        }
        return LayaSequential(blocks, name = "head")
    }

    /** `nn.Sequential(LayerNorm(d), Linear(d, d), GELU(), Linear(d, 1))`. */
    public fun <T : DType, V> scorer(hiddenSize: Int, dtype: KClass<T>, exactGelu: Boolean): Module<T, V> = LayaSequential(
        listOf(
            LayerNormalization(intArrayOf(hiddenSize), eps = 1e-5, name = "0", dtype = dtype),
            LayaLinear(hiddenSize, hiddenSize, bias = true, dtype = dtype, name = "1"),
            LayaGelu(dtype, exactGelu, name = "2"),
            LayaLinear(hiddenSize, 1, bias = true, dtype = dtype, name = "3"),
        ),
        name = "scorer",
    )
}

@Suppress("UNCHECKED_CAST")
internal fun <T : DType, V> voidTensor(dtype: KClass<T>, vararg dims: Int): Tensor<T, V> = VoidOpsTensor(
    object : TensorData<T, V> {
        override val shape = Shape(*dims)
        override fun get(vararg indices: Int): V = 0.0f as V
        override fun set(vararg indices: Int, value: V) {}
    },
    dtype,
)
