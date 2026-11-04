package sk.ainet.models.lfm2

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin

data class ModelDimensions(
    val width: Int, val vocabulary: Int, val queryHeads: Int, val kvHeads: Int,
    val headDim: Int, val intermediate: Int, val kernel: Int, val epsilon: Float,
    val theta: Double, val layerTypes: List<String>, val contextLimit: Int = 2048,
) {
    init {
        require(width > 0 && vocabulary > 0 && queryHeads > 0 && kvHeads > 0 && headDim > 0 && headDim % 2 == 0)
        require(width == queryHeads * headDim && queryHeads % kvHeads == 0 && intermediate > 0 && kernel > 0 && contextLimit > 0)
        require(epsilon.isFinite() && epsilon > 0f && theta.isFinite() && theta > 0)
        require(layerTypes.isNotEmpty() && layerTypes.all { it in setOf("conv", "full_attention") })
    }
}

/** One request owns all hybrid state. Reset never relies on a reference runtime's cache helper. */
class HybridModelState {
    var position: Int = 0
        private set
    private val convolutionState = mutableMapOf<Int, Tensor<FP32, Float>>()
    private val keyState = mutableMapOf<Int, Tensor<FP32, Float>>()
    private val valueState = mutableMapOf<Int, Tensor<FP32, Float>>()
    val convolution: Map<Int, Tensor<FP32, Float>> get() = convolutionState
    val keys: Map<Int, Tensor<FP32, Float>> get() = keyState
    val values: Map<Int, Tensor<FP32, Float>> get() = valueState

    fun reset() {
        position = 0; convolutionState.clear(); keyState.clear(); valueState.clear()
    }

    internal fun commit(nextPosition: Int, convolution: Map<Int, Tensor<FP32, Float>>, keys: Map<Int, Tensor<FP32, Float>>, values: Map<Int, Tensor<FP32, Float>>) {
        convolutionState.clear(); convolutionState.putAll(convolution)
        keyState.clear(); keyState.putAll(keys)
        valueState.clear(); valueState.putAll(values)
        position = nextPosition
    }
}

data class ModelChunk(
    val logits: Tensor<FP32, Float>, val finalNorm: Tensor<FP32, Float>, val layerOutputs: Map<Int, Tensor<FP32, Float>>,
)

data class GreedyResult(val generatedIds: IntArray, val stopReason: String, val stateInputTokens: Int)

/** Eager full LFM2 network. Host embedding gathering and position/mask creation stay outside block graphs. */
class HybridLfm2Model(val dimensions: ModelDimensions, val ctx: DirectCpuExecutionContext, weights: Map<String, Tensor<FP32, Float>>,
    val rmsReduction: RmsReductionPolicy = RmsReductionPolicy.BALANCED_FP32,
) {
    companion object {
        /** The loader/export contract: source tensor names and layouts for one configured architecture. */
        fun expectedWeights(dim: ModelDimensions): LinkedHashMap<String, Shape> = linkedMapOf<String, Shape>().apply {
            put("model.embed_tokens.weight", Shape(dim.vocabulary, dim.width))
            put("model.embedding_norm.weight", Shape(dim.width))
            for ((index, kind) in dim.layerTypes.withIndex()) {
                val prefix = "model.layers.$index."
                put(prefix + "operator_norm.weight", Shape(dim.width))
                put(prefix + "ffn_norm.weight", Shape(dim.width))
                put(prefix + "feed_forward.w1.weight", Shape(dim.intermediate, dim.width))
                put(prefix + "feed_forward.w2.weight", Shape(dim.width, dim.intermediate))
                put(prefix + "feed_forward.w3.weight", Shape(dim.intermediate, dim.width))
                if (kind == "conv") {
                    put(prefix + "conv.in_proj.weight", Shape(3 * dim.width, dim.width))
                    put(prefix + "conv.conv.weight", Shape(dim.width, 1, dim.kernel))
                    put(prefix + "conv.out_proj.weight", Shape(dim.width, dim.width))
                } else {
                    put(prefix + "self_attn.q_proj.weight", Shape(dim.width, dim.width))
                    put(prefix + "self_attn.k_proj.weight", Shape(dim.kvHeads * dim.headDim, dim.width))
                    put(prefix + "self_attn.v_proj.weight", Shape(dim.kvHeads * dim.headDim, dim.width))
                    put(prefix + "self_attn.out_proj.weight", Shape(dim.width, dim.width))
                    put(prefix + "self_attn.q_layernorm.weight", Shape(dim.headDim))
                    put(prefix + "self_attn.k_layernorm.weight", Shape(dim.headDim))
                }
            }
        }
    }

    private val embedding = weights.getValue("model.embed_tokens.weight")
    private val finalNorm = weights.getValue("model.embedding_norm.weight")
    private val convolution = mutableMapOf<Int, BlockTensorOp>()
    private val attention = mutableMapOf<Int, AttentionTensorOp>()

    init {
        val expected = expectedWeights(dimensions)
        require(weights.keys == expected.keys && expected.all { (name, shape) -> weights.getValue(name).shape == shape }) { "Full-model tensor names/layouts differ from configured architecture" }
        for ((index, kind) in dimensions.layerTypes.withIndex()) {
            val prefix = "model.layers.$index."
            fun w(name: String) = weights.getValue(prefix + name + ".weight")
            if (kind == "conv") {
                convolution[index] = BlockTensorOp(dimensions.width, dimensions.kernel, dimensions.intermediate, dimensions.epsilon,
                    ShortConvTensorOp(dimensions.width, dimensions.kernel, w("conv.in_proj"), w("conv.conv"), w("conv.out_proj")),
                    w("operator_norm"), w("ffn_norm"), w("feed_forward.w1"), w("feed_forward.w2"), w("feed_forward.w3"), rmsReduction)
            } else {
                attention[index] = AttentionTensorOp(dimensions.width, dimensions.queryHeads, dimensions.kvHeads, dimensions.headDim, dimensions.intermediate, dimensions.epsilon,
                    w("operator_norm"), w("ffn_norm"), w("self_attn.q_proj"), w("self_attn.k_proj"), w("self_attn.v_proj"), w("self_attn.out_proj"),
                    w("self_attn.q_layernorm"), w("self_attn.k_layernorm"), w("feed_forward.w1"), w("feed_forward.w2"), w("feed_forward.w3"), rmsReduction)
            }
        }
    }

    fun forward(inputIds: IntArray, state: HybridModelState): ModelChunk {
        val d = dimensions
        require(inputIds.isNotEmpty() && inputIds.all { it in 0 until d.vocabulary }) { "Input IDs must be nonempty and inside the vocabulary" }
        require(state.position.toLong() + inputIds.size <= d.contextLimit) { "Request exceeds the configured evaluation context limit" }
        val past = state.position
        require((past == 0 && state.convolution.isEmpty() && state.keys.isEmpty() && state.values.isEmpty()) ||
            (past > 0 && state.convolution.keys == convolution.keys && state.keys.keys == attention.keys && state.values.keys == attention.keys)) { "Hybrid state is incomplete or inconsistent" }
        val length = inputIds.size
        val embeddings = FloatArray(length * d.width) { index -> embedding.data[inputIds[index / d.width], index % d.width] }
        var hidden = ctx.fromFloatArray<FP32, Float>(Shape(length, d.width), FP32::class, embeddings)
        val cosines = FloatArray(length * d.headDim)
        val sines = FloatArray(length * d.headDim)
        for (token in 0 until length) for (index in 0 until d.headDim / 2) {
            val angle = (past + token) * d.theta.pow(-2.0 * index / d.headDim)
            val c = cos(angle).toFloat(); val s = sin(angle).toFloat()
            cosines[token * d.headDim + index] = c; cosines[token * d.headDim + index + d.headDim / 2] = c
            sines[token * d.headDim + index] = s; sines[token * d.headDim + index + d.headDim / 2] = s
        }
        val cosine = ctx.fromFloatArray<FP32, Float>(Shape(length, d.headDim), FP32::class, cosines)
        val sine = ctx.fromFloatArray<FP32, Float>(Shape(length, d.headDim), FP32::class, sines)
        val maskValues = FloatArray(length * (past + length)) { index ->
            if (index % (past + length) <= past + index / (past + length)) 0f else -Float.MAX_VALUE
        }
        val mask = ctx.fromFloatArray<FP32, Float>(Shape(length, past + length), FP32::class, maskValues)
        val nextConvolution = state.convolution.toMutableMap()
        val nextKeys = state.keys.toMutableMap()
        val nextValues = state.values.toMutableMap()
        val layers = linkedMapOf<Int, Tensor<FP32, Float>>()
        for ((index, kind) in d.layerTypes.withIndex()) {
            hidden = if (kind == "conv") {
                val history = state.convolution[index] ?: ctx.fromFloatArray(Shape(d.width, d.kernel), FP32::class, FloatArray(d.width * d.kernel))
                val result = convolution.getValue(index).forward(hidden, history, ctx)
                nextConvolution[index] = result.nextHistory
                result.output
            } else {
                val key = state.keys[index]; val value = state.values[index]
                require(past == 0 || (key?.shape == Shape(d.kvHeads, past, d.headDim) && value?.shape == Shape(d.kvHeads, past, d.headDim))) { "Attention cache position does not match global position" }
                val result = attention.getValue(index).forward(hidden, key, value, cosine, sine, mask, ctx)
                nextKeys[index] = result.nextKey; nextValues[index] = result.nextValue
                result.output
            }
            layers[index] = hidden
        }
        val ops = ctx.ops
        val normalized = rmsNormalize(hidden, finalNorm, ctx, rmsReduction, d.epsilon)
        val logits = ops.matmulWeightTransposed(normalized, embedding)
        state.commit(past + length, nextConvolution, nextKeys, nextValues)
        return ModelChunk(logits, normalized, layers)
    }

    fun generateGreedy(promptIds: IntArray, maxNewTokens: Int, stopTokenIds: Set<Int>,
        observeLogits: (Int, Tensor<FP32, Float>) -> Unit = { _, _ -> },
    ): GreedyResult {
        require(maxNewTokens in 1..32 && promptIds.isNotEmpty()) { "Greedy evaluation requires 1..32 new tokens and a nonempty prompt" }
        require(promptIds.size.toLong() + maxNewTokens - 1 <= dimensions.contextLimit) { "Greedy request exceeds the configured evaluation context limit" }
        require(stopTokenIds.all { it in 0 until dimensions.vocabulary }) { "Stop token ID outside vocabulary" }
        val state = HybridModelState()
        var result = forward(promptIds, state)
        val generated = mutableListOf<Int>()
        var stopReason = "length"
        for (step in 0 until maxNewTokens) {
            val last = ctx.ops.narrow(result.logits, dim = 0, start = result.logits.shape[0] - 1, length = 1)
            observeLogits(step, last)
            val values = last.data.copyToFloatArray()
            require(values.all { it.isFinite() }) { "Non-finite greedy logits" }
            var token = 0
            for (index in 1 until values.size) if (values[index] > values[token]) token = index
            generated.add(token)
            if (token in stopTokenIds) { stopReason = "eos"; break }
            if (step + 1 < maxNewTokens) result = forward(intArrayOf(token), state)
        }
        return GreedyResult(generated.toIntArray(), stopReason, state.position)
    }
}

