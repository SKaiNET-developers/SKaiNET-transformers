package sk.ainet.models.lfm2

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

/** One fixed-capacity step: logits and all state to feed to the next invocation. */
data class Lfm2CachedTensorResult(
    val logits: Tensor<FP32, Float>,
    val states: LinkedHashMap<String, Tensor<FP32, Float>>,
)

/**
 * Materialize contiguous vocabulary-row shards before tracing. Each shard is at most 64 MiB.
 * The same resulting tensors must serve both embedding selection and the tied output head.
 */
fun splitLfm2Embedding(
    embedding: Tensor<FP32, Float>,
    ctx: ExecutionContext,
    rowsPerShard: Int = 16384,
): List<Tensor<FP32, Float>> {
    require(!ctx.isRecording) { "Embedding shards must be materialized outside graph recording" }
    require(embedding.shape.dimensions.size == 2 && embedding.shape[0] > 0 && embedding.shape[1] > 0)
    require(rowsPerShard > 0 && rowsPerShard.toLong() * embedding.shape[1] * 4L <= MAX_EMBEDDING_SHARD_BYTES) {
        "Embedding shard must fit the 64 MiB evaluation bound"
    }
    return (0 until embedding.shape[0] step rowsPerShard).map { first ->
        ctx.ops.narrow(embedding, dim = 0, start = first, length = minOf(rowsPerShard, embedding.shape[0] - first))
    }
}

private const val MAX_EMBEDDING_SHARD_BYTES: Long = 64L * 1024L * 1024L

/**
 * Complete single-token tensor network for sequential prompt ingestion and cached decoding.
 *
 * Input selection, every decoder block, tied output projection and cache updates use tensor
 * operations. The caller supplies one-hot token/control tensors, RoPE values and explicit
 * state. It must enforce position in 0 until [capacity], an unfilled initial cache, and one
 * write at that position. This is a bounded prefix cache, not a sliding or wrapping cache.
 *
 * [embeddingShards] must be the ordered contiguous rows of the source embedding, prepared
 * with [splitLfm2Embedding]. The original full embedding tensor never enters the traced graph.
 * This graph construction alone does not establish Vulkan numerical or performance results.
 */
class Lfm2CachedTensorNetwork(
    val dimensions: ModelDimensions,
    weights: Map<String, Tensor<FP32, Float>>,
    val capacity: Int,
    embeddingShards: List<Tensor<FP32, Float>>,
) {
    private val shards = embeddingShards.toList()
    private val finalNorm = weights.getValue("model.embedding_norm.weight")
    private val convolution = linkedMapOf<Int, BlockTensorOp>()
    private val attention = linkedMapOf<Int, AttentionTensorOp>()
    val embeddingShardRows: List<Int> get() = shards.map { it.shape[0] }

    init {
        val d = dimensions
        require(capacity in 1..d.contextLimit) { "Cache capacity exceeds the configured context bound" }
        require(d.layerTypes.any { it == "full_attention" }) { "Cached export requires an attention layer" }
        val expected = expectedModelWeights(d)
        require(weights.keys == expected.keys && expected.all { (name, shape) -> weights.getValue(name).shape == shape }) {
            "Cached full-model weight inventory/layout mismatch"
        }
        require(shards.isNotEmpty() && shards.all {
            it.shape.dimensions.size == 2 && it.shape[0] > 0 && it.shape[1] == d.width &&
                it.shape[0].toLong() * d.width * 4L <= MAX_EMBEDDING_SHARD_BYTES
        } && shards.sumOf { it.shape[0].toLong() } == d.vocabulary.toLong()) {
            "Embedding shards must cover the vocabulary in contiguous row order, each at most 64 MiB"
        }
        for ((layer, kind) in d.layerTypes.withIndex()) {
            fun w(name: String) = weights.getValue("model.layers.$layer.$name.weight")
            if (kind == "conv") {
                convolution[layer] = BlockTensorOp(d.width, d.kernel, d.intermediate, d.epsilon,
                    ShortConvTensorOp(d.width, d.kernel, w("conv.in_proj"), w("conv.conv"), w("conv.out_proj")),
                    w("operator_norm"), w("ffn_norm"), w("feed_forward.w1"), w("feed_forward.w2"), w("feed_forward.w3"),
                    RmsReductionPolicy.BALANCED_FP32)
            } else {
                attention[layer] = AttentionTensorOp(d.width, d.queryHeads, d.kvHeads, d.headDim, d.intermediate, d.epsilon,
                    w("operator_norm"), w("ffn_norm"), w("self_attn.q_proj"), w("self_attn.k_proj"), w("self_attn.v_proj"),
                    w("self_attn.out_proj"), w("self_attn.q_layernorm"), w("self_attn.k_layernorm"),
                    w("feed_forward.w1"), w("feed_forward.w2"), w("feed_forward.w3"),
                    RmsReductionPolicy.BALANCED_FP32, AttentionLayoutPolicy.DIRECT_KV_PER_HEAD)
            }
        }
    }

    /** State names and shapes are identical at the graph input and output. */
    fun stateShapes(): LinkedHashMap<String, Shape> = linkedMapOf<String, Shape>().apply {
        for ((layer, kind) in dimensions.layerTypes.withIndex()) {
            if (kind == "conv") put("history_$layer", Shape(dimensions.width, dimensions.kernel)) else {
                put("key_$layer", Shape(dimensions.kvHeads, capacity, dimensions.headDim))
                put("value_$layer", Shape(dimensions.kvHeads, capacity, dimensions.headDim))
            }
        }
    }

    /**
     * [mask] has zero at populated prefix slots and the appended current-token slot C,
     * and a large negative finite value at all unfilled slots. [writeMask] is one-hot
     * along its capacity axis. Runtime values are never read by this traceable function.
     */
    fun forward(
        tokenOneHot: Tensor<FP32, Float>,
        cos: Tensor<FP32, Float>,
        sin: Tensor<FP32, Float>,
        mask: Tensor<FP32, Float>,
        writeMask: Tensor<FP32, Float>,
        states: Map<String, Tensor<FP32, Float>>,
        ctx: ExecutionContext,
    ): Lfm2CachedTensorResult {
        val d = dimensions
        require(tokenOneHot.shape == Shape(1, d.vocabulary))
        require(cos.shape == Shape(1, d.headDim) && sin.shape == cos.shape)
        require(mask.shape == Shape(1, capacity + 1) && writeMask.shape == Shape(1, capacity, 1))
        val expected = stateShapes()
        require(states.keys == expected.keys && expected.all { (name, shape) -> states.getValue(name).shape == shape }) {
            "Every decoder layer needs its complete fixed-capacity runtime state"
        }
        val ops = ctx.ops
        var offset = 0
        val selected = shards.map { shard ->
            val oneHot = ops.narrow(tokenOneHot, dim = 1, start = offset, length = shard.shape[0])
            offset += shard.shape[0]
            ops.matmul(oneHot, shard)
        }
        var hidden = selected.reduce { total, part -> ops.add(total, part) }
        val keepMask = ops.rsubScalar(1f, writeMask)
        val nextStates = linkedMapOf<String, Tensor<FP32, Float>>()
        for ((layer, kind) in d.layerTypes.withIndex()) {
            hidden = if (kind == "conv") {
                val result = convolution.getValue(layer).forward(hidden, states.getValue("history_$layer"), ctx)
                nextStates["history_$layer"] = result.nextHistory
                result.output
            } else {
                val oldKey = states.getValue("key_$layer")
                val oldValue = states.getValue("value_$layer")
                val result = attention.getValue(layer).forward(hidden, oldKey, oldValue, cos, sin, mask, ctx)
                fun update(old: Tensor<FP32, Float>, appended: Tensor<FP32, Float>): Tensor<FP32, Float> {
                    val current = ops.narrow(appended, dim = 1, start = capacity, length = 1)
                    return ops.add(ops.multiply(old, keepMask), ops.multiply(current, writeMask))
                }
                nextStates["key_$layer"] = update(oldKey, result.nextKey)
                nextStates["value_$layer"] = update(oldValue, result.nextValue)
                result.output
            }
        }
        val normalized = rmsNormalize(hidden, finalNorm, ctx, RmsReductionPolicy.BALANCED_FP32, d.epsilon)
        val logits = shards.map { ops.matmulWeightTransposed(normalized, it) }
        return Lfm2CachedTensorResult(if (logits.size == 1) logits.single() else ops.concat(logits, dim = 1), nextStates)
    }
}
