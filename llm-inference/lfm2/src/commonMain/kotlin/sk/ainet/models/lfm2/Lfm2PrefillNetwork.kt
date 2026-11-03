package sk.ainet.models.lfm2

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

data class Lfm2PrefillResult(val logits: Tensor<FP32, Float>, val states: LinkedHashMap<String, Tensor<FP32, Float>>)

/** Complete prefill network: embedding lookup and position scheduling remain caller-owned. */
class Lfm2PrefillNetwork(val dimensions: ModelDimensions, weights: Map<String, Tensor<FP32, Float>>) {
    private val head = weights.getValue("model.embed_tokens.weight")
    private val finalNorm = weights.getValue("model.embedding_norm.weight")
    private val convolution = linkedMapOf<Int, BlockTensorOp>()
    private val attention = linkedMapOf<Int, AttentionTensorOp>()

    init {
        val d = dimensions
        val expected = expectedModelWeights(d)
        require(weights.keys == expected.keys && expected.all { (name, shape) -> weights.getValue(name).shape == shape }) { "Full-prefill weight inventory/layout mismatch" }
        for ((layer, kind) in d.layerTypes.withIndex()) {
            fun w(name: String) = weights.getValue("model.layers.$layer.$name.weight")
            if (kind == "conv") {
                convolution[layer] = BlockTensorOp(d.width, d.kernel, d.intermediate, d.epsilon,
                    ShortConvTensorOp(d.width, d.kernel, w("conv.in_proj"), w("conv.conv"), w("conv.out_proj")),
                    w("operator_norm"), w("ffn_norm"), w("feed_forward.w1"), w("feed_forward.w2"), w("feed_forward.w3"), RmsReductionPolicy.BALANCED_FP32)
            } else {
                attention[layer] = AttentionTensorOp(d.width, d.queryHeads, d.kvHeads, d.headDim, d.intermediate, d.epsilon,
                    w("operator_norm"), w("ffn_norm"), w("self_attn.q_proj"), w("self_attn.k_proj"), w("self_attn.v_proj"), w("self_attn.out_proj"),
                    w("self_attn.q_layernorm"), w("self_attn.k_layernorm"), w("feed_forward.w1"), w("feed_forward.w2"), w("feed_forward.w3"), RmsReductionPolicy.BALANCED_FP32)
            }
        }
    }

    fun forward(embeddings: Tensor<FP32, Float>, cos: Tensor<FP32, Float>, sin: Tensor<FP32, Float>, mask: Tensor<FP32, Float>,
        histories: Map<Int, Tensor<FP32, Float>>, ctx: ExecutionContext,
    ): Lfm2PrefillResult {
        val d = dimensions
        require(embeddings.shape.dimensions.size == 2 && embeddings.shape[0] > 0 && embeddings.shape[1] == d.width)
        val length = embeddings.shape[0]
        require(cos.shape == Shape(length, d.headDim) && sin.shape == cos.shape && mask.shape == Shape(length, length))
        require(histories.keys == convolution.keys && histories.values.all { it.shape == Shape(d.width, d.kernel) }) { "Each convolution layer needs its own runtime history" }
        var hidden = embeddings
        val states = linkedMapOf<String, Tensor<FP32, Float>>()
        for ((layer, kind) in d.layerTypes.withIndex()) {
            hidden = if (kind == "conv") {
                val result = convolution.getValue(layer).forward(hidden, histories.getValue(layer), ctx)
                states["history_$layer"] = result.nextHistory
                result.output
            } else {
                val result = attention.getValue(layer).forward(hidden, null, null, cos, sin, mask, ctx)
                states["key_$layer"] = result.nextKey
                states["value_$layer"] = result.nextValue
                result.output
            }
        }
        val normalized = rmsNormalize(hidden, finalNorm, ctx, RmsReductionPolicy.BALANCED_FP32, d.epsilon)
        return Lfm2PrefillResult(ctx.ops.matmulWeightTransposed(normalized, head), states)
    }
}

