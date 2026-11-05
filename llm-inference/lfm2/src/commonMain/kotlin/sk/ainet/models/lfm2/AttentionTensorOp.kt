package sk.ainet.models.lfm2

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import kotlin.math.sqrt

enum class AttentionLayoutPolicy { CURRENT_BATCHED, DIRECT_KV_PER_HEAD }

data class AttentionTensorResult(
    val output: Tensor<FP32, Float>,
    val nextKey: Tensor<FP32, Float>,
    val nextValue: Tensor<FP32, Float>,
    val intermediates: Map<String, Tensor<FP32, Float>>,
)

/** Complete attention variant of Lfm2DecoderLayer with chronological, un-repeated KV state. */
class AttentionTensorOp(
    val width: Int,
    val queryHeads: Int,
    val kvHeads: Int,
    val headDim: Int,
    val intermediateSize: Int,
    val normEps: Float,
    private val operatorNorm: Tensor<FP32, Float>,
    private val ffnNorm: Tensor<FP32, Float>,
    private val qWeight: Tensor<FP32, Float>,
    private val kWeight: Tensor<FP32, Float>,
    private val vWeight: Tensor<FP32, Float>,
    private val outWeight: Tensor<FP32, Float>,
    private val qNorm: Tensor<FP32, Float>,
    private val kNorm: Tensor<FP32, Float>,
    private val w1: Tensor<FP32, Float>,
    private val w2: Tensor<FP32, Float>,
    private val w3: Tensor<FP32, Float>,
    val rmsReduction: RmsReductionPolicy = RmsReductionPolicy.BALANCED_FP32,
    val layoutPolicy: AttentionLayoutPolicy = AttentionLayoutPolicy.CURRENT_BATCHED,
) {
    init {
        require(width > 0 && queryHeads > 0 && kvHeads > 0 && headDim > 0 && headDim % 2 == 0 && intermediateSize > 0)
        require(width == queryHeads * headDim && queryHeads % kvHeads == 0) { "Invalid attention head dimensions" }
        require(normEps.isFinite() && normEps > 0f)
        require(operatorNorm.shape == Shape(width) && ffnNorm.shape == Shape(width)) { "Block norm shape mismatch" }
        require(qWeight.shape == Shape(queryHeads * headDim, width) && outWeight.shape == Shape(width, queryHeads * headDim)) { "Query/output projection shape mismatch" }
        require(kWeight.shape == Shape(kvHeads * headDim, width) && vWeight.shape == Shape(kvHeads * headDim, width)) { "KV projection shape mismatch" }
        require(qNorm.shape == Shape(headDim) && kNorm.shape == Shape(headDim)) { "Head norm shape mismatch" }
        require(w1.shape == Shape(intermediateSize, width) && w3.shape == Shape(intermediateSize, width) && w2.shape == Shape(width, intermediateSize)) { "MLP shape mismatch" }
    }

    private fun rms(input: Tensor<FP32, Float>, weight: Tensor<FP32, Float>, ctx: ExecutionContext): Tensor<FP32, Float> {
        return rmsNormalize(input, weight, ctx, rmsReduction, normEps)
    }

    private fun rotate(input: Tensor<FP32, Float>, cos: Tensor<FP32, Float>, sin: Tensor<FP32, Float>, ctx: ExecutionContext): Tensor<FP32, Float> {
        val ops = ctx.ops
        val first = ops.narrow(input, dim = 2, start = 0, length = headDim / 2)
        val second = ops.narrow(input, dim = 2, start = headDim / 2, length = headDim / 2)
        val rotated = ops.concat(listOf(ops.mulScalar(second, -1f), first), dim = 2)
        return ops.add(ops.multiply(input, cos), ops.multiply(rotated, sin))
    }

    private fun repeatKv(input: Tensor<FP32, Float>, ctx: ExecutionContext): Tensor<FP32, Float> {
        if (queryHeads == kvHeads) return input
        val groups = queryHeads / kvHeads
        return ctx.ops.concat((0 until queryHeads).map { head ->
            ctx.ops.narrow(input, dim = 0, start = head / groups, length = 1)
        }, dim = 0)
    }

    /**
     * Input [T,D], optional past KV [KV,P,H], cos/sin [T,H], additive mask [T,P+T].
     * Positions and the causal mask are explicit runtime data, not graph constants.
     */
    fun forward(
        input: Tensor<FP32, Float>, pastKey: Tensor<FP32, Float>?, pastValue: Tensor<FP32, Float>?,
        cos: Tensor<FP32, Float>, sin: Tensor<FP32, Float>, mask: Tensor<FP32, Float>, ctx: ExecutionContext,
    ): AttentionTensorResult {
        require(input.shape.dimensions.size == 2 && input.shape[0] > 0 && input.shape[1] == width) { "Expected nonempty [T,$width] input" }
        require((pastKey == null) == (pastValue == null)) { "Both KV tensors must be supplied together" }
        val length = input.shape[0]
        val past = pastKey?.shape?.get(1) ?: 0
        if (pastKey != null && pastValue != null) {
            require(past > 0 && pastKey.shape == Shape(kvHeads, past, headDim) && pastValue.shape == pastKey.shape) { "Past KV shape mismatch" }
        }
        require(cos.shape == Shape(length, headDim) && sin.shape == cos.shape) { "RoPE runtime shape mismatch" }
        require(mask.shape == Shape(length, past + length)) { "Attention mask shape mismatch" }
        val ops = ctx.ops
        val normalized = rms(input, operatorNorm, ctx)
        val queryNormalized = ops.reshape(rms(ops.reshape(ops.matmulWeightTransposed(normalized, qWeight), Shape(length * queryHeads, headDim)), qNorm, ctx), Shape(length, queryHeads, headDim))
        val keyNormalized = ops.reshape(rms(ops.reshape(ops.matmulWeightTransposed(normalized, kWeight), Shape(length * kvHeads, headDim)), kNorm, ctx), Shape(length, kvHeads, headDim))
        val query = rotate(ops.permute(queryNormalized, intArrayOf(1, 0, 2)), cos, sin, ctx)
        val currentKey = rotate(ops.permute(keyNormalized, intArrayOf(1, 0, 2)), cos, sin, ctx)
        val currentValue = ops.permute(ops.reshape(ops.matmulWeightTransposed(normalized, vWeight), Shape(length, kvHeads, headDim)), intArrayOf(1, 0, 2))
        val key = if (pastKey == null) currentKey else ops.concat(listOf(pastKey, currentKey), dim = 1)
        val value = if (pastValue == null) currentValue else ops.concat(listOf(pastValue, currentValue), dim = 1)
        val sourceLength = past + length
        val groups = queryHeads / kvHeads
        fun head(tensor: Tensor<FP32, Float>, index: Int, rows: Int, columns: Int): Tensor<FP32, Float> =
            ops.reshape(ops.narrow(tensor, dim = 0, start = index, length = 1), Shape(rows, columns))
        val products = when (layoutPolicy) {
            AttentionLayoutPolicy.CURRENT_BATCHED -> ops.matmul(query, ops.permute(repeatKv(key, ctx), intArrayOf(0, 2, 1)))
            AttentionLayoutPolicy.DIRECT_KV_PER_HEAD -> ops.concat((0 until queryHeads).map { index ->
                val q = head(query, index, length, headDim)
                val k = head(key, index / groups, sourceLength, headDim)
                ops.reshape(ops.matmul(q, ops.permute(k, intArrayOf(1, 0))), Shape(1, length, sourceLength))
            }, dim = 0)
        }
        val scores = ops.add(ops.mulScalar(products, (1.0 / sqrt(headDim.toDouble())).toFloat()), mask)
        val probabilities = ops.softmax(scores, dim = 2)
        val context = when (layoutPolicy) {
            AttentionLayoutPolicy.CURRENT_BATCHED -> ops.matmul(probabilities, repeatKv(value, ctx))
            AttentionLayoutPolicy.DIRECT_KV_PER_HEAD -> ops.concat((0 until queryHeads).map { index ->
                val p = head(probabilities, index, length, sourceLength)
                val v = head(value, index / groups, sourceLength, headDim)
                ops.reshape(ops.matmul(p, v), Shape(1, length, headDim))
            }, dim = 0)
        }
        val attended = ops.reshape(ops.permute(context, intArrayOf(1, 0, 2)), Shape(length, width))
        val attention = ops.matmulWeightTransposed(attended, outWeight)
        val residual = ops.add(attention, input)
        val ffn = rms(residual, ffnNorm, ctx)
        val mlp = ops.matmulWeightTransposed(ops.multiply(ops.silu(ops.matmulWeightTransposed(ffn, w1)), ops.matmulWeightTransposed(ffn, w3)), w2)
        return AttentionTensorResult(ops.add(residual, mlp), key, value,
            linkedMapOf("operator_norm" to normalized, "q_norm" to queryNormalized, "k_norm" to keyNormalized,
                "attention_output" to attention, "operator_residual" to residual, "ffn_norm" to ffn, "mlp_output" to mlp))
    }
}
