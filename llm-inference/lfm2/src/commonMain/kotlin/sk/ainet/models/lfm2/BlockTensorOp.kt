package sk.ainet.models.lfm2

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

data class BlockTensorResult(
    val output: Tensor<FP32, Float>,
    val nextHistory: Tensor<FP32, Float>,
    val intermediates: Map<String, Tensor<FP32, Float>>,
)

/** The complete convolution variant of Lfm2DecoderLayer, with explicit chronological state. */
class BlockTensorOp(
    val width: Int,
    val kernel: Int,
    val intermediateSize: Int,
    val normEps: Float,
    private val convolution: ShortConvTensorOp,
    private val operatorNormWeight: Tensor<FP32, Float>,
    private val ffnNormWeight: Tensor<FP32, Float>,
    private val w1: Tensor<FP32, Float>,
    private val w2: Tensor<FP32, Float>,
    private val w3: Tensor<FP32, Float>,
    val rmsReduction: RmsReductionPolicy = RmsReductionPolicy.SERIAL_FP32,
) {
    init {
        require(width > 0 && kernel > 0 && intermediateSize > 0 && normEps.isFinite() && normEps > 0f)
        require(convolution.width == width && convolution.kernel == kernel) { "Convolution dimensions mismatch" }
        require(operatorNormWeight.shape == Shape(width) && ffnNormWeight.shape == Shape(width)) { "RMSNorm weight shape mismatch" }
        require(w1.shape == Shape(intermediateSize, width) && w3.shape == Shape(intermediateSize, width)) { "MLP input weight shape mismatch" }
        require(w2.shape == Shape(width, intermediateSize)) { "MLP output weight shape mismatch" }
    }

    private fun rmsNorm(input: Tensor<FP32, Float>, weight: Tensor<FP32, Float>, ctx: ExecutionContext): Tensor<FP32, Float> {
        return rmsNormalize(input, weight, ctx, rmsReduction, normEps)
    }

    /** Runtime input [T,D] and history [D,K]; no host reads or fixture values enter the graph. */
    fun forward(input: Tensor<FP32, Float>, history: Tensor<FP32, Float>, ctx: ExecutionContext): BlockTensorResult {
        require(input.shape.dimensions.size == 2 && input.shape[0] > 0 && input.shape[1] == width) { "Expected nonempty [T,$width] block input" }
        val ops = ctx.ops
        val operatorNormalized = rmsNorm(input, operatorNormWeight, ctx)
        val convolved = convolution.forward(operatorNormalized, history, ctx)
        val residual = ops.add(convolved.output, input)
        val ffnNormalized = rmsNorm(residual, ffnNormWeight, ctx)
        val mlp = ops.matmulWeightTransposed(
            ops.multiply(ops.silu(ops.matmulWeightTransposed(ffnNormalized, w1)), ops.matmulWeightTransposed(ffnNormalized, w3)),
            w2,
        )
        return BlockTensorResult(
            ops.add(residual, mlp), convolved.nextHistory,
            linkedMapOf("operator_norm" to operatorNormalized, "conv_output" to convolved.output,
                "operator_residual" to residual, "ffn_norm" to ffnNormalized, "mlp_output" to mlp),
        )
    }
}
