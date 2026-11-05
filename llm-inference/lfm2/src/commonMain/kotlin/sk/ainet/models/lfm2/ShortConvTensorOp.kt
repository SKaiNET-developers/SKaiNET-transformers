package sk.ainet.models.lfm2

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

data class ShortConvTensorResult(
    val output: Tensor<FP32, Float>,
    val nextHistory: Tensor<FP32, Float>,
)

/** Pure tensor/state form shared by eager evaluation and StableHLO export. */
class ShortConvTensorOp(
    val width: Int,
    val kernel: Int,
    private val inWeight: Tensor<FP32, Float>,
    private val convWeight: Tensor<FP32, Float>,
    private val outWeight: Tensor<FP32, Float>,
) {
    init {
        require(width > 0 && kernel > 0) { "Dimensions must be positive" }
        require(inWeight.shape == Shape(3 * width, width)) { "in_proj shape mismatch" }
        require(convWeight.shape == Shape(width, 1, kernel)) { "conv shape mismatch" }
        require(outWeight.shape == Shape(width, width)) { "out_proj shape mismatch" }
    }

    /** Input [T,D], history [D,K], output [T,D], next history [D,K]. */
    fun forward(input: Tensor<FP32, Float>, history: Tensor<FP32, Float>, ctx: ExecutionContext): ShortConvTensorResult {
        require(input.shape.dimensions.size == 2 && input.shape[0] > 0 && input.shape[1] == width) { "Expected nonempty [T,$width] input" }
        require(history.shape == Shape(width, kernel)) { "Expected [$width,$kernel] history" }
        val length = input.shape[0]
        val ops = ctx.ops
        val projected = ops.matmulWeightTransposed(input, inWeight)
        val (b, c, values) = ops.split(projected, width, 1)
        val gated = ops.multiply(b, values)
        val combined = ops.concat(listOf(history, ops.transpose(gated)), dim = 1)
        // K history samples are retained; only the most recent K-1 affect a new token.
        val window = ops.narrow(combined, dim = 1, start = 1, length = kernel - 1 + length)
        val convolved = ops.conv1d(
            ops.reshape(window, Shape(1, width, kernel - 1 + length)),
            convWeight, bias = null, padding = 0, groups = width,
        )
        val timeMajor = ops.transpose(ops.reshape(convolved, Shape(width, length)))
        val output = ops.matmulWeightTransposed(ops.multiply(c, timeMajor), outWeight)
        // Keep this as a separate terminal operation: both results are graph outputs.
        val nextHistory = ops.narrow(combined, dim = 1, start = length, length = kernel)
        return ShortConvTensorResult(output, nextHistory)
    }
}
