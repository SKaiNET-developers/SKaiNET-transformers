package sk.ainet.models.lfm2

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

data class ShortConvTensorResult(
    val output: Tensor<FP32, Float>,
    val nextHistory: Tensor<FP32, Float>,
)

/**
 * How the depthwise convolution is lowered. [DEPTHWISE] is one grouped `conv1d`. [MULTIPLY_SUM]
 * applies to a single token only: the convolution over a `[D, K]` window is then
 * `sum(window * weight, dim = 1)`, the same arithmetic as plain elementwise ops, for compilers
 * without grouped convolutions. Longer inputs always use [DEPTHWISE].
 */
enum class ShortConvLowering { DEPTHWISE, MULTIPLY_SUM }

/** Pure tensor/state form shared by eager evaluation and StableHLO export. */
class ShortConvTensorOp(
    val width: Int,
    val kernel: Int,
    private val inWeight: Tensor<FP32, Float>,
    private val convWeight: Tensor<FP32, Float>,
    private val outWeight: Tensor<FP32, Float>,
    val lowering: ShortConvLowering,
) {
    constructor(
        width: Int,
        kernel: Int,
        inWeight: Tensor<FP32, Float>,
        convWeight: Tensor<FP32, Float>,
        outWeight: Tensor<FP32, Float>,
    ) : this(width, kernel, inWeight, convWeight, outWeight, ShortConvLowering.DEPTHWISE)

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
        val timeMajor = if (lowering == ShortConvLowering.MULTIPLY_SUM && length == 1) {
            val taps = ops.multiply(window, ops.reshape(convWeight, Shape(width, kernel)))
            ops.reshape(ops.sum(taps, dim = 1), Shape(1, width))
        } else {
            val convolved = ops.conv1d(
                ops.reshape(window, Shape(1, width, kernel - 1 + length)),
                convWeight, bias = null, padding = 0, groups = width,
            )
            ops.transpose(ops.reshape(convolved, Shape(width, length)))
        }
        val output = ops.matmulWeightTransposed(ops.multiply(c, timeMajor), outWeight)
        // Keep this as a separate terminal operation: both results are graph outputs.
        val nextHistory = ops.narrow(combined, dim = 1, start = length, length = kernel)
        return ShortConvTensorResult(output, nextHistory)
    }
}
