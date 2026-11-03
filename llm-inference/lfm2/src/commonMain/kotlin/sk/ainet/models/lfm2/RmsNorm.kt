package sk.ainet.models.lfm2

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

enum class RmsReductionPolicy { SERIAL_FP32, BALANCED_FP32 }

/** FP32 throughout. Two shorter reductions limit rounding drift in wide RMS sums. */
fun rmsNormalize(input: Tensor<FP32, Float>, weight: Tensor<FP32, Float>, ctx: ExecutionContext,
    policy: RmsReductionPolicy = RmsReductionPolicy.SERIAL_FP32,
    epsilon: Float = 1e-5f,
): Tensor<FP32, Float> {
    val ops = ctx.ops
    val rows = input.shape[0]
    val width = input.shape[1]
    require(input.shape.dimensions.size == 2 && weight.shape == Shape(width))
    val squared = ops.multiply(input, input)
    var factor = 1
    if (policy == RmsReductionPolicy.BALANCED_FP32) {
        while (factor * 2 <= width / (factor * 2) && width % (factor * 2) == 0) factor *= 2
    }
    val mean = if (factor == 1) ops.mean(squared, dim = 1) else {
        val grouped = ops.reshape(squared, Shape(rows, width / factor, factor))
        ops.mean(ops.mean(grouped, dim = 2), dim = 1)
    }
    val inverse = ops.rdivScalar(1f, ops.sqrt(ops.addScalar(mean, epsilon)))
    return ops.multiply(weight, ops.multiply(input, ops.reshape(inverse, Shape(rows, 1))))
}
