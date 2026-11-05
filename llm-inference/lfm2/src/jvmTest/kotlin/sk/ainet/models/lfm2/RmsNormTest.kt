package sk.ainet.models.lfm2

import sk.ainet.models.lfm2.export.*

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.test.*

class RmsNormTest {
    @Test
    fun balancedFp32ReductionRetainsSmallContributionsInWideRmsAgainstDoubleOracle() {
        val ctx = DirectCpuExecutionContext.create()
        val values = FloatArray(1024) { if (it == 0) 10000f else 1f }
        val input = ctx.fromFloatArray<FP32, Float>(Shape(1, 1024), FP32::class, values)
        val weight = ctx.fromFloatArray<FP32, Float>(Shape(1024), FP32::class, FloatArray(1024) { 1f })
        val denominator = sqrt(values.sumOf { it.toDouble() * it } / values.size + 1e-5)
        val expected = (values[0] / denominator).toFloat()
        val serial = rmsNormalize(input, weight, ctx, RmsReductionPolicy.SERIAL_FP32).data.copyToFloatArray()[0]
        val balanced = rmsNormalize(input, weight, ctx, RmsReductionPolicy.BALANCED_FP32).data.copyToFloatArray()[0]
        assertTrue(abs(serial - expected) > 1e-4f, "The input must expose serial FP32 accumulation loss")
        assertTrue(abs(balanced - expected) < 1e-5f)
        assertTrue(abs(balanced - expected) < abs(serial - expected) / 8f)
    }
}
