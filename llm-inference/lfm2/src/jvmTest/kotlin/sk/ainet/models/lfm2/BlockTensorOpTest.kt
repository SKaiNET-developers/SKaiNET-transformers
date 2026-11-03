package sk.ainet.models.lfm2

import sk.ainet.models.lfm2.export.*

import sk.ainet.compile.hlo.ConstantMaterializationPolicy
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import kotlin.math.exp
import kotlin.math.sqrt
import kotlin.test.*

class BlockTensorOpTest {
    private val ctx = DirectCpuExecutionContext.create()
    private fun tensor(shape: Shape, vararg values: Float) = ctx.fromFloatArray<FP32, Float>(shape, FP32::class, values)
    private fun operation(): BlockTensorOp {
        val inWeight = tensor(Shape(6, 2), 1f, 0f, 0f, 1f, 1f, 0f, 0f, 1f, 1f, 0f, 0f, 1f)
        val conv = ShortConvTensorOp(2, 1, inWeight, tensor(Shape(2, 1, 1), 0.5f, -0.25f), tensor(Shape(2, 2), 1f, 0f, 0f, 1f))
        return BlockTensorOp(2, 1, 2, 1e-5f, conv, tensor(Shape(2), 2f, 0.5f), tensor(Shape(2), 0.75f, 1.25f),
            tensor(Shape(2, 2), 0.2f, 0f, 0f, -0.3f), tensor(Shape(2, 2), 0.7f, 0f, 0f, 0.9f),
            tensor(Shape(2, 2), 0.4f, 0f, 0f, 0.6f))
    }

    @Test
    fun completeBlockMatchesClosedFormIncludingNormWeightsSiluResidualsAndState() {
        val result = operation().forward(tensor(Shape(1, 2), 3f, 4f), tensor(Shape(2, 1), 0f, 0f), ctx)
        val norm = sqrt(12.5 + 1e-5)
        val normalized = doubleArrayOf(6.0 / norm, 2.0 / norm)
        val conv = doubleArrayOf(0.5 * normalized[0] * normalized[0] * normalized[0], -0.25 * normalized[1] * normalized[1] * normalized[1])
        val residual = doubleArrayOf(3.0 + conv[0], 4.0 + conv[1])
        val ffnDenominator = sqrt((residual[0] * residual[0] + residual[1] * residual[1]) / 2 + 1e-5)
        val ffn = doubleArrayOf(0.75 * residual[0] / ffnDenominator, 1.25 * residual[1] / ffnDenominator)
        val gate = doubleArrayOf(0.2 * ffn[0], -0.3 * ffn[1])
        val mlp = doubleArrayOf(0.7 * gate[0] / (1 + exp(-gate[0])) * 0.4 * ffn[0], 0.9 * gate[1] / (1 + exp(-gate[1])) * 0.6 * ffn[1])
        val actual = result.output.data.copyToFloatArray()
        for (i in 0..1) {
            assertEquals((residual[i] + mlp[i]).toFloat(), actual[i], 2e-6f)
            assertEquals((normalized[i] * normalized[i]).toFloat(), result.nextHistory.data.copyToFloatArray()[i], 2e-6f)
            assertEquals(normalized[i].toFloat(), result.intermediates.getValue("operator_norm").data.copyToFloatArray()[i], 2e-6f)
            assertEquals(mlp[i].toFloat(), result.intermediates.getValue("mlp_output").data.copyToFloatArray()[i], 2e-6f)
        }
    }

    @Test
    fun exportedBlockHasExplicitRuntimeStateAndEightUnmodifiedParameterSources() {
        val traced = traceLfm2ConvBlock(operation(), 3, ConstantMaterializationPolicy.ExternalAlways(scope = "model"))
        assertEquals("decoder_block", traced.module.functionName)
        assertEquals(2, traced.module.inputSpecs.size)
        assertEquals(2, traced.module.outputSpecs.size)
        assertEquals(8, traced.module.externalParameters.size)
        assertTrue(traced.module.content.contains("stablehlo.convolution"))
        assertTrue(traced.module.content.contains("stablehlo.reduce"))
        assertFalse(traced.module.content.contains("custom_call"))
        assertTrue(traced.operationCount > 20)
    }

    @Test
    fun invalidNormEpsilonOrMlpLayoutAreRejectedBeforeExecution() {
        val conv = ShortConvTensorOp(2, 1, tensor(Shape(6, 2), *FloatArray(12)), tensor(Shape(2, 1, 1), 1f, 1f), tensor(Shape(2, 2), 1f, 0f, 0f, 1f))
        fun construct(eps: Float, outputShape: Shape) = BlockTensorOp(2, 1, 2, eps, conv, tensor(Shape(2), 1f, 1f), tensor(Shape(2), 1f, 1f),
            tensor(Shape(2, 2), *FloatArray(4)), tensor(outputShape, *FloatArray(outputShape.dimensions.reduce(Int::times))), tensor(Shape(2, 2), *FloatArray(4)))
        assertFailsWith<IllegalArgumentException> { construct(Float.NaN, Shape(2, 2)) }
        assertFailsWith<IllegalArgumentException> { construct(0f, Shape(2, 2)) }
        assertFailsWith<IllegalArgumentException> { construct(1e-5f, Shape(1, 4)) }
    }
}
