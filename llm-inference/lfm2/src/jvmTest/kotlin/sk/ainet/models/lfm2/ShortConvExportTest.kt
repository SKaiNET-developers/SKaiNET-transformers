package sk.ainet.models.lfm2

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import sk.ainet.models.lfm2.export.*
import kotlin.test.*

class ShortConvExportTest {
    private val ctx = DirectCpuExecutionContext.create()
    private fun operation(): ShortConvTensorOp {
        val f = readFixture(javaClass.getResource("/short-conv.json")!!.readText())
        val d = f.config.hiddenSize; val k = f.config.kernelSize
        fun tensor(shape: Shape, values: List<List<Float>>) = ctx.fromFloatArray<FP32, Float>(shape, FP32::class, values.flatten().toFloatArray())
        return ShortConvTensorOp(d, k, tensor(Shape(3 * d, d), f.weights.inProjection),
            tensor(Shape(d, 1, k), f.weights.conv), tensor(Shape(d, d), f.weights.outProjection))
    }

    @Test fun tracingUsesUnreadableRuntimeArgumentsAndReturnsExplicitStateAtSeveralLengths() {
        for (length in listOf(1, 2, 3, 4, 7)) {
            val trace = traceLfm2ShortConv(operation(), length)
            assertEquals(listOf("input", "history"), trace.inputNames)
            assertEquals(listOf("output", "next_history"), trace.outputNames)
            assertEquals(Shape(length, 4), trace.inputShapes.first())
            assertEquals(Shape(4, 3), trace.outputShapes.last())
            assertEquals(3, trace.module.externalParameters.size)
            assertTrue(trace.module.content.contains("stablehlo.convolution"))
            assertFalse(trace.module.content.contains("custom_call"))
        }
    }

    @Test fun genericTracerRejectsHostReadOfAnyRuntimeInput() {
        val failure = assertFailsWith<IllegalStateException> {
            traceLfm2Function("invalid", linkedMapOf("x" to Shape(1, 2))) { input, _ ->
                input.getValue("x").data[0, 0]
                linkedMapOf("x" to input.getValue("x"))
            }
        }
        assertTrue(failure.message!!.contains("cannot be read"))
    }
}
