package sk.ainet.models.lfm2

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.test.*

class Lfm2SessionTest {
    private fun model(): HybridLfm2Model {
        val ctx = DirectCpuExecutionContext.create()
        val dimensions = ModelDimensions(8, 12, 4, 2, 2, 4, 3, 1e-5f, 1e6,
            listOf("conv", "full_attention"), contextLimit = 5)
        val weights = expectedModelWeights(dimensions).mapValues { (name, shape) ->
            val size = shape.dimensions.reduce(Int::times)
            val values = if (name.endsWith("norm.weight")) FloatArray(size) { 0.75f + (it % 3) * 0.1f }
                else FloatArray(size) { (((it * 7 + name.hashCode().mod(17)) % 19) - 9) * 0.025f }
            ctx.fromFloatArray<FP32, Float>(shape, FP32::class, values)
        }
        return HybridLfm2Model(dimensions, ctx, weights)
    }

    private fun closeEnough(expected: FloatArray, actual: FloatArray) {
        assertEquals(expected.size, actual.size)
        expected.indices.forEach { assertTrue(abs(expected[it] - actual[it]) <= 2e-6f) }
    }

    @Test fun batchedLastLogitsMatchTokenwiseAndIndependentSessions() {
        val model = model()
        assertEquals(RmsReductionPolicy.BALANCED_FP32, model.rmsReduction)
        Lfm2Session(model).use { a ->
            Lfm2Session(model).use { b ->
                val batched = a.forwardBatched(intArrayOf(1, 2, 3))
                assertEquals(Shape(12), batched.shape)
                b.forward(1); b.forward(2)
                closeEnough(batched.data.copyToFloatArray(), b.forward(3).data.copyToFloatArray())
                a.forward(4)
                assertEquals(4, a.position)
                assertEquals(3, b.position)
                b.reset()
                assertEquals(0, b.position)
                closeEnough(batched.data.copyToFloatArray(), b.forwardBatched(intArrayOf(1, 2, 3)).data.copyToFloatArray())
            }
        }
    }

    @Test fun rejectedRequestsPreserveStateAndClosedSessionsCannotRestart() {
        val model = model()
        val session = Lfm2Session(model)
        session.forwardBatched(intArrayOf(1, 2))
        assertFailsWith<IllegalArgumentException> { session.forward(12) }
        assertFailsWith<IllegalArgumentException> { session.forwardBatched(intArrayOf()) }
        assertFailsWith<IllegalArgumentException> { session.forwardBatched(intArrayOf(1, 2, 3, 4)) }
        assertEquals(2, session.position)
        Lfm2Session(model).use { fresh ->
            closeEnough(fresh.forwardBatched(intArrayOf(1, 2, 3)).data.copyToFloatArray(), session.forward(3).data.copyToFloatArray())
        }
        session.close(); session.close()
        assertEquals(0, session.position)
        assertFailsWith<IllegalStateException> { session.forward(1) }
        assertFailsWith<IllegalStateException> { session.reset() }
    }
}
