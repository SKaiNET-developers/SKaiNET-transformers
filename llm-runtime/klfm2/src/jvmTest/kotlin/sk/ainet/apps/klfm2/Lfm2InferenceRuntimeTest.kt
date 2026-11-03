package sk.ainet.apps.klfm2

import sk.ainet.models.lfm2.HybridLfm2Model
import sk.ainet.models.lfm2.HybridModelState
import sk.ainet.models.lfm2.ModelDimensions
import sk.ainet.models.lfm2.expectedModelWeights
import sk.ainet.apps.llm.InferenceRuntime
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class Lfm2InferenceRuntimeTest {
    private fun model(): HybridLfm2Model {
        val ctx = DirectCpuExecutionContext.create()
        val dimensions = ModelDimensions(
            8, 12, 4, 2, 2, 4, 3, 1e-5f, 1e6,
            listOf("conv", "full_attention"), contextLimit = 5,
        )
        val weights = expectedModelWeights(dimensions).mapValues { (name, shape) ->
            val size = shape.dimensions.reduce(Int::times)
            val values = if (name.endsWith("norm.weight")) {
                FloatArray(size) { 0.75f + (it % 3) * 0.1f }
            } else {
                FloatArray(size) { (((it * 7 + name.hashCode().mod(17)) % 19) - 9) * 0.025f }
            }
            ctx.fromFloatArray<FP32, Float>(shape, FP32::class, values)
        }
        return HybridLfm2Model(dimensions, ctx, weights)
    }

    private fun assertLogits(expected: FloatArray, actual: Tensor<FP32, Float>) {
        assertEquals(Shape(expected.size), actual.shape)
        val values = actual.data.copyToFloatArray()
        expected.indices.forEach { index ->
            assertTrue(values[index].isFinite(), "Non-finite logit at $index")
            assertTrue(abs(expected[index] - values[index]) <= 2e-6f,
                "Logit $index: expected ${expected[index]}, actual ${values[index]}")
        }
    }

    @Test fun interfaceConsumerReceivesOnlyLastPositionAndEquivalentTokenwiseLogits() {
        val model = model()
        val input = intArrayOf(1, 2, 3)
        val all = model.forward(input, HybridModelState()).logits.data.copyToFloatArray()
        val expected = all.copyOfRange(24, 36)
        Lfm2InferenceRuntime(model).use { batched ->
            // Calling only the published interface catches an accidental local-only API.
            val consumer: InferenceRuntime<FP32> = batched
            assertLogits(expected, consumer.forwardBatched(input))
            assertEquals(3, batched.position)
            Lfm2InferenceRuntime(model).use { tokenwise ->
                val other: InferenceRuntime<FP32> = tokenwise
                other.forward(1)
                other.forward(2)
                assertLogits(expected, other.forward(3))
                assertEquals(3, tokenwise.position)
            }
        }
    }

    @Test fun resetThroughInterfaceReplaysBothHybridCachesAndLeavesOtherRuntimeIndependent() {
        val model = model()
        Lfm2InferenceRuntime(model).use { first ->
            Lfm2InferenceRuntime(model).use { second ->
                val consumer: InferenceRuntime<FP32> = first
                val reference: InferenceRuntime<FP32> = second
                val expected = consumer.forwardBatched(intArrayOf(1, 2, 3)).data.copyToFloatArray()
                reference.forwardBatched(intArrayOf(1, 2))
                consumer.forward(4)
                consumer.reset()
                assertEquals(0, first.position)
                assertEquals(2, second.position)
                assertLogits(expected, consumer.forwardBatched(intArrayOf(1, 2, 3)))
                assertLogits(expected, reference.forward(3))
                assertEquals(3, first.position)
                assertEquals(3, second.position)
            }
        }
    }

    @Test fun invalidInputsThroughInterfacePreserveState() {
        val model = model()
        Lfm2InferenceRuntime(model).use { owner ->
            val consumer: InferenceRuntime<FP32> = owner
            consumer.forwardBatched(intArrayOf(1, 2))
            assertFailsWith<IllegalArgumentException> { consumer.forward(-1) }
            assertFailsWith<IllegalArgumentException> { consumer.forward(12) }
            assertFailsWith<IllegalArgumentException> { consumer.forwardBatched(intArrayOf()) }
            assertFailsWith<IllegalArgumentException> { consumer.forwardBatched(intArrayOf(1, 2, 3, 4)) }
            assertEquals(2, owner.position)
            val all = model.forward(intArrayOf(1, 2, 3), HybridModelState()).logits.data.copyToFloatArray()
            assertLogits(all.copyOfRange(24, 36), consumer.forward(3))
        }
    }

    @Test fun closeIsTerminalAndIdempotentWithoutDisposingSharedModel() {
        val model = model()
        val owner = Lfm2InferenceRuntime(model)
        val consumer: InferenceRuntime<FP32> = owner
        val expected = consumer.forwardBatched(intArrayOf(1, 2)).data.copyToFloatArray()
        owner.close()
        owner.close()
        assertEquals(0, owner.position)
        assertFailsWith<IllegalStateException> { consumer.forward(1) }
        assertFailsWith<IllegalStateException> { consumer.forwardBatched(intArrayOf(1, 2)) }
        assertFailsWith<IllegalStateException> { consumer.reset() }
        Lfm2InferenceRuntime(model).use { next ->
            assertLogits(expected, next.forwardBatched(intArrayOf(1, 2)))
        }
    }
}
