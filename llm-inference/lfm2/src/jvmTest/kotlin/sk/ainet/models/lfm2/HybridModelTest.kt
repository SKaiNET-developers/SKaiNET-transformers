package sk.ainet.models.lfm2

import sk.ainet.models.lfm2.export.*

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.test.*

class HybridModelTest {
    private val ctx = DirectCpuExecutionContext.create()
    private val dimensions = ModelDimensions(8, 12, 4, 2, 2, 4, 3, 1e-5f, 1e6, listOf("conv", "full_attention"), contextLimit = 5)

    private fun model(): HybridLfm2Model {
        val weights = HybridLfm2Model.expectedWeights(dimensions).mapValues { (name, shape) ->
            val size = shape.dimensions.reduce(Int::times)
            val values = if (name.endsWith("norm.weight")) FloatArray(size) { 0.75f + (it % 3) * 0.1f }
                else FloatArray(size) { index -> (((index * 7 + name.hashCode().mod(17)) % 19) - 9) * 0.025f }
            ctx.fromFloatArray<FP32, Float>(shape, FP32::class, values)
        }
        return HybridLfm2Model(dimensions, ctx, weights)
    }

    private fun close(first: FloatArray, second: FloatArray, tolerance: Float = 2e-6f) {
        assertEquals(first.size, second.size)
        for (i in first.indices) assertTrue(abs(first[i] - second[i]) <= tolerance, "Value $i differs: ${first[i]} versus ${second[i]}")
    }

    @Test
    fun mixedConvolutionAttentionStreamingAndResetReproduceFreshFullPrefix() {
        val model = model()
        val all = HybridModelState()
        val full = model.forward(intArrayOf(1, 2, 3), all)
        val streaming = HybridModelState()
        val first = model.forward(intArrayOf(1), streaming)
        val rest = model.forward(intArrayOf(2, 3), streaming)
        val joined = first.logits.data.copyToFloatArray() + rest.logits.data.copyToFloatArray()
        close(full.logits.data.copyToFloatArray(), joined)
        assertEquals(3, streaming.position)
        assertEquals(setOf(0), streaming.convolution.keys)
        assertEquals(setOf(1), streaming.keys.keys)
        close(all.convolution.getValue(0).data.copyToFloatArray(), streaming.convolution.getValue(0).data.copyToFloatArray())
        close(all.keys.getValue(1).data.copyToFloatArray(), streaming.keys.getValue(1).data.copyToFloatArray())
        close(all.values.getValue(1).data.copyToFloatArray(), streaming.values.getValue(1).data.copyToFloatArray())
        streaming.reset()
        assertEquals(0, streaming.position)
        assertTrue(streaming.convolution.isEmpty() && streaming.keys.isEmpty() && streaming.values.isEmpty())
        close(full.logits.data.copyToFloatArray(), model.forward(intArrayOf(1, 2, 3), streaming).logits.data.copyToFloatArray())
    }

    @Test
    fun interleavedRequestsDoNotShareHybridCachesAndInvalidIdsDoNotAdvanceState() {
        val model = model()
        val a = HybridModelState(); val b = HybridModelState()
        model.forward(intArrayOf(1), a)
        model.forward(intArrayOf(8, 7), b)
        val continued = model.forward(intArrayOf(2, 3), a)
        val fresh = model.forward(intArrayOf(1, 2, 3), HybridModelState())
        close(fresh.logits.data.copyToFloatArray().copyOfRange(12, 36), continued.logits.data.copyToFloatArray())
        val previous = a.keys.getValue(1).data.copyToFloatArray()
        assertFailsWith<IllegalArgumentException> { model.forward(intArrayOf(12), a) }
        assertFailsWith<IllegalArgumentException> { model.forward(intArrayOf(-1), a) }
        assertFailsWith<IllegalArgumentException> { model.forward(intArrayOf(1, 2, 3), a) }
        assertEquals(3, a.position)
        assertContentEquals(previous, a.keys.getValue(1).data.copyToFloatArray())
        assertEquals(2, b.position)
    }

    @Test
    fun pinnedModelMappingContainsExactly148WeightsAndOneTiedEmbedding() {
        val kinds = listOf("conv", "conv", "full_attention", "conv", "conv", "full_attention", "conv", "conv", "full_attention", "conv", "full_attention", "conv", "full_attention", "conv", "full_attention", "conv")
        val mapping = HybridLfm2Model.expectedWeights(ModelDimensions(1024, 65536, 16, 8, 64, 4608, 3, 1e-5f, 1e6, kinds))
        assertEquals(148, mapping.size)
        assertEquals(Shape(65536, 1024), mapping.getValue("model.embed_tokens.weight"))
        assertEquals(Shape(1024), mapping.getValue("model.embedding_norm.weight"))
        assertEquals(Shape(64), mapping.getValue("model.layers.14.self_attn.q_layernorm.weight"))
        assertEquals(Shape(1024, 1, 3), mapping.getValue("model.layers.15.conv.conv.weight"))
        assertFalse(mapping.keys.any { it.startsWith("lm_head") })
        val model = model()
        assertEquals(Shape(1, 12), model.forward(intArrayOf(1), HybridModelState()).logits.shape)
    }

    @Test
    fun greedyStreamingMatchesFreshPrefixArgmaxAndHonorsStopAndLength() {
        val model = model(); val prompt = intArrayOf(1, 2)
        val streamed = model.generateGreedy(prompt, 3, emptySet())
        var prefix = prompt
        val expected = mutableListOf<Int>()
        repeat(3) {
            val logits = model.forward(prefix, HybridModelState()).logits.data.copyToFloatArray().takeLast(12)
            val next = logits.indices.maxBy { logits[it] }
            expected.add(next); prefix += next
        }
        assertContentEquals(expected.toIntArray(), streamed.generatedIds)
        assertEquals("length", streamed.stopReason)
        assertEquals(4, streamed.stateInputTokens)
        val stopped = model.generateGreedy(prompt, 3, setOf(expected.first()))
        assertContentEquals(intArrayOf(expected.first()), stopped.generatedIds)
        assertEquals("eos", stopped.stopReason)
        assertEquals(2, stopped.stateInputTokens)
        assertFailsWith<IllegalArgumentException> { model.generateGreedy(prompt, 0, emptySet()) }
        assertFailsWith<IllegalArgumentException> { model.generateGreedy(prompt, 5, emptySet()) }
    }
}
