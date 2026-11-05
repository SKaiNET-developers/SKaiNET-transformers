package sk.ainet.models.lfm2

import kotlinx.serialization.json.*
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.storage.DefaultBufferResolver
import sk.ainet.lang.types.FP32
import sk.ainet.models.lfm2.export.lfm2CachedStepContract
import sk.ainet.models.lfm2.export.traceLfm2CachedStep
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import kotlin.test.*

class Lfm2CachedTensorNetworkTest {
    private val ctx = DirectCpuExecutionContext.create()
    private val dimensions = ModelDimensions(8, 12, 4, 2, 2, 4, 3, 1e-5f, 1e6,
        listOf("conv", "full_attention", "conv", "full_attention"), contextLimit = 6)
    private val weights = HybridLfm2Model.expectedWeights(dimensions).mapValues { (name, shape) ->
        val values = FloatArray(shape.dimensions.reduce(Int::times)) { i ->
            if (name.endsWith("norm.weight")) 0.75f + i % 3 * 0.1f
            else (((i * 7 + name.hashCode().mod(17)) % 19) - 9) * 0.025f
        }
        tensor(shape, values)
    }
    private val shards = splitLfm2Embedding(weights.getValue("model.embed_tokens.weight"), ctx, rowsPerShard = 3)
    private val network = Lfm2CachedTensorNetwork(dimensions, weights, 6, shards)

    private fun tensor(shape: Shape, values: FloatArray = FloatArray(shape.dimensions.reduce(Int::times))): Tensor<FP32, Float> =
        ctx.fromFloatArray(shape, FP32::class, values)

    private fun initial(): LinkedHashMap<String, Tensor<FP32, Float>> =
        network.stateShapes().mapValuesTo(linkedMapOf()) { (_, shape) -> tensor(shape) }

    private fun step(token: Int, position: Int, states: Map<String, Tensor<FP32, Float>>): Lfm2CachedTensorResult {
        require(token in 0 until dimensions.vocabulary && position in 0 until network.capacity)
        val d = dimensions
        val cosines = FloatArray(d.headDim)
        val sines = FloatArray(d.headDim)
        for (i in 0 until d.headDim / 2) {
            val angle = position * d.theta.pow(-2.0 * i / d.headDim)
            for (lane in intArrayOf(i, i + d.headDim / 2)) {
                cosines[lane] = cos(angle).toFloat(); sines[lane] = sin(angle).toFloat()
            }
        }
        return network.forward(tensor(Shape(1, d.vocabulary), FloatArray(d.vocabulary) { if (it == token) 1f else 0f }),
            tensor(Shape(1, d.headDim), cosines), tensor(Shape(1, d.headDim), sines),
            tensor(Shape(1, network.capacity + 1), FloatArray(network.capacity + 1) {
                if (it < position || it == network.capacity) 0f else -Float.MAX_VALUE
            }), tensor(Shape(1, network.capacity, 1), FloatArray(network.capacity) { if (it == position) 1f else 0f }), states, ctx)
    }

    private fun closeEnough(expected: FloatArray, actual: FloatArray) {
        assertEquals(expected.size, actual.size)
        expected.indices.forEach { i ->
            assertTrue(expected[i].isFinite() && actual[i].isFinite() && abs(expected[i] - actual[i]) <= 2e-6f,
                "Value $i differs: ${expected[i]} versus ${actual[i]}")
        }
    }

    @Test
    fun sequentialPromptAndDecodePreserveFullNetworkLogitsAndEveryLayerState() {
        val model = HybridLfm2Model(dimensions, ctx, weights)
        val referenceState = HybridModelState()
        var state: Map<String, Tensor<FP32, Float>> = initial()
        // Touch every embedding shard and finish at the last legal cache slot.
        for ((position, token) in intArrayOf(0, 3, 6, 11, 4, 9).withIndex()) {
            val expected = model.forward(intArrayOf(token), referenceState)
            val actual = step(token, position, state)
            closeEnough(expected.logits.data.copyToFloatArray(), actual.logits.data.copyToFloatArray())
            assertEquals(network.stateShapes().keys.toList(), actual.states.keys.toList())
            for ((layer, kind) in dimensions.layerTypes.withIndex()) {
                if (kind == "conv") {
                    closeEnough(referenceState.convolution.getValue(layer).data.copyToFloatArray(), actual.states.getValue("history_$layer").data.copyToFloatArray())
                } else {
                    for ((name, reference) in listOf("key" to referenceState.keys, "value" to referenceState.values)) {
                        val cache = actual.states.getValue("${name}_$layer")
                        assertEquals(Shape(dimensions.kvHeads, network.capacity, dimensions.headDim), cache.shape)
                        val prefix = ctx.ops.narrow(cache, dim = 1, start = 0, length = position + 1)
                        closeEnough(reference.getValue(layer).data.copyToFloatArray(), prefix.data.copyToFloatArray())
                    }
                }
            }
            state = actual.states
        }
    }

    @Test
    fun greedyDecodeMatchesEagerAndFreshStateReplaysTheRequest() {
        val prompt = intArrayOf(1, 8)
        val expected = HybridLfm2Model(dimensions, ctx, weights).generateGreedy(prompt, 4, emptySet())
        fun generate(): List<Int> {
            var state: Map<String, Tensor<FP32, Float>> = initial()
            var position = 0
            var result: Lfm2CachedTensorResult? = null
            for (token in prompt) {
                result = step(token, position++, state); state = result.states
            }
            val ids = mutableListOf<Int>()
            repeat(4) { index ->
                val logits = result!!.logits.data.copyToFloatArray()
                val token = logits.indices.maxBy { logits[it] }
                ids.add(token)
                if (index < 3) { result = step(token, position++, state); state = result!!.states }
            }
            assertEquals(expected.stateInputTokens, position)
            return ids
        }
        assertEquals(expected.generatedIds.toList(), generate())
        assertEquals(expected.generatedIds.toList(), generate())
    }

    @Test
    fun maskedUnusedCacheDoesNotAffectLogitsAndOnlySelectedSlotIsWritten() {
        val zero = step(2, 0, initial())
        val poison = initial().mapValues { (name, old) ->
            if (name.startsWith("history_")) old else tensor(old.shape, FloatArray(old.shape.dimensions.reduce(Int::times)) { 17f + it % 5 })
        }
        val actual = step(2, 0, poison)
        closeEnough(zero.logits.data.copyToFloatArray(), actual.logits.data.copyToFloatArray())
        for (name in poison.keys.filterNot { it.startsWith("history_") }) {
            closeEnough(ctx.ops.narrow(zero.states.getValue(name), 1, 0, 1).data.copyToFloatArray(),
                ctx.ops.narrow(actual.states.getValue(name), 1, 0, 1).data.copyToFloatArray())
            assertContentEquals(ctx.ops.narrow(poison.getValue(name), 1, 1, 5).data.copyToFloatArray(),
                ctx.ops.narrow(actual.states.getValue(name), 1, 1, 5).data.copyToFloatArray())
        }
    }

    @Test
    fun shardingPreservesEveryEmbeddingRowIncludingAnUnevenFinalShard() {
        val uneven = splitLfm2Embedding(weights.getValue("model.embed_tokens.weight"), ctx, rowsPerShard = 5)
        assertEquals(listOf(5, 5, 2), uneven.map { it.shape[0] })
        assertContentEquals(weights.getValue("model.embed_tokens.weight").data.copyToFloatArray(),
            uneven.flatMap { it.data.copyToFloatArray().toList() }.toFloatArray())
        assertFailsWith<IllegalArgumentException> { splitLfm2Embedding(weights.getValue("model.embed_tokens.weight"), ctx, 0) }
        assertFailsWith<IllegalArgumentException> { Lfm2CachedTensorNetwork(dimensions, weights, 7, shards) }
        assertFailsWith<IllegalArgumentException> { Lfm2CachedTensorNetwork(dimensions, weights, 1, shards.dropLast(1)) }
    }

    @Test
    fun incompleteOrWronglyShapedCacheIsRejectedBeforeGraphExecution() {
        assertFailsWith<IllegalArgumentException> { step(1, 0, initial() - "key_1") }
        assertFailsWith<IllegalArgumentException> { step(1, 0, initial() + ("key_1" to tensor(Shape(2, 5, 2)))) }
        assertFailsWith<IllegalArgumentException> { step(1, network.capacity, initial()) }
    }

    @Test
    fun exportedGraphUsesFixedStateAndOneExternalCopyOfEachHeadShard() {
        val trace = traceLfm2CachedStep(network)
        val names = listOf("history_0", "key_1", "value_1", "history_2", "key_3", "value_3")
        assertEquals(listOf("token_one_hot", "cos", "sin", "mask", "write_mask") + names, trace.inputNames)
        assertEquals(listOf("logits") + names, trace.outputNames)
        assertEquals(trace.inputShapes.drop(5), trace.outputShapes.drop(1))
        assertEquals(weights.size + 3, trace.module.externalParameters.size)
        assertTrue(trace.module.content.contains("stablehlo.convolution"))
        assertTrue(trace.module.content.contains("across dimensions = [2]"))
        assertFalse(trace.module.content.contains("custom_call"))
        assertFalse(trace.module.content.contains("stablehlo.gather"))
        val resolver = DefaultBufferResolver()
        val embeddingBytes = dimensions.vocabulary.toLong() * dimensions.width * 4
        assertTrue(trace.module.externalParameters.none { resolver.resolve(it.source).use { source -> source.sizeInBytes == embeddingBytes } })
        val contract = lfm2CachedStepContract(network, trace)
        assertEquals(6, contract.getValue("capacity").jsonPrimitive.int)
        assertEquals("lfm2_cached_step", contract.getValue("function_name").jsonPrimitive.content)
        assertEquals(listOf(3, 3, 3, 3), contract.getValue("embedding_shard_rows").jsonArray.map { it.jsonPrimitive.int })
        assertEquals(trace.inputNames, contract.getValue("inputs").jsonArray.map { it.jsonObject.getValue("name").jsonPrimitive.content })
    }
}
