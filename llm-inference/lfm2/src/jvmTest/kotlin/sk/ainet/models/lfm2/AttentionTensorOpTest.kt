package sk.ainet.models.lfm2

import sk.ainet.models.lfm2.export.*

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.test.*

class AttentionTensorOpTest {
    private val ctx = DirectCpuExecutionContext.create()
    private fun tensor(shape: Shape, values: FloatArray) = ctx.fromFloatArray<FP32, Float>(shape, FP32::class, values)
    private fun identity(size: Int) = FloatArray(size * size) { i -> if (i / size == i % size) 1f else 0f }
    private val norm = floatArrayOf(1.2f, 0.8f, 1.1f, 0.9f, 1.2f, 0.8f, 1.1f, 0.9f)
    private fun operation(layoutPolicy: AttentionLayoutPolicy = AttentionLayoutPolicy.CURRENT_BATCHED): AttentionTensorOp {
        val k = FloatArray(32) { i -> if (i / 8 == i % 8) 1f else 0f }
        val v = FloatArray(32) { i -> if (i / 8 + 4 == i % 8) 1f else 0f }
        return AttentionTensorOp(8, 4, 2, 2, 4, 1e-5f,
            tensor(Shape(8), norm), tensor(Shape(8), FloatArray(8) { 1f }),
            tensor(Shape(8, 8), identity(8)), tensor(Shape(4, 8), k), tensor(Shape(4, 8), v), tensor(Shape(8, 8), identity(8)),
            tensor(Shape(2), floatArrayOf(1.25f, 0.75f)), tensor(Shape(2), floatArrayOf(0.9f, 1.1f)),
            tensor(Shape(4, 8), FloatArray(32)), tensor(Shape(8, 4), FloatArray(32)), tensor(Shape(4, 8), FloatArray(32)), layoutPolicy = layoutPolicy)
    }
    private fun positions(start: Int, length: Int): Pair<Tensor<FP32, Float>, Tensor<FP32, Float>> {
        val c = FloatArray(length * 2) { cos((start + it / 2).toDouble()).toFloat() }
        val s = FloatArray(length * 2) { sin((start + it / 2).toDouble()).toFloat() }
        return tensor(Shape(length, 2), c) to tensor(Shape(length, 2), s)
    }
    private val inputs = arrayOf(floatArrayOf(1f, 2f, 3f, 4f, -1f, .5f, 2f, -3f), floatArrayOf(-1f, .5f, 2f, -3f, 1f, 2f, 3f, 4f))
    private fun mask(past: Int, length: Int) = tensor(Shape(length, past + length), FloatArray(length * (past + length)) { i ->
        if (i % (past + length) <= past + i / (past + length)) 0f else -Float.MAX_VALUE
    })

    @Test
    fun realHeadOrderRotaryCausalityAndStreamingMatchIndependentScalarAttention() {
        for (policy in AttentionLayoutPolicy.entries) {
            val op = operation(policy)
            val (c, s) = positions(0, 2)
            val result = op.forward(tensor(Shape(2, 8), inputs.flatMap { it.toList() }.toFloatArray()), null, null, c, s, mask(0, 2), ctx)
            val normalized = inputs.map { x ->
                val scale = 1.0 / sqrt(x.sumOf { it.toDouble() * it } / 8 + 1e-5)
                DoubleArray(8) { x[it] * scale * norm[it] }
            }
            fun headNorm(x: DoubleArray, start: Int, weights: DoubleArray): DoubleArray {
                val denominator = sqrt((x[start] * x[start] + x[start + 1] * x[start + 1]) / 2 + 1e-5)
                return DoubleArray(2) { x[start + it] / denominator * weights[it] }
            }
            fun rotate(x: DoubleArray, position: Int): DoubleArray {
                val cosine = cos(position.toDouble()); val sine = sin(position.toDouble())
                return doubleArrayOf(x[0] * cosine - x[1] * sine, x[1] * cosine + x[0] * sine)
            }
            val keys = Array(2) { kv -> Array(2) { token -> rotate(headNorm(normalized[token], 2 * kv, doubleArrayOf(.9, 1.1)), token) } }
            val values = Array(2) { kv -> Array(2) { token -> doubleArrayOf(normalized[token][4 + 2 * kv], normalized[token][5 + 2 * kv]) } }
            val expected = DoubleArray(16)
            for (token in 0..1) for (head in 0..3) {
                val q = rotate(headNorm(normalized[token], 2 * head, doubleArrayOf(1.25, .75)), token)
                val logits = DoubleArray(token + 1) { key -> (q[0] * keys[head / 2][key][0] + q[1] * keys[head / 2][key][1]) / sqrt(2.0) }
                val max = logits.max(); val exps = logits.map { exp(it - max) }; val total = exps.sum()
                for (dim in 0..1) expected[token * 8 + head * 2 + dim] = inputs[token][head * 2 + dim] + exps.indices.sumOf { key -> exps[key] / total * values[head / 2][key][dim] }
            }
            val actual = result.output.data.copyToFloatArray()
            for (i in expected.indices) assertEquals(expected[i].toFloat(), actual[i], 3e-6f, "independent output $i")
            val actualKeys = result.nextKey.data.copyToFloatArray()
            val actualValues = result.nextValue.data.copyToFloatArray()
            for (kv in 0..1) for (token in 0..1) for (dim in 0..1) {
                val i = (kv * 2 + token) * 2 + dim
                assertEquals(keys[kv][token][dim].toFloat(), actualKeys[i], 2e-6f)
                assertEquals(values[kv][token][dim].toFloat(), actualValues[i], 2e-6f)
            }
            val (c0, s0) = positions(0, 1)
            val first = op.forward(tensor(Shape(1, 8), inputs[0]), null, null, c0, s0, mask(0, 1), ctx)
            val (c1, s1) = positions(1, 1)
            val second = op.forward(tensor(Shape(1, 8), inputs[1]), first.nextKey, first.nextValue, c1, s1, mask(1, 1), ctx)
            assertContentEquals(actual.copyOfRange(0, 8), first.output.data.copyToFloatArray())
            assertContentEquals(actual.copyOfRange(8, 16), second.output.data.copyToFloatArray())
            assertContentEquals(actualKeys, second.nextKey.data.copyToFloatArray())
            assertContentEquals(actualValues, second.nextValue.data.copyToFloatArray())
        }
    }

    @Test
    fun directKvPerHeadPreservesRuntimeVariationsCachedFeedbackAndReset() {
        val baseline = operation()
        val direct = operation(AttentionLayoutPolicy.DIRECT_KV_PER_HEAD)
        var baselineKey: Tensor<FP32, Float>? = null; var baselineValue: Tensor<FP32, Float>? = null
        var directKey: Tensor<FP32, Float>? = null; var directValue: Tensor<FP32, Float>? = null
        fun close(a: Tensor<FP32, Float>, b: Tensor<FP32, Float>, name: String) {
            assertEquals(a.shape, b.shape, name)
            val av = a.data.copyToFloatArray(); val bv = b.data.copyToFloatArray()
            for (i in av.indices) assertEquals(av[i], bv[i], 2e-6f, "$name/$i")
        }
        for ((index, length) in listOf(3, 1, 2, 2).withIndex()) {
            if (index == 3) { baselineKey = null; baselineValue = null; directKey = null; directValue = null }
            val past = baselineKey?.shape?.get(1) ?: 0
            val (c, s) = positions(past + 2, length)
            val x = tensor(Shape(length, 8), FloatArray(length * 8) { (((it * 13 + index * 7) % 31) - 15) * 0.07f })
            val additive = tensor(Shape(length, past + length), FloatArray(length * (past + length)) { i ->
                if (i % (past + length) > past + i / (past + length)) -Float.MAX_VALUE else -(i % (past + length)) * 0.17f
            })
            val expected = baseline.forward(x, baselineKey, baselineValue, c, s, additive, ctx)
            val result = direct.forward(x, directKey, directValue, c, s, additive, ctx)
            close(expected.output, result.output, "output/$index")
            close(expected.nextKey, result.nextKey, "key/$index"); close(expected.nextValue, result.nextValue, "value/$index")
            for (name in expected.intermediates.keys) close(expected.intermediates.getValue(name), result.intermediates.getValue(name), "$name/$index")
            baselineKey = expected.nextKey; baselineValue = expected.nextValue
            directKey = result.nextKey; directValue = result.nextValue
            assertEquals(past + length, result.nextKey.shape[1])
        }
    }

    @Test
    fun directKvExportUsesOnlyTwoDimensionalDotsAndPreservesRuntimeAbi() {
        assertEquals(AttentionLayoutPolicy.CURRENT_BATCHED, operation().layoutPolicy)
        for (past in listOf(0, 3)) {
            val traced = traceLfm2Attention(operation(AttentionLayoutPolicy.DIRECT_KV_PER_HEAD), past, 2)
            assertEquals(if (past == 0) 4 else 6, traced.module.inputSpecs.size)
            assertEquals(3, traced.module.outputSpecs.size)
            assertEquals(11, traced.module.externalParameters.size)
            assertFalse(traced.module.content.contains("batching_dims"))
            assertEquals(15, traced.module.content.lineSequence().count { it.contains("stablehlo.dot_general") })
            assertTrue(traced.module.content.contains("across dimensions = [2]"))
            assertFalse(traced.module.content.contains("custom_call"))
        }
    }

    @Test
    fun prefillAndCachedExportExposeAllRuntimeDataAndElevenLearnedParameters() {
        for (past in listOf(0, 3)) {
            val traced = traceLfm2Attention(operation(), past, 2)
            assertEquals(if (past == 0) 4 else 6, traced.module.inputSpecs.size)
            assertEquals(3, traced.module.outputSpecs.size)
            assertEquals(11, traced.module.externalParameters.size)
            assertTrue(traced.module.content.contains("across dimensions = [2]"))
            assertFalse(traced.module.content.contains("custom_call"))
        }
    }

    @Test
    fun inconsistentKvAndMaskShapesAreRejectedBeforeAttentionExecution() {
        val op = operation(); val (c, s) = positions(0, 1); val input = tensor(Shape(1, 8), inputs[0])
        assertFailsWith<IllegalArgumentException> { op.forward(input, tensor(Shape(2, 1, 2), FloatArray(4)), null, c, s, mask(1, 1), ctx) }
        assertFailsWith<IllegalArgumentException> { op.forward(input, null, null, c, s, tensor(Shape(1, 2), FloatArray(2)), ctx) }
        assertFailsWith<IllegalArgumentException> { op.forward(input, null, null, tensor(Shape(2, 2), FloatArray(4)), s, mask(0, 1), ctx) }
    }
}
