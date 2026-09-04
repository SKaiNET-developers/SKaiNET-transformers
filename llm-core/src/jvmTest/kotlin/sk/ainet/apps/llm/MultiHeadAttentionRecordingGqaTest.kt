package sk.ainet.apps.llm

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.context.schedule.Schedule
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.nn.transformer.MultiHeadAttention
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * SKEEP-005 phase 2: the recording path of grouped-query attention hands K/V to SDPA with
 * their own head count. The tape carries no `narrow`/`concat` expansion, the sdpa node sees
 * K/V with nKVHeads, and the recorded forward equals the eager one.
 */
class MultiHeadAttentionRecordingGqaTest {

    private val dim = 64

    private fun weights(ctx: ExecutionContext, out: Int, inDim: Int, seed: Int): Tensor<FP32, Float> =
        ctx.fromFloatArray(Shape(out, inDim), FP32::class, FloatArray(out * inDim) { i -> kotlin.math.sin((seed * 1000 + i).toFloat()) * 0.3f })

    private fun mha(ctx: ExecutionContext, nHeads: Int, nKVHeads: Int): MultiHeadAttention<FP32, Float> {
        val headDim = dim / nHeads
        val m = MultiHeadAttention<FP32, Float>(dim = dim, nHeads = nHeads, nKVHeads = nKVHeads, causal = true, kvCache = null, name = "attn")
        m.params[0].value = weights(ctx, nHeads * headDim, dim, 1)
        m.params[1].value = weights(ctx, nKVHeads * headDim, dim, 2)
        m.params[2].value = weights(ctx, nKVHeads * headDim, dim, 3)
        m.params[3].value = weights(ctx, dim, nHeads * headDim, 4)
        return m
    }

    @Test
    fun recordedGqaAttentionCarriesNoHeadExpansion() {
        val ctx = DirectCpuExecutionContext(schedule = Schedule.Sequential)
        val m = mha(ctx, nHeads = 8, nKVHeads = 2)
        val x = ctx.fromFloatArray<FP32, Float>(Shape(16, dim), FP32::class, FloatArray(16 * dim) { i -> ((i * 7 + 13) % 17 - 8) / 8f })
        val eager = m.forward(x, ctx).data.copyToFloatArray()

        val taping = DefaultGraphExecutionContext.tape(baseOps = ctx.ops)
        taping.startRecording()
        val recorded = m.forward(x, taping).data.copyToFloatArray()
        val tape = taping.stopRecording() as DefaultExecutionTape

        assertContentEquals(eager, recorded, "recording must not change the numbers")
        val graph = tape.toComputeGraph(synthesizeExternalInputs = true, inputTensorIds = emptySet(), embedConstants = false)
        val names = graph.nodes.map { it.operation.name.lowercase() }
        assertTrue(names.none { it == "narrow" || it == "concat" }, "no K/V head expansion on the tape: $names")
        val sdpa = graph.nodes.single { it.operation.name.lowercase().let { n -> n == "scaleddotproductattention" || n == "sdpa" } }
        assertEquals(listOf(1, 8, 16, dim / 8), sdpa.inputs[0].shape, "Q keeps nHeads")
        assertEquals(listOf(1, 2, 16, dim / 8), sdpa.inputs[1].shape, "K keeps nKVHeads")
        assertEquals(listOf(1, 2, 16, dim / 8), sdpa.inputs[2].shape, "V keeps nKVHeads")
    }
}
