package sk.ainet.models.gemma

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.types.FP32
import sk.ainet.tape.Execution
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Regression test for SKaiNET#763: [PositionalKVCache.update] must wire K/V functionally while
 * tracing (ctx.isRecording) instead of round-tripping through its heap buffer, which under tracing
 * baked an all-zero KV-cache as `stablehlo.constant` and disconnected the computed k_proj/v_proj —
 * so the exported decoder attended over K=V=0. (Masked by the unnormalized FFN in non-sandwich
 * configs; exposed by Gemma's post_ffw_norm.)
 */
class KvCacheTraceFidelityTest {
    @Test
    fun tracedDecoderKeepsComputedKV() {
        val meta = Gemma4ModelMetadata(
            architecture = "gemma3", embeddingLength = 64, contextLength = 128, blockCount = 2,
            headCount = 2, kvHeadCount = 1, intermediateSize = 128, headDim = 32, globalHeadDim = 32,
            vocabSize = 48, slidingWindow = 64, kvSharedLayers = 0, layerTypes = List(2) { "full_attention" },
            ropeParametersFull = Gemma4RopeConfig(base = 10000.0f),
            ropeParametersSliding = Gemma4RopeConfig(base = 10000.0f), maxPositionEmbeddings = 128,
        )
        val model = gemmaNetwork<FP32, Float>(meta, FP32::class, maxInferenceLen = 4, sandwichNorms = true)
        val input = VoidOpsTensor(object : TensorData<FP32, Float> {
            override val shape = Shape(4); override fun get(vararg i: Int) = 0.0f; override fun set(vararg i: Int, value: Float) {}
        }, FP32::class)
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val tape = ctx.record {
            val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
            Execution.tapeStack.pushTape(ct)
            try { model.forward(input, this as ExecutionContext) } finally { Execution.tapeStack.popTape() }
        }.first
        val graph = (tape as DefaultExecutionTape).toComputeGraph(synthesizeExternalInputs = true, embedConstants = true)
        val mlir = sk.ainet.compile.hlo.toStableHlo(graph, "gemma").content

        // The KV head shape here is [1, 4, 32] (nKVHeads=1, seq=4, headDim=32). Before the fix the
        // K and V for each block appeared as `stablehlo.constant dense<0.0> : tensor<1x4x32xf32>`.
        val kvZeroConstants = Regex("stablehlo\\.constant dense<0\\.0> : tensor<1x4x32xf32>").findAll(mlir).count()
        assertEquals(0, kvZeroConstants,
            "Traced sandwich Gemma decoder baked $kvZeroConstants zero KV-cache constants — K/V not wired (regression of #763)")
        // Sanity: K/V projections are present as real dot_generals (q,k,v,o + ffn etc.).
        assertTrue(Regex("dot_general").findAll(mlir).count() >= 6, "expected real projection dot_generals in the export")
    }
}
