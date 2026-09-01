package sk.ainet.models.llama

import sk.ainet.lang.nn.dsl.decoder.DecoderGgufWeights
import sk.ainet.lang.nn.dsl.decoder.DecoderTensorNames
import sk.ainet.lang.nn.dsl.decoder.GgufDecoderMetadata

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

/**
 * #343 steady-state pins for the per-step [sk.ainet.lang.memory.ForwardScope]
 * in [OptimizedLLMRuntime]'s DIRECT decode path:
 *
 * 1. **Bit-identity** — a scoped decode produces exactly the logits of an
 *    unscoped one (`forwardSlabFloats = 0`), step for step. The slab changes
 *    where activations live, never what they hold.
 * 2. **Flat peak** — `peakFloats` stops growing once decode is warm. The KV
 *    history lives outside the scope (the cache detaches it to ambient
 *    storage), so per-step slab use grows only with the KV *views* concat
 *    produces — bounded by seqLen, and identical for the same step count on
 *    every run.
 * 3. **Zero overflow** — the default slab holds every step of this model;
 *    `overflowBytes` stays 0 after each forward.
 *
 * Tiny deterministic model, same scaffolding as [LlamaDslPipelineTest].
 */
@OptIn(ExperimentalMemoryApi::class)
class ForwardScopeSteadyStateTest {

    private val dim = 8
    private val ffDim = 16
    private val vocabSize = 16
    private val nHeads = 2
    private val kvHeads = 2
    private val headDim = dim / nHeads
    private val seqLen = 32

    private val ctx = DirectCpuExecutionContext()

    private fun randn(shape: Shape, seed: Int): Tensor<FP32, Float> {
        val rng = kotlin.random.Random(seed)
        val values = FloatArray(shape.volume) { (rng.nextFloat() - 0.5f) * 0.1f }
        return ctx.fromFloatArray(shape, FP32::class, values)
    }

    private fun ones(shape: Shape): Tensor<FP32, Float> =
        ctx.fromFloatArray(shape, FP32::class, FloatArray(shape.volume) { 1.0f })

    private val metadata = GgufDecoderMetadata(
        architecture = "llama",
        embeddingLength = dim,
        contextLength = seqLen,
        blockCount = 1,
        headCount = nHeads,
        kvHeadCount = kvHeads,
        feedForwardLength = ffDim,
        ropeDimensionCount = headDim,
        vocabSize = vocabSize,
    )

    private fun buildWeightTensors(): Map<String, Tensor<FP32, Float>> = linkedMapOf(
        DecoderTensorNames.TOKEN_EMBEDDINGS to randn(Shape(vocabSize, dim), seed = 10),
        DecoderTensorNames.OUTPUT_NORM to ones(Shape(dim)),
        DecoderTensorNames.OUTPUT_WEIGHT to randn(Shape(vocabSize, dim), seed = 11),
        DecoderTensorNames.attnNorm(0) to ones(Shape(dim)),
        DecoderTensorNames.attnQ(0) to randn(Shape(dim, dim), seed = 1),
        DecoderTensorNames.attnK(0) to randn(Shape(dim, dim), seed = 2),
        DecoderTensorNames.attnV(0) to randn(Shape(dim, dim), seed = 3),
        DecoderTensorNames.attnOut(0) to randn(Shape(dim, dim), seed = 4),
        DecoderTensorNames.ffnNorm(0) to ones(Shape(dim)),
        DecoderTensorNames.ffnGate(0) to randn(Shape(ffDim, dim), seed = 5),
        DecoderTensorNames.ffnDown(0) to randn(Shape(dim, ffDim), seed = 6),
        DecoderTensorNames.ffnUp(0) to randn(Shape(ffDim, dim), seed = 7),
    )

    private fun newRuntime(slabFloats: Int): OptimizedLLMRuntime<FP32> {
        val model = LlamaNetworkLoader.fromWeights(
            DecoderGgufWeights(metadata, buildWeightTensors())
        )
        return OptimizedLLMRuntime(
            model = model,
            ctx = ctx,
            mode = OptimizedLLMMode.DIRECT,
            dtype = FP32::class,
            forwardSlabFloats = slabFloats,
        )
    }

    @Test
    fun `scoped decode is bit-identical to unscoped and holds a flat zero-overflow slab`() {
        val scoped = newRuntime(OptimizedLLMRuntime.DEFAULT_FORWARD_SLAB_FLOATS)
        val unscoped = newRuntime(0)
        val tokens = intArrayOf(1, 5, 3, 7, 2, 9, 4, 11, 6, 13, 8, 15)

        var warmPeak = -1
        for ((step, tokenId) in tokens.withIndex()) {
            val a = scoped.forward(tokenId).data.copyToFloatArray()
            val b = unscoped.forward(tokenId).data.copyToFloatArray()
            assertEquals(b.size, a.size)
            for (i in a.indices) {
                // Bit-identical, not approximately equal: same kernels, same
                // values, different storage.
                assertEquals(
                    b[i].toRawBits(), a[i].toRawBits(),
                    "step=$step logit[$i]: scoped=${a[i]} unscoped=${b[i]}",
                )
            }

            val scope = scoped.forwardScopeMetrics
            assertNotNull(scope, "DIRECT decode with a slab must arm the forward scope")
            assertEquals(0L, scope.overflowBytes, "step=$step overflowed the slab")
            assertTrue(scope.usedFloats > 0, "step=$step allocated nothing from the slab — scope inert?")

            // KV concat views grow with history until seqKV saturates the
            // step's working set; by mid-sequence the peak must stop moving.
            if (step == tokens.size / 2) warmPeak = scope.peakFloats
        }

        val scope = scoped.forwardScopeMetrics!!
        assertTrue(warmPeak > 0)
        // Flat within the KV-view growth of the remaining steps: for this
        // tiny model that growth is a few hundred floats — pin it to the
        // same order, not to a magic constant.
        assertTrue(
            scope.peakFloats - warmPeak <= dim * seqLen,
            "peakFloats kept climbing after warm-up: warm=$warmPeak final=${scope.peakFloats}",
        )
        assertEquals(tokens.size.toLong() - 1, scope.steps, "one reset per decode step after the first")

        // The unscoped runtime never armed a scope.
        assertEquals(null, unscoped.forwardScopeMetrics)
    }
}
