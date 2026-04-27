package sk.ainet.models.gemma

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

/**
 * End-to-end test for the Gemma DSL pipeline (Phase 5a of
 * PLAN-unified-pipeline.md):
 *
 *   GemmaNetworkLoader.fromWeights → gemmaNetwork() → OptimizedLLMRuntime
 *                                 → DAG → CPU-on-JVM
 *
 * Uses a tiny model (dim=8, 1 layer, vocab=16) with deterministic
 * weights so no external model file is needed. Mirrors
 * QwenDslPipelineTest.
 */
class GemmaDslPipelineTest {

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

    private fun ones(shape: Shape): Tensor<FP32, Float> {
        val values = FloatArray(shape.volume) { 1.0f }
        return ctx.fromFloatArray(shape, FP32::class, values)
    }

    private val metadata = Gemma4ModelMetadata(
        architecture = "gemma4",
        embeddingLength = dim,
        contextLength = seqLen,
        blockCount = 1,
        headCount = nHeads,
        kvHeadCount = kvHeads,
        intermediateSize = ffDim,
        headDim = headDim,
        globalHeadDim = headDim,
        vocabSize = vocabSize,
        slidingWindow = seqLen,
        kvSharedLayers = 0,
        layerTypes = listOf("full_attention"),
        ropeParametersFull = Gemma4RopeConfig(base = 10000f),
        ropeParametersSliding = Gemma4RopeConfig(base = 10000f),
        maxPositionEmbeddings = seqLen
    )

    private fun buildWeights(): Gemma4Weights<FP32, Float> {
        val tensors = linkedMapOf<String, Tensor<FP32, Float>>(
            Gemma4TensorNames.TOKEN_EMBEDDINGS to randn(Shape(vocabSize, dim), seed = 10),
            Gemma4TensorNames.OUTPUT_NORM to ones(Shape(dim)),
            Gemma4TensorNames.OUTPUT_WEIGHT to randn(Shape(vocabSize, dim), seed = 11),
            Gemma4TensorNames.inputLayernorm(0) to ones(Shape(dim)),
            Gemma4TensorNames.attnQ(0) to randn(Shape(dim, dim), seed = 1),
            Gemma4TensorNames.attnK(0) to randn(Shape(dim, dim), seed = 2),
            Gemma4TensorNames.attnV(0) to randn(Shape(dim, dim), seed = 3),
            Gemma4TensorNames.attnOut(0) to randn(Shape(dim, dim), seed = 4),
            Gemma4TensorNames.postAttentionLayernorm(0) to ones(Shape(dim)),
            Gemma4TensorNames.ffnGate(0) to randn(Shape(ffDim, dim), seed = 5),
            Gemma4TensorNames.ffnDown(0) to randn(Shape(dim, ffDim), seed = 6),
            Gemma4TensorNames.ffnUp(0) to randn(Shape(ffDim, dim), seed = 7)
        )
        return Gemma4Weights(metadata = metadata, tensors = tensors)
    }

    @Test
    fun `DSL network definition builds correct module tree`() {
        val model = gemmaNetwork<FP32, Float>(metadata)

        val topLevelNames = model.modules.map { it.name }
        assertTrue("token_embd" in topLevelNames, "Should have token_embd module")
        assertTrue("blk.0" in topLevelNames, "Should have blk.0 module")
        assertTrue("output_norm" in topLevelNames, "Should have output_norm module")
        assertTrue("output" in topLevelNames, "Should have output module")
    }

    @Test
    fun `weight loading maps all parameters`() {
        val model = GemmaNetworkLoader.fromWeights(ctx, buildWeights())
        assertTrue(model.modules.isNotEmpty(), "Model should have child modules after weight loading")
    }

    @Test
    fun `DIRECT mode forward produces finite logits with correct shape`() {
        val model = GemmaNetworkLoader.fromWeights(ctx, buildWeights())

        val runtime = OptimizedLLMRuntime(
            model = model,
            ctx = ctx,
            mode = OptimizedLLMMode.DIRECT,
            dtype = FP32::class
        )

        val testTokens = intArrayOf(1, 5, 3)
        for (tokenId in testTokens) {
            val logits = runtime.forward(tokenId)

            assertEquals(
                vocabSize,
                logits.shape[logits.shape.rank - 1],
                "Last dim should be vocabSize for token $tokenId"
            )

            val buf = logits.data.copyToFloatArray()
            for (i in buf.indices) {
                assertTrue(buf[i].isFinite(), "logit[$i] = ${buf[i]} is not finite for token $tokenId")
            }
        }
    }

    @Test
    fun `DIRECT mode forward is deterministic`() {
        val weights = buildWeights()

        val runtime1 = OptimizedLLMRuntime(
            GemmaNetworkLoader.fromWeights(ctx, weights),
            ctx,
            OptimizedLLMMode.DIRECT,
            FP32::class
        )
        val runtime2 = OptimizedLLMRuntime(
            GemmaNetworkLoader.fromWeights(ctx, weights),
            ctx,
            OptimizedLLMMode.DIRECT,
            FP32::class
        )

        val logits1 = runtime1.forward(1).data.copyToFloatArray()
        val logits2 = runtime2.forward(1).data.copyToFloatArray()

        assertEquals(logits1.size, logits2.size)
        for (i in logits1.indices) {
            assertEquals(logits1[i], logits2[i], "Logit[$i] differs between runs")
        }
    }

    @Test
    fun `generate produces valid token sequence`() {
        val model = GemmaNetworkLoader.fromWeights(ctx, buildWeights())

        val runtime = OptimizedLLMRuntime(
            model = model,
            ctx = ctx,
            mode = OptimizedLLMMode.DIRECT,
            dtype = FP32::class
        )

        val generated = mutableListOf<Int>()
        runtime.generate(
            prompt = intArrayOf(1, 5),
            steps = 4,
            temperature = 1.0f
        ) { tokenId ->
            generated.add(tokenId)
        }

        assertEquals(4, generated.size, "Should generate exactly 4 tokens")
        for (tokenId in generated) {
            assertTrue(
                tokenId in 0 until vocabSize,
                "Generated token $tokenId should be in [0, $vocabSize)"
            )
        }
    }

    @Test
    fun `logits change with different input tokens`() {
        val model = GemmaNetworkLoader.fromWeights(ctx, buildWeights())
        val runtime = OptimizedLLMRuntime(model, ctx, OptimizedLLMMode.DIRECT, FP32::class)

        val logits1 = runtime.forward(1).data.copyToFloatArray()
        val logits2 = runtime.forward(5).data.copyToFloatArray()

        var allSame = true
        for (i in logits1.indices) {
            if (abs(logits1[i] - logits2[i]) > 1e-6f) {
                allSame = false
                break
            }
        }
        assertTrue(!allSame, "Different tokens should produce different logits")
    }
}
