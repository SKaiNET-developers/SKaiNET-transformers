package sk.ainet.models.apertus

import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Smoke test: builds a tiny Apertus model (dim=8, 1 layer, vocab=16)
 * with random weights and verifies that forward pass produces finite logits.
 */
class ApertusRuntimeSmokeTest {

    private val dim = 8
    private val ffDim = 16
    private val vocabSize = 16
    private val nHeads = 2
    private val kvHeads = 2
    private val headDim = dim / nHeads
    private val kvDim = kvHeads * headDim

    private val ctx = DefaultDataExecutionContext()

    private fun ones(shape: Shape): Tensor<FP32, Float> {
        val values = FloatArray(shape.volume) { 0.01f }
        return ctx.fromFloatArray(shape, FP32::class, values)
    }

    private fun randn(shape: Shape, seed: Int = 42): Tensor<FP32, Float> {
        val rng = kotlin.random.Random(seed)
        val values = FloatArray(shape.volume) { (rng.nextFloat() - 0.5f) * 0.1f }
        return ctx.fromFloatArray(shape, FP32::class, values)
    }

    @Test
    fun forwardPassProducesFiniteLogits() {
        val metadata = ApertusModelMetadata(
            architecture = "apertus",
            embeddingLength = dim,
            contextLength = 32,
            blockCount = 1,
            headCount = nHeads,
            kvHeadCount = kvHeads,
            feedForwardLength = ffDim,
            ropeDimensionCount = headDim,
            vocabSize = vocabSize,
            ropeTheta = 12000000f,
            qkNorm = true,
            hiddenAct = "xielu",
            tiedEmbeddings = false
        )

        val layer = ApertusLayerWeights(
            attnNorm = ones(Shape(dim)),
            wq = randn(Shape(dim, dim), seed = 1),
            wk = randn(Shape(kvDim, dim), seed = 2),
            wv = randn(Shape(kvDim, dim), seed = 3),
            wo = randn(Shape(dim, dim), seed = 4),
            qNorm = ones(Shape(headDim)),
            kNorm = ones(Shape(headDim)),
            ffnNorm = ones(Shape(dim)),
            ffnDown = randn(Shape(dim, ffDim), seed = 5),
            ffnUp = randn(Shape(ffDim, dim), seed = 6),
            xieluParams = ApertusXIELUParams(
                alphaP = -0.5f,
                alphaN = -0.3f,
                beta = 0.8f,
                eps = -5.0f
            )
        )

        val weights = ApertusRuntimeWeights(
            metadata = metadata,
            tokenEmbedding = randn(Shape(vocabSize, dim), seed = 10),
            layers = listOf(layer),
            outputNorm = ones(Shape(dim)),
            outputWeight = randn(Shape(vocabSize, dim), seed = 11)
        )

        val backend = ApertusCpuAttentionBackend<FP32>(
            ctx = ctx,
            weights = weights,
            dtype = FP32::class,
            ropeFreqBase = 12000000f
        )

        val runtime = ApertusRuntime(
            ctx = ctx,
            weights = weights,
            attentionBackend = backend,
            dtype = FP32::class
        )

        // Forward pass with token ID 1 (BOS)
        val logits = runtime.forward(1)

        // Verify shape: should be [1, vocabSize]
        assertEquals(2, logits.shape.rank, "logits should be 2D")
        assertEquals(vocabSize, logits.shape[1], "logits dim should match vocab size")

        // Verify all values are finite
        val buf = logits.data.copyToFloatArray()
        for (i in buf.indices) {
            assertTrue(buf[i].isFinite(), "logit[$i] = ${buf[i]} is not finite")
        }
    }

    @Test
    fun generateProducesTokens() {
        val metadata = ApertusModelMetadata(
            architecture = "apertus",
            embeddingLength = dim,
            contextLength = 32,
            blockCount = 1,
            headCount = nHeads,
            kvHeadCount = kvHeads,
            feedForwardLength = ffDim,
            ropeDimensionCount = headDim,
            vocabSize = vocabSize,
            ropeTheta = 12000000f
        )

        val layer = ApertusLayerWeights(
            attnNorm = ones(Shape(dim)),
            wq = randn(Shape(dim, dim), seed = 1),
            wk = randn(Shape(kvDim, dim), seed = 2),
            wv = randn(Shape(kvDim, dim), seed = 3),
            wo = randn(Shape(dim, dim), seed = 4),
            qNorm = ones(Shape(headDim)),
            kNorm = ones(Shape(headDim)),
            ffnNorm = ones(Shape(dim)),
            ffnDown = randn(Shape(dim, ffDim), seed = 5),
            ffnUp = randn(Shape(ffDim, dim), seed = 6),
            xieluParams = ApertusXIELUParams(-0.5f, -0.3f, 0.8f, -5.0f)
        )

        val weights = ApertusRuntimeWeights(
            metadata = metadata,
            tokenEmbedding = randn(Shape(vocabSize, dim), seed = 10),
            layers = listOf(layer),
            outputNorm = ones(Shape(dim)),
            outputWeight = randn(Shape(vocabSize, dim), seed = 11)
        )

        val backend = ApertusCpuAttentionBackend<FP32>(
            ctx = ctx,
            weights = weights,
            dtype = FP32::class
        )

        val runtime = ApertusRuntime(
            ctx = ctx,
            weights = weights,
            attentionBackend = backend,
            dtype = FP32::class
        )

        val generated = mutableListOf<Int>()
        runtime.generate(
            prompt = intArrayOf(1, 5, 3),
            steps = 4,
            temperature = 1.0f
        ) { tokenId ->
            generated.add(tokenId)
        }

        assertEquals(4, generated.size, "Should generate exactly 4 tokens")
        for (tokenId in generated) {
            assertTrue(tokenId in 0 until vocabSize, "Token $tokenId should be in [0, $vocabSize)")
        }
    }
}
