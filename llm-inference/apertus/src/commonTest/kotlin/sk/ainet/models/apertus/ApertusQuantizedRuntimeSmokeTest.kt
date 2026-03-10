package sk.ainet.models.apertus

import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Smoke test: builds a tiny Apertus model with quantized (simulated) weights
 * and verifies the lazy-dequant runtime produces finite logits and generates tokens.
 *
 * Uses F32-typed QuantizedTensors stored in GGUF column-major format as an identity
 * dequantization, so we can verify correctness against the eager FP32 runtime.
 */
class ApertusQuantizedRuntimeSmokeTest {

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

    private fun randnArray(shape: Shape, seed: Int): FloatArray {
        val rng = kotlin.random.Random(seed)
        return FloatArray(shape.volume) { (rng.nextFloat() - 0.5f) * 0.1f }
    }

    private fun randn(shape: Shape, seed: Int = 42): Tensor<FP32, Float> =
        ctx.fromFloatArray(shape, FP32::class, randnArray(shape, seed))

    private fun floatArrayToBytes(data: FloatArray): ByteArray {
        val bytes = ByteArray(data.size * 4)
        for (i in data.indices) {
            val bits = data[i].toRawBits()
            bytes[i * 4] = (bits and 0xFF).toByte()
            bytes[i * 4 + 1] = ((bits shr 8) and 0xFF).toByte()
            bytes[i * 4 + 2] = ((bits shr 16) and 0xFF).toByte()
            bytes[i * 4 + 3] = ((bits shr 24) and 0xFF).toByte()
        }
        return bytes
    }

    /**
     * Create a QuantizedTensor in GGUF column-major format from an eager row-major weight.
     *
     * For 2D tensors, GGUF stores shape [ne0, ne1] where ne0 = cols (fast dim),
     * ne1 = rows, with data in column-major order. [dequant2D] will transpose this
     * back to row-major [ne1, ne0] and then the runtime uses `.t()`.
     *
     * The eager runtime stores shape [rows, cols] in row-major and also uses `.t()`.
     * Both paths produce the same effective matrix in the matmul.
     *
     * @param eagerShape The shape the eager runtime uses (row-major [rows, cols])
     * @param rowMajorData The eager row-major float data
     */
    private fun asQuantizedGGUF(eagerShape: Shape, rowMajorData: FloatArray): QuantizedTensor {
        if (eagerShape.rank == 2) {
            val rows = eagerShape[0]
            val cols = eagerShape[1]
            // Convert row-major [rows, cols] to column-major [cols, rows] (GGUF layout)
            val colMajor = FloatArray(rowMajorData.size)
            for (r in 0 until rows) {
                for (c in 0 until cols) {
                    colMajor[c * rows + r] = rowMajorData[r * cols + c]
                }
            }
            return QuantizedTensor(
                data = floatArrayToBytes(colMajor),
                quantType = GGMLQuantizationType.F32,
                shape = Shape(cols, rows), // GGUF shape: [ne0=cols, ne1=rows]
                nElements = eagerShape.volume
            )
        }
        return QuantizedTensor(
            data = floatArrayToBytes(rowMajorData),
            quantType = GGMLQuantizationType.F32,
            shape = eagerShape,
            nElements = eagerShape.volume
        )
    }

    /** Convenience: generate random data and create a GGUF-format QuantizedTensor. */
    private fun randnQuantized(eagerShape: Shape, seed: Int): QuantizedTensor =
        asQuantizedGGUF(eagerShape, randnArray(eagerShape, seed))

    private fun buildMetadata() = ApertusModelMetadata(
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

    private val xieluParams = ApertusXIELUParams(-0.5f, -0.3f, 0.8f, -5.0f)

    @Test
    fun quantizedForwardPassProducesFiniteLogits() {
        val metadata = buildMetadata()
        val layer = ApertusQuantizedLayerWeights(
            attnNorm = ones(Shape(dim)),
            qNorm = ones(Shape(headDim)),
            kNorm = ones(Shape(headDim)),
            ffnNorm = ones(Shape(dim)),
            xieluParams = xieluParams,
            wq = randnQuantized(Shape(dim, dim), seed = 1),
            wk = randnQuantized(Shape(kvDim, dim), seed = 2),
            wv = randnQuantized(Shape(kvDim, dim), seed = 3),
            wo = randnQuantized(Shape(dim, dim), seed = 4),
            ffnUp = randnQuantized(Shape(ffDim, dim), seed = 6),
            ffnDown = randnQuantized(Shape(dim, ffDim), seed = 5)
        )
        val weights = ApertusQuantizedRuntimeWeights(
            metadata = metadata,
            tokenEmbedding = randn(Shape(vocabSize, dim), seed = 10),
            layers = listOf(layer),
            outputNorm = ones(Shape(dim)),
            outputWeight = randnQuantized(Shape(vocabSize, dim), seed = 11)
        )
        val backend = ApertusCpuAttentionBackend<FP32>(
            ctx = ctx, metadata = metadata, dtype = FP32::class, ropeFreqBase = 12000000f
        )
        val runtime = ApertusQuantizedRuntime(
            ctx = ctx, weights = weights, attentionBackend = backend
        )

        val logits = runtime.forward(1)
        assertEquals(2, logits.shape.rank, "logits should be 2D")
        assertEquals(vocabSize, logits.shape[1], "logits dim should match vocab size")
        val buf = logits.data.copyToFloatArray()
        for (i in buf.indices) {
            assertTrue(buf[i].isFinite(), "logit[$i] = ${buf[i]} is not finite")
        }
    }

    @Test
    fun quantizedAndEagerProduceSameLogits() {
        val metadata = buildMetadata()
        val tokenEmb = randn(Shape(vocabSize, dim), seed = 10)
        val outputNorm = ones(Shape(dim))

        // Generate weight data arrays (same seeds for both runtimes)
        val wqData = randnArray(Shape(dim, dim), seed = 1)
        val wkData = randnArray(Shape(kvDim, dim), seed = 2)
        val wvData = randnArray(Shape(kvDim, dim), seed = 3)
        val woData = randnArray(Shape(dim, dim), seed = 4)
        val ffnDownData = randnArray(Shape(dim, ffDim), seed = 5)
        val ffnUpData = randnArray(Shape(ffDim, dim), seed = 6)
        val outputWData = randnArray(Shape(vocabSize, dim), seed = 11)

        // --- Eager FP32 runtime ---
        val eagerLayer = ApertusLayerWeights(
            attnNorm = ones(Shape(dim)),
            wq = ctx.fromFloatArray(Shape(dim, dim), FP32::class, wqData.copyOf()),
            wk = ctx.fromFloatArray(Shape(kvDim, dim), FP32::class, wkData.copyOf()),
            wv = ctx.fromFloatArray(Shape(kvDim, dim), FP32::class, wvData.copyOf()),
            wo = ctx.fromFloatArray(Shape(dim, dim), FP32::class, woData.copyOf()),
            qNorm = ones(Shape(headDim)),
            kNorm = ones(Shape(headDim)),
            ffnNorm = ones(Shape(dim)),
            ffnDown = ctx.fromFloatArray(Shape(dim, ffDim), FP32::class, ffnDownData.copyOf()),
            ffnUp = ctx.fromFloatArray(Shape(ffDim, dim), FP32::class, ffnUpData.copyOf()),
            xieluParams = xieluParams
        )
        val eagerWeights = ApertusRuntimeWeights(
            metadata = metadata, tokenEmbedding = tokenEmb,
            layers = listOf(eagerLayer), outputNorm = outputNorm,
            outputWeight = ctx.fromFloatArray(Shape(vocabSize, dim), FP32::class, outputWData.copyOf())
        )
        val eagerBackend = ApertusCpuAttentionBackend<FP32>(
            ctx = ctx, weights = eagerWeights, dtype = FP32::class
        )
        val eagerRuntime = ApertusRuntime(
            ctx = ctx, weights = eagerWeights,
            attentionBackend = eagerBackend, dtype = FP32::class
        )

        // --- Quantized runtime (F32 bytes = identity dequant, GGUF column-major layout) ---
        val qLayer = ApertusQuantizedLayerWeights(
            attnNorm = ones(Shape(dim)),
            qNorm = ones(Shape(headDim)),
            kNorm = ones(Shape(headDim)),
            ffnNorm = ones(Shape(dim)),
            xieluParams = xieluParams,
            wq = asQuantizedGGUF(Shape(dim, dim), wqData.copyOf()),
            wk = asQuantizedGGUF(Shape(kvDim, dim), wkData.copyOf()),
            wv = asQuantizedGGUF(Shape(kvDim, dim), wvData.copyOf()),
            wo = asQuantizedGGUF(Shape(dim, dim), woData.copyOf()),
            ffnUp = asQuantizedGGUF(Shape(ffDim, dim), ffnUpData.copyOf()),
            ffnDown = asQuantizedGGUF(Shape(dim, ffDim), ffnDownData.copyOf())
        )
        val qWeights = ApertusQuantizedRuntimeWeights(
            metadata = metadata, tokenEmbedding = tokenEmb,
            layers = listOf(qLayer), outputNorm = outputNorm,
            outputWeight = asQuantizedGGUF(Shape(vocabSize, dim), outputWData.copyOf())
        )
        val qBackend = ApertusCpuAttentionBackend<FP32>(
            ctx = ctx, metadata = metadata, dtype = FP32::class
        )
        val qRuntime = ApertusQuantizedRuntime(
            ctx = ctx, weights = qWeights, attentionBackend = qBackend
        )

        // Compare logits
        val eagerLogits = eagerRuntime.forward(1).data.copyToFloatArray()
        val quantLogits = qRuntime.forward(1).data.copyToFloatArray()

        assertEquals(eagerLogits.size, quantLogits.size, "logit sizes should match")
        for (i in eagerLogits.indices) {
            val diff = kotlin.math.abs(eagerLogits[i] - quantLogits[i])
            assertTrue(diff < 1e-4f, "logit[$i] diff=$diff: eager=${eagerLogits[i]} vs quant=${quantLogits[i]}")
        }
    }

    @Test
    fun quantizedGenerateProducesTokens() {
        val metadata = buildMetadata()
        val layer = ApertusQuantizedLayerWeights(
            attnNorm = ones(Shape(dim)),
            qNorm = ones(Shape(headDim)),
            kNorm = ones(Shape(headDim)),
            ffnNorm = ones(Shape(dim)),
            xieluParams = xieluParams,
            wq = randnQuantized(Shape(dim, dim), seed = 1),
            wk = randnQuantized(Shape(kvDim, dim), seed = 2),
            wv = randnQuantized(Shape(kvDim, dim), seed = 3),
            wo = randnQuantized(Shape(dim, dim), seed = 4),
            ffnUp = randnQuantized(Shape(ffDim, dim), seed = 6),
            ffnDown = randnQuantized(Shape(dim, ffDim), seed = 5)
        )
        val weights = ApertusQuantizedRuntimeWeights(
            metadata = metadata,
            tokenEmbedding = randn(Shape(vocabSize, dim), seed = 10),
            layers = listOf(layer),
            outputNorm = ones(Shape(dim)),
            outputWeight = randnQuantized(Shape(vocabSize, dim), seed = 11)
        )
        val backend = ApertusCpuAttentionBackend<FP32>(
            ctx = ctx, metadata = metadata, dtype = FP32::class
        )
        val runtime = ApertusQuantizedRuntime(
            ctx = ctx, weights = weights, attentionBackend = backend
        )

        val generated = mutableListOf<Int>()
        runtime.generate(
            prompt = intArrayOf(1, 5, 3), steps = 4, temperature = 1.0f
        ) { generated.add(it) }

        assertEquals(4, generated.size, "Should generate exactly 4 tokens")
        for (tokenId in generated) {
            assertTrue(tokenId in 0 until vocabSize, "Token $tokenId should be in [0, $vocabSize)")
        }
    }
}
