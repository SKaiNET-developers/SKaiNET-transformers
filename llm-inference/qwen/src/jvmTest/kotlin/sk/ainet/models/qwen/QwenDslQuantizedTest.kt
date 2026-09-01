package sk.ainet.models.qwen

import java.lang.foreign.Arena
import java.lang.foreign.ValueLayout
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertTrue
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Q8MemorySegmentTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.FP32
import sk.ainet.lang.nn.dsl.decoder.DecoderGgufWeights
import sk.ainet.lang.nn.dsl.decoder.GgufDecoderMetadata
import sk.ainet.lang.nn.dsl.decoder.DecoderTensorNames

/**
 * Empirical probe: can the DSL Qwen path (`qwenNetwork()` +
 * `OptimizedLLMRuntime` DIRECT mode) forward a Q8_0-quantized weight tree
 * without dequantising? Mirrors [`GemmaDslQuantizedTest`] for Qwen3 —
 * including `qkNorm = true` so QK-norm submodules also flow through.
 *
 * Build two parallel weight maps with identical logical values, one FP32
 * baseline and one with the 7 matmul projections (Q/K/V/O + gate/up/down)
 * swapped to `Q8MemorySegmentTensorData`. Logit divergence must stay below
 * `1e-3` — any larger gap means the upstream Q8 dispatch isn't composing
 * with the DSL forward as expected.
 *
 * Synthetic weights use small integer values + scale = 1, so Q8 round-to-
 * int8 is exact and the only source of variance is the matmul kernel
 * itself.
 */
class QwenDslQuantizedTest {

    private val ctx = DirectCpuExecutionContext()

    // Q8_0 kernels assume `inputDim % 32 == 0`. dim=32, ffnDim=64 satisfy
    // this for all projections (Q/K/V output and FFN gate/up/down).
    private val dim = 32
    private val nHeads = 2
    private val kvHeads = 2
    private val headDim = dim / nHeads
    private val ffnDim = 64
    private val vocabSize = 64
    private val seqLen = 16

    private val metadata = GgufDecoderMetadata(
        architecture = "qwen3",
        embeddingLength = dim,
        contextLength = seqLen,
        blockCount = 1,
        headCount = nHeads,
        kvHeadCount = kvHeads,
        feedForwardLength = ffnDim,
        ropeDimensionCount = headDim,
        vocabSize = vocabSize,
        ropeFreqBase = 1_000_000f,
        rmsNormEps = 1e-6f,
    )

    @Test
    fun `Q8 MemSeg projections flow through DSL Qwen and match FP32 within tolerance`() {
        Arena.ofConfined().use { arena ->
            val (fp32Weights, q8Weights) = buildWeightPair(arena)

            val fp32Model = QwenNetworkLoader.fromWeights(fp32Weights)
            val q8Model = QwenNetworkLoader.fromWeights(q8Weights)

            val fp32Runtime = OptimizedLLMRuntime(fp32Model, ctx, OptimizedLLMMode.DIRECT, FP32::class)
            val q8Runtime = OptimizedLLMRuntime(q8Model, ctx, OptimizedLLMMode.DIRECT, FP32::class)

            val tokens = intArrayOf(1, 5, 3)
            var maxDiff = 0f
            for ((step, tokenId) in tokens.withIndex()) {
                val a = fp32Runtime.forward(tokenId).data.copyToFloatArray()
                val b = q8Runtime.forward(tokenId).data.copyToFloatArray()
                for (i in a.indices) {
                    val d = abs(a[i] - b[i])
                    maxDiff = max(maxDiff, d)
                    assertTrue(
                        d < 1e-3f,
                        "step=$step logit[$i] diverged: fp32=${a[i]} q8=${b[i]} diff=$d",
                    )
                }
            }
            println("Qwen DSL Q8 probe PASSED. Max |Δlogit| across ${tokens.size} tokens: $maxDiff")
        }
    }

    /**
     * Build two `DecoderGgufWeights` sharing identical logical values.
     * The Q8 variant swaps the 7 matmul projections to packed MemSeg
     * tensors; norms, embeddings, output head, and QK-norm scales stay
     * FP32 (those aren't matmul targets, or aren't 32-block-multiple).
     */
    private fun buildWeightPair(
        arena: Arena,
    ): Pair<DecoderGgufWeights<FP32, Float>, DecoderGgufWeights<FP32, Float>> {
        // Small integer values keep Q8 round-to-int exact.
        val tokenEmb = FloatArray(vocabSize * dim) { ((it % 5) - 2).toFloat() }
        val outputW = FloatArray(vocabSize * dim) { ((it % 5) - 2).toFloat() }
        val ones = FloatArray(dim) { 1f }
        val onesHead = FloatArray(headDim) { 1f }

        val q = FloatArray(nHeads * headDim * dim) { (((it * 3) % 7) - 3).toFloat() }
        val k = FloatArray(kvHeads * headDim * dim) { (((it * 5) % 7) - 3).toFloat() }
        val v = FloatArray(kvHeads * headDim * dim) { (((it * 7) % 7) - 3).toFloat() }
        val o = FloatArray(dim * nHeads * headDim) { (((it * 11) % 7) - 3).toFloat() }
        val gate = FloatArray(ffnDim * dim) { (((it * 3) % 5) - 2).toFloat() }
        val up = FloatArray(ffnDim * dim) { (((it * 5) % 5) - 2).toFloat() }
        val down = FloatArray(dim * ffnDim) { (((it * 7) % 5) - 2).toFloat() }

        val fp32Map = linkedMapOf<String, Tensor<FP32, Float>>(
            DecoderTensorNames.TOKEN_EMBEDDINGS to fp32Tensor(Shape(vocabSize, dim), tokenEmb),
            DecoderTensorNames.OUTPUT_NORM to fp32Tensor(Shape(dim), ones),
            DecoderTensorNames.OUTPUT_WEIGHT to fp32Tensor(Shape(vocabSize, dim), outputW),
            DecoderTensorNames.attnNorm(0) to fp32Tensor(Shape(dim), ones),
            DecoderTensorNames.attnQ(0) to fp32Tensor(Shape(nHeads * headDim, dim), q),
            DecoderTensorNames.attnK(0) to fp32Tensor(Shape(kvHeads * headDim, dim), k),
            DecoderTensorNames.attnV(0) to fp32Tensor(Shape(kvHeads * headDim, dim), v),
            DecoderTensorNames.attnOut(0) to fp32Tensor(Shape(dim, nHeads * headDim), o),
            // QK-norm scales — staying FP32 (per-head RMSNorm, not a matmul target).
            // Triggers the loader's auto-detect into a qkNorm=true network.
            DecoderTensorNames.attnQNorm(0) to fp32Tensor(Shape(headDim), onesHead),
            DecoderTensorNames.attnKNorm(0) to fp32Tensor(Shape(headDim), onesHead),
            DecoderTensorNames.ffnNorm(0) to fp32Tensor(Shape(dim), ones),
            DecoderTensorNames.ffnGate(0) to fp32Tensor(Shape(ffnDim, dim), gate),
            DecoderTensorNames.ffnUp(0) to fp32Tensor(Shape(ffnDim, dim), up),
            DecoderTensorNames.ffnDown(0) to fp32Tensor(Shape(dim, ffnDim), down),
        )
        val fp32 = DecoderGgufWeights(metadata, fp32Map)

        val q8Map = linkedMapOf<String, Tensor<FP32, Float>>(
            DecoderTensorNames.TOKEN_EMBEDDINGS to fp32Tensor(Shape(vocabSize, dim), tokenEmb),
            DecoderTensorNames.OUTPUT_NORM to fp32Tensor(Shape(dim), ones),
            DecoderTensorNames.OUTPUT_WEIGHT to fp32Tensor(Shape(vocabSize, dim), outputW),
            DecoderTensorNames.attnNorm(0) to fp32Tensor(Shape(dim), ones),
            DecoderTensorNames.attnQ(0) to q8Tensor(nHeads * headDim, dim, q, arena),
            DecoderTensorNames.attnK(0) to q8Tensor(kvHeads * headDim, dim, k, arena),
            DecoderTensorNames.attnV(0) to q8Tensor(kvHeads * headDim, dim, v, arena),
            DecoderTensorNames.attnOut(0) to q8Tensor(dim, nHeads * headDim, o, arena),
            DecoderTensorNames.attnQNorm(0) to fp32Tensor(Shape(headDim), onesHead),
            DecoderTensorNames.attnKNorm(0) to fp32Tensor(Shape(headDim), onesHead),
            DecoderTensorNames.ffnNorm(0) to fp32Tensor(Shape(dim), ones),
            DecoderTensorNames.ffnGate(0) to q8Tensor(ffnDim, dim, gate, arena),
            DecoderTensorNames.ffnUp(0) to q8Tensor(ffnDim, dim, up, arena),
            DecoderTensorNames.ffnDown(0) to q8Tensor(dim, ffnDim, down, arena),
        )
        val q8 = DecoderGgufWeights(metadata, q8Map)

        return fp32 to q8
    }

    private fun fp32Tensor(shape: Shape, values: FloatArray): Tensor<FP32, Float> =
        ctx.fromFloatArray(shape, FP32::class, values)

    /** Pack a small-integer FloatArray into Q8_0 with scale = 1.0 per block. */
    private fun packQ8_0_unitScale(values: FloatArray): ByteArray {
        require(values.size % 32 == 0) {
            "packQ8_0_unitScale: values.size (${values.size}) must be a multiple of 32"
        }
        val nBlocks = values.size / 32
        val bytes = ByteArray(nBlocks * 34)
        // f16 1.0 little-endian: 0x3C00 → bytes [0x00, 0x3C].
        for (b in 0 until nBlocks) {
            val off = b * 34
            bytes[off] = 0x00
            bytes[off + 1] = 0x3C
            for (i in 0 until 32) {
                val intVal = values[b * 32 + i].toInt()
                require(intVal in -128..127) {
                    "packQ8_0_unitScale: value $intVal out of int8 range"
                }
                bytes[off + 2 + i] = intVal.toByte()
            }
        }
        return bytes
    }

    private fun q8Tensor(
        rows: Int,
        cols: Int,
        values: FloatArray,
        arena: Arena,
    ): Tensor<FP32, Float> {
        require(values.size == rows * cols) {
            "values.size=${values.size} != rows*cols=${rows * cols}"
        }
        val bytes = packQ8_0_unitScale(values)
        val segment = arena.allocate(bytes.size.toLong())
        for ((i, b) in bytes.withIndex()) {
            segment.set(ValueLayout.JAVA_BYTE, i.toLong(), b)
        }
        val data = Q8MemorySegmentTensorData(Shape(rows, cols), segment)
        @Suppress("UNCHECKED_CAST")
        return ctx.fromData(data as TensorData<FP32, Float>, FP32::class)
    }
}
