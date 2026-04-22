package sk.ainet.models.gemma

import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertTrue
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q4_KTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.FP32

/**
 * Phase 7d parity probe: the DSL path running Q4_K weights end-to-end
 * should agree, within Q4_K quantisation tolerance, with the same network
 * running on the corresponding dequantised FP32 weights.
 *
 * Validates:
 * 1. `relayoutQ4_KRowMajorToBlockMajor` correctly reshuffles GGUF
 *    row-major Q4_K bytes into the input-block-major layout expected by
 *    `JvmQuantizedVectorKernels.matmulQ4_KVec`.
 * 2. The lazy `ops.transpose(Q4_KTensorData)` in `DefaultCpuOpsJvm`
 *    composes correctly with the Q4_K matmul kernel.
 * 3. Running a tiny Gemma DSL model with Q4_K-backed weights produces
 *    logits close to the FP32 baseline (i.e., no wrong-math bug from
 *    layout or dispatch).
 *
 * The minimum Q4_K block size is 256, so the test's inner dimensions are
 * dim=256, ffnDim=256 (one block per matmul row).
 */
class GemmaDslQ4KTest {

    private val ctx = DirectCpuExecutionContext()

    private val dim = 256
    private val nHeads = 2
    private val nKvHeads = 1
    private val headDim = dim / nHeads
    private val ffnDim = 256
    private val vocabSize = 32
    private val seqLen = 8

    private val metadata = Gemma4ModelMetadata(
        architecture = "gemma4",
        embeddingLength = dim,
        contextLength = seqLen,
        blockCount = 1,
        headCount = nHeads,
        kvHeadCount = nKvHeads,
        intermediateSize = ffnDim,
        headDim = headDim,
        globalHeadDim = headDim,
        vocabSize = vocabSize,
        slidingWindow = seqLen,
        kvSharedLayers = 0,
        layerTypes = listOf("full_attention"),
        ropeParametersFull = Gemma4RopeConfig(base = 10000f, ropeType = "default", factor = 1.0f, partialRotaryFactor = 1.0f),
        ropeParametersSliding = Gemma4RopeConfig(base = 10000f, ropeType = "default"),
        maxPositionEmbeddings = seqLen
    )

    /** f16 bit pattern for 1.0 (little-endian: 0x00, 0x3C). */
    private val F16_ONE: Pair<Byte, Byte> = 0x00.toByte() to 0x3C.toByte()

    /** f16 bit pattern for 0.0 (little-endian: 0x00, 0x00). */
    private val F16_ZERO: Pair<Byte, Byte> = 0x00.toByte() to 0x00.toByte()

    /**
     * Pack 8 sub-block (scaleIdx, minIdx) pairs into 12 bytes of Q4_K's
     * scale/min packed region. Each sub-block gets 12 bits: low 6 for
     * scale index, next 6 for min index.
     */
    private fun packScaleMins(scaleIdx: Int, minIdx: Int): ByteArray {
        val packed = LongArray(2) // 8 subblocks × 12 bits = 96 bits; use two longs
        val sm = ((minIdx and 0x3F).toLong() shl 6) or (scaleIdx and 0x3F).toLong()
        var bits = 0L
        for (sb in 0 until 8) {
            bits = bits or (sm shl (sb * 12))
        }
        // lower 64 bits → first 8 bytes, upper 32 bits (of a logical 96-bit) → last 4 bytes
        val high = bits.shl(0).shr(64) // Kotlin Long is 64 bits; need another 32 bits for bits 64..95
        // Redo using two Long fields covering the bit range:
        // easier: compute byte-by-byte
        val out = ByteArray(12)
        for (sb in 0 until 8) {
            val bitPos = sb * 12
            val bytePos = bitPos / 8
            val bitShift = bitPos % 8
            val value = ((minIdx and 0x3F) shl 6) or (scaleIdx and 0x3F) // 12 bits
            val raw = value.toLong() shl bitShift // up to 12 + 7 = 19 bits, fits in Long
            // Spread across up to 3 bytes
            for (b in 0 until 3) {
                val dstIdx = bytePos + b
                if (dstIdx >= out.size) break
                val shifted = (raw ushr (b * 8)).toInt() and 0xFF
                out[dstIdx] = (out[dstIdx].toInt() or shifted).toByte()
            }
        }
        return out
    }

    /**
     * Build one Q4_K block (144 bytes) with d=1, dMin=0, all sub-block
     * scales = 63/63 = 1.0, all mins = 0, codes provided by the caller.
     * Dequantisation: output[i] = codes[i].
     */
    private fun q4kBlock(codes: IntArray): ByteArray {
        require(codes.size == 256) { "Q4_K block needs 256 codes, got ${codes.size}" }
        val block = ByteArray(144)
        block[0] = F16_ONE.first
        block[1] = F16_ONE.second
        block[2] = F16_ZERO.first
        block[3] = F16_ZERO.second
        val scaleMinBytes = packScaleMins(scaleIdx = 63, minIdx = 0)
        for (i in 0 until 12) block[4 + i] = scaleMinBytes[i]
        for (i in 0 until 128) {
            val lo = codes[i * 2] and 0x0F
            val hi = codes[i * 2 + 1] and 0x0F
            block[16 + i] = ((hi shl 4) or lo).toByte()
        }
        return block
    }

    /**
     * Build a Q4_K weight of shape `[outDim, inDim]` whose dequantised
     * values are integer-valued `[0..15]` per the provided function. Returns
     * both the Q4_K bytes (in GGUF row-major block order) and the
     * equivalent FP32 weight array.
     */
    private fun buildQ4KWeight(
        outDim: Int,
        inDim: Int,
        value: (row: Int, col: Int) -> Int
    ): Pair<ByteArray, FloatArray> {
        require(inDim % 256 == 0) { "inDim ($inDim) must be a multiple of 256 for Q4_K" }
        val blocksPerRow = inDim / 256
        val bytes = ByteArray(outDim * blocksPerRow * 144)
        val fp32 = FloatArray(outDim * inDim)
        for (r in 0 until outDim) {
            for (b in 0 until blocksPerRow) {
                val codes = IntArray(256) { i ->
                    val col = b * 256 + i
                    val v = value(r, col) and 0x0F
                    fp32[r * inDim + col] = v.toFloat()
                    v
                }
                val blockBytes = q4kBlock(codes)
                val dstOff = (r * blocksPerRow + b) * 144
                System.arraycopy(blockBytes, 0, bytes, dstOff, 144)
            }
        }
        return bytes to fp32
    }

    private fun fp32Tensor(shape: Shape, values: FloatArray): Tensor<FP32, Float> =
        ctx.fromFloatArray(shape, FP32::class, values)

    private fun ones(shape: Shape): Tensor<FP32, Float> =
        fp32Tensor(shape, FloatArray(shape.volume) { 1.0f })

    @Suppress("UNCHECKED_CAST")
    private fun q4kTensor(rows: Int, cols: Int, bytes: ByteArray): Tensor<FP32, Float> {
        val relaid = relayoutQ4_KRowMajorToBlockMajor(bytes, Shape(rows, cols))
        val data = Q4_KBlockTensorData.fromRawBytes(Shape(rows, cols), relaid)
        return ctx.fromData(data as TensorData<FP32, Float>, FP32::class)
    }

    @Test
    fun `Q4_K weights run through the DSL and match their dequantised FP32 equivalent`() {
        // Every projection weight is Q4_K with dequantised values in [0..15].
        val tokenEmb = FloatArray(vocabSize * dim) { ((it % 7) - 3).toFloat() }
        val finalNormVals = FloatArray(dim) { 1f }
        val lmHeadVals = FloatArray(vocabSize * dim) { ((it % 7) - 3).toFloat() }
        val attnNormVals = FloatArray(dim) { 1f }
        val postNormVals = FloatArray(dim) { 1f }

        // Shape [out, in] = [qDim, dim] = [256, 256]
        val (qBytes, qFp32) = buildQ4KWeight(nHeads * headDim, dim) { r, c -> (r + c) % 16 }
        // Shape [kvDim, dim] = [128, 256]
        val (kBytes, kFp32) = buildQ4KWeight(nKvHeads * headDim, dim) { r, c -> (r * 3 + c) % 16 }
        val (vBytes, vFp32) = buildQ4KWeight(nKvHeads * headDim, dim) { r, c -> (r * 5 + c) % 16 }
        // Shape [dim, qDim]
        val (oBytes, oFp32) = buildQ4KWeight(dim, nHeads * headDim) { r, c -> (r * 7 + c) % 16 }
        // Shape [ffnDim, dim]
        val (gateBytes, gateFp32) = buildQ4KWeight(ffnDim, dim) { r, c -> (r + c * 2) % 16 }
        val (upBytes, upFp32) = buildQ4KWeight(ffnDim, dim) { r, c -> (r * 2 + c) % 16 }
        // Shape [dim, ffnDim]
        val (downBytes, downFp32) = buildQ4KWeight(dim, ffnDim) { r, c -> (r + c * 3) % 16 }

        fun fp32Weights() = linkedMapOf<String, Tensor<FP32, Float>>(
            Gemma4TensorNames.TOKEN_EMBEDDINGS to fp32Tensor(Shape(vocabSize, dim), tokenEmb),
            Gemma4TensorNames.OUTPUT_NORM to fp32Tensor(Shape(dim), finalNormVals),
            Gemma4TensorNames.OUTPUT_WEIGHT to fp32Tensor(Shape(vocabSize, dim), lmHeadVals),
            Gemma4TensorNames.inputLayernorm(0) to fp32Tensor(Shape(dim), attnNormVals),
            Gemma4TensorNames.attnQ(0) to fp32Tensor(Shape(nHeads * headDim, dim), qFp32),
            Gemma4TensorNames.attnK(0) to fp32Tensor(Shape(nKvHeads * headDim, dim), kFp32),
            Gemma4TensorNames.attnV(0) to fp32Tensor(Shape(nKvHeads * headDim, dim), vFp32),
            Gemma4TensorNames.attnOut(0) to fp32Tensor(Shape(dim, nHeads * headDim), oFp32),
            Gemma4TensorNames.postAttentionLayernorm(0) to fp32Tensor(Shape(dim), postNormVals),
            Gemma4TensorNames.ffnGate(0) to fp32Tensor(Shape(ffnDim, dim), gateFp32),
            Gemma4TensorNames.ffnUp(0) to fp32Tensor(Shape(ffnDim, dim), upFp32),
            Gemma4TensorNames.ffnDown(0) to fp32Tensor(Shape(dim, ffnDim), downFp32),
        )

        val q4k = linkedMapOf<String, Tensor<FP32, Float>>(
            Gemma4TensorNames.TOKEN_EMBEDDINGS to fp32Tensor(Shape(vocabSize, dim), tokenEmb),
            Gemma4TensorNames.OUTPUT_NORM to fp32Tensor(Shape(dim), finalNormVals),
            Gemma4TensorNames.OUTPUT_WEIGHT to fp32Tensor(Shape(vocabSize, dim), lmHeadVals),
            Gemma4TensorNames.inputLayernorm(0) to fp32Tensor(Shape(dim), attnNormVals),
            Gemma4TensorNames.attnQ(0) to q4kTensor(nHeads * headDim, dim, qBytes),
            Gemma4TensorNames.attnK(0) to q4kTensor(nKvHeads * headDim, dim, kBytes),
            Gemma4TensorNames.attnV(0) to q4kTensor(nKvHeads * headDim, dim, vBytes),
            Gemma4TensorNames.attnOut(0) to q4kTensor(dim, nHeads * headDim, oBytes),
            Gemma4TensorNames.postAttentionLayernorm(0) to fp32Tensor(Shape(dim), postNormVals),
            Gemma4TensorNames.ffnGate(0) to q4kTensor(ffnDim, dim, gateBytes),
            Gemma4TensorNames.ffnUp(0) to q4kTensor(ffnDim, dim, upBytes),
            Gemma4TensorNames.ffnDown(0) to q4kTensor(dim, ffnDim, downBytes),
        )

        val fp32Model = GemmaNetworkLoader.fromWeights(ctx, Gemma4Weights(metadata, fp32Weights()))
        val q4kModel = GemmaNetworkLoader.fromWeights(ctx, Gemma4Weights(metadata, q4k))

        val fp32Runtime = OptimizedLLMRuntime(fp32Model, ctx, OptimizedLLMMode.DIRECT, FP32::class)
        val q4kRuntime = OptimizedLLMRuntime(q4kModel, ctx, OptimizedLLMMode.DIRECT, FP32::class)

        val tokens = intArrayOf(1, 5, 3)
        var maxDiff = 0f
        for ((step, tokenId) in tokens.withIndex()) {
            val a = fp32Runtime.forward(tokenId).data.copyToFloatArray()
            val b = q4kRuntime.forward(tokenId).data.copyToFloatArray()
            for (i in a.indices) {
                val d = abs(a[i] - b[i])
                maxDiff = max(maxDiff, d)
                // Tolerance: the Q4_K path and the FP32 path compute the exact same
                // math on the same dequantised values, so divergence can only come
                // from floating-point accumulation order. Stay tight — 1e-2 is more
                // than enough.
                assertTrue(
                    d < 1e-2f,
                    "step=$step logit[$i] diverged: fp32=${a[i]} q4k=${b[i]} diff=$d"
                )
            }
        }
        println("Q4_K DSL probe PASSED. Max |Δlogit| across ${tokens.size} tokens: $maxDiff")
    }
}
