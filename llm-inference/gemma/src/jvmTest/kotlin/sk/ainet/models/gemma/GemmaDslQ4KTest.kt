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
 * 1. Canonical (GGUF row-major) Q4_K bytes under an `[out, in]` shape flow
 *    through the engine's physical packed `ops.transpose` (>= 0.40.1,
 *    SKaiNET#968) into the input-block-major layout the Q4_K matmul kernel
 *    expects.
 * 2. That transpose composes correctly with the Q4_K matmul kernel dispatch.
 * 3. Running a tiny Gemma DSL model with Q4_K-backed weights produces
 *    logits close to the FP32 baseline (i.e., no wrong-math bug from
 *    layout or dispatch).
 *
 * The minimum Q4_K block size is 256, so the test's inner dimensions are
 * dim=256, ffnDim=256 (one block per matmul row).
 */
class GemmaDslQ4KTest {
    // These are DSL-plumbing parity tests: both legs must compute the SAME math, which was
    // true while dispatch served the decoding reference kernel. Since the 0.52.0 self-healing
    // dispatch, the first matmul auto-discovers the platform's native quant kernels — on x86
    // the ggml-faithful int8 Q4_K path (SKaiNET#944) — and the "exact same math" premise
    // breaks by more than the tight tolerance. Pin the reference kernels explicitly: a
    // non-empty dispatch table suppresses auto-install (the engine's documented contract).
    @kotlin.test.BeforeTest
    fun pinReferenceKernels() {
        sk.ainet.backend.api.kernel.KernelDispatch.clearForTesting()
        sk.ainet.backend.api.kernel.KernelPacks.installReference()
    }

    @kotlin.test.AfterTest
    fun unpinKernels() = sk.ainet.backend.api.kernel.KernelDispatch.clearForTesting()


    private val ctx = DirectCpuExecutionContext()

    // dim and ffnDim chosen so each Q4_K weight has multiple input blocks per
    // row — `inDim = 512 → blocksPerRow = 2`. With one block per row the
    // canonical and input-block-major orders coincide and the test silently
    // misses any block-order bug. Two blocks per row exercises the real ggml
    // strided codes layout end-to-end.
    private val dim = 512
    private val nHeads = 2
    private val nKvHeads = 1
    private val headDim = dim / nHeads
    private val ffnDim = 512
    private val vocabSize = 32
    private val seqLen = 8

    private val metadata = GemmaModelMetadata(
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
        ropeParametersFull = GemmaRopeConfig(base = 10000f, ropeType = "default", factor = 1.0f, partialRotaryFactor = 1.0f),
        ropeParametersSliding = GemmaRopeConfig(base = 10000f, ropeType = "default"),
        maxPositionEmbeddings = seqLen
    )

    /** f16 bit pattern for 1.0 (little-endian: 0x00, 0x3C). */
    private val F16_ONE: Pair<Byte, Byte> = 0x00.toByte() to 0x3C.toByte()

    /** f16 bit pattern for 0.0 (little-endian: 0x00, 0x00). */
    private val F16_ZERO: Pair<Byte, Byte> = 0x00.toByte() to 0x00.toByte()

    /**
     * Pack 8 sub-block (scaleIdx, minIdx) pairs into 12 bytes using ggml's
     * `get_scale_min_k4` layout — *not* a flat 12-bits-per-sub-block packing.
     * For sub-blocks 0..3:
     *   `scales[j]   = scaleIdx & 0x3F` plus top-2-bits of `(scaleIdx_{j+4} >> 4)`
     *   `scales[j+4] = minIdx   & 0x3F` plus top-2-bits of `(minIdx_{j+4}   >> 4)`
     * For sub-blocks 4..7:
     *   `scales[j+4] low 4 bits = scaleIdx & 0x0F`; high 4 bits = `minIdx & 0x0F`.
     *
     * This helper assumes a uniform `(scaleIdx, minIdx)` across all 8
     * sub-blocks (the only mode the test currently exercises) and rejects
     * inputs whose top 2 bits would conflict between sub-block groups.
     */
    private fun packCanonicalScaleMins(scaleIdx: Int, minIdx: Int): ByteArray {
        require(scaleIdx in 0..0x3F && minIdx in 0..0x3F) {
            "scaleIdx/minIdx must fit in 6 bits, got scaleIdx=$scaleIdx minIdx=$minIdx"
        }
        val out = ByteArray(12)
        val scaleHigh2 = (scaleIdx ushr 4) and 0x03
        val minHigh2 = (minIdx ushr 4) and 0x03
        // Sub-blocks 0..3: low 6 bits in bytes [j] / [j+4]; bits 6..7 carry the
        // top-2-bits of sub-blocks 4..7's scaleIdx/minIdx.
        for (j in 0 until 4) {
            out[j] = ((scaleIdx and 0x3F) or (scaleHigh2 shl 6)).toByte()
            out[j + 4] = ((minIdx and 0x3F) or (minHigh2 shl 6)).toByte()
        }
        // Sub-blocks 4..7 (j = 4..7): byte [j+4] holds (minIdx & 0x0F) in
        // high 4 bits and (scaleIdx & 0x0F) in low 4 bits.
        for (j in 4 until 8) {
            out[j + 4] = (((minIdx and 0x0F) shl 4) or (scaleIdx and 0x0F)).toByte()
        }
        return out
    }

    /**
     * Build one Q4_K block (144 bytes) with `d=1`, `dMin=0`, scaleIdx=1 across
     * all 8 sub-blocks (so per-element scale `= d * sc = 1.0`) and codes
     * provided by the caller. Codes are laid out in the canonical *strided*
     * ggml order: in each 32-byte qs group `j`, byte `j*32 + i` holds the
     * code at position `i` of sub-block `2j` in its lo nibble, and the code at
     * position `i` of sub-block `2j+1` in its hi nibble.
     */
    private fun q4kBlock(codes: IntArray): ByteArray {
        require(codes.size == 256) { "Q4_K block needs 256 codes, got ${codes.size}" }
        val block = ByteArray(144)
        block[0] = F16_ONE.first
        block[1] = F16_ONE.second
        block[2] = F16_ZERO.first
        block[3] = F16_ZERO.second
        val scaleMinBytes = packCanonicalScaleMins(scaleIdx = 1, minIdx = 0)
        for (i in 0 until 12) block[4 + i] = scaleMinBytes[i]
        // Strided codes: 4 groups of 32 bytes each; group `j` carries
        // sub-blocks (2j, 2j+1).
        for (j in 0 until 4) {
            for (i in 0 until 32) {
                val lo = codes[2 * j * 32 + i] and 0x0F
                val hi = codes[(2 * j + 1) * 32 + i] and 0x0F
                block[16 + j * 32 + i] = ((hi shl 4) or lo).toByte()
            }
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
        // Canonical row-major bytes verbatim: the classic [out, in] path defers
        // the block-grid permutation to the engine's physical packed
        // ops.transpose (>= 0.40.1) inside linearProject. Relayouting here
        // (as before the SKaiNET#968 layout-contract fix landed in 0.40.1)
        // would get the blocks permuted a second time.
        val data = Q4_KBlockTensorData.fromRawBytes(Shape(rows, cols), bytes)
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
            GemmaTensorNames.TOKEN_EMBEDDINGS to fp32Tensor(Shape(vocabSize, dim), tokenEmb),
            GemmaTensorNames.OUTPUT_NORM to fp32Tensor(Shape(dim), finalNormVals),
            GemmaTensorNames.OUTPUT_WEIGHT to fp32Tensor(Shape(vocabSize, dim), lmHeadVals),
            GemmaTensorNames.inputLayernorm(0) to fp32Tensor(Shape(dim), attnNormVals),
            GemmaTensorNames.attnQ(0) to fp32Tensor(Shape(nHeads * headDim, dim), qFp32),
            GemmaTensorNames.attnK(0) to fp32Tensor(Shape(nKvHeads * headDim, dim), kFp32),
            GemmaTensorNames.attnV(0) to fp32Tensor(Shape(nKvHeads * headDim, dim), vFp32),
            GemmaTensorNames.attnOut(0) to fp32Tensor(Shape(dim, nHeads * headDim), oFp32),
            GemmaTensorNames.postAttentionLayernorm(0) to fp32Tensor(Shape(dim), postNormVals),
            GemmaTensorNames.ffnGate(0) to fp32Tensor(Shape(ffnDim, dim), gateFp32),
            GemmaTensorNames.ffnUp(0) to fp32Tensor(Shape(ffnDim, dim), upFp32),
            GemmaTensorNames.ffnDown(0) to fp32Tensor(Shape(dim, ffnDim), downFp32),
        )

        val q4k = linkedMapOf<String, Tensor<FP32, Float>>(
            GemmaTensorNames.TOKEN_EMBEDDINGS to fp32Tensor(Shape(vocabSize, dim), tokenEmb),
            GemmaTensorNames.OUTPUT_NORM to fp32Tensor(Shape(dim), finalNormVals),
            GemmaTensorNames.OUTPUT_WEIGHT to fp32Tensor(Shape(vocabSize, dim), lmHeadVals),
            GemmaTensorNames.inputLayernorm(0) to fp32Tensor(Shape(dim), attnNormVals),
            GemmaTensorNames.attnQ(0) to q4kTensor(nHeads * headDim, dim, qBytes),
            GemmaTensorNames.attnK(0) to q4kTensor(nKvHeads * headDim, dim, kBytes),
            GemmaTensorNames.attnV(0) to q4kTensor(nKvHeads * headDim, dim, vBytes),
            GemmaTensorNames.attnOut(0) to q4kTensor(dim, nHeads * headDim, oBytes),
            GemmaTensorNames.postAttentionLayernorm(0) to fp32Tensor(Shape(dim), postNormVals),
            GemmaTensorNames.ffnGate(0) to q4kTensor(ffnDim, dim, gateBytes),
            GemmaTensorNames.ffnUp(0) to q4kTensor(ffnDim, dim, upBytes),
            GemmaTensorNames.ffnDown(0) to q4kTensor(dim, ffnDim, downBytes),
        )

        val fp32Model = GemmaNetworkLoader.fromWeights(ctx, GemmaWeights(metadata, fp32Weights()))
        val q4kModel = GemmaNetworkLoader.fromWeights(ctx, GemmaWeights(metadata, q4k))

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
