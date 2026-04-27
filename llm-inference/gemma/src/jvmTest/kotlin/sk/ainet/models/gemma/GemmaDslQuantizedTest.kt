package sk.ainet.models.gemma

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
import sk.ainet.lang.types.FP32

/**
 * Phase 7b empirical probe: can the DSL path (`gemmaNetwork()` +
 * `OptimizedLLMRuntime` DIRECT mode) forward a Q8_0-quantized weight tree
 * without dequantising to FP32?
 *
 * The CPU backend (`DefaultCpuOpsJvm`) advertises a **lazy transpose** for
 * `Q8MemorySegmentMarker` and a direct `matmul(fp32, Q8_MemSeg)` dispatch.
 * If both dispatches compose correctly, running a DSL module tree whose
 * weights are `Q8MemorySegmentTensorData` should produce the same output
 * as the FP32 original, modulo quantization error. This test pins that
 * behaviour.
 *
 * Scope:
 * - **Q8_0 only.** Q4_K / Q6_K dispatch is a separate kernel chain and
 *   goes through a `Q4_KTensorData` type (not MemSeg) — out of scope for
 *   this probe.
 * - **Synthetic weights with integer values**, so Q8 round-to-int-8 is
 *   exact and the only variance in output comes from the matmul kernels
 *   themselves, not the quantization step.
 */
class GemmaDslQuantizedTest {

    // Default DenseTensorDataFactory → input/activation tensors land in
    // FloatArray-backed storage. Q8 weight tensors are explicitly MemSeg via
    // Q8MemorySegmentTensorData below. The mix routes through the
    // `matmul(FloatArray, Q8_MemSeg)` kernel, not the all-MemSeg fast path
    // (which is SIMD-32-wide and would choke on dim=8 test inputs).
    private val ctx = DirectCpuExecutionContext()

    // Q8_0 matmul kernels assume `inputDim % 32 == 0` (SIMD lane width + block size),
    // so every matmul in the forward pass needs an inputDim ≥ 32 and multiple of 32.
    // dim=32, ffnDim=64 satisfy this for all projections (Q/K/V, O, gate/up/down).
    private val dim = 32
    private val nHeads = 2
    private val nKvHeads = 1
    private val headDim = dim / nHeads
    private val ffnDim = 64
    private val vocabSize = 64
    private val seqLen = 16

    /** f16 bit pattern for 1.0: sign=0, exp=15+15=0x0F+?, mant=0. Simply 0x3C00. */
    private val F16_ONE_LE: ByteArray = byteArrayOf(0x00, 0x3C)

    /**
     * Pack a float array whose values are small whole numbers into Q8_0
     * bytes using scale = 1.0 per block. Dequantised result exactly equals
     * round(values) — the test relies on this to eliminate quant error as
     * a variable.
     */
    private fun packQ8_0_unitScale(values: FloatArray): ByteArray {
        require(values.size % 32 == 0) {
            "packQ8_0_unitScale: values.size (${values.size}) must be a multiple of 32"
        }
        val nBlocks = values.size / 32
        val bytes = ByteArray(nBlocks * 34)
        for (b in 0 until nBlocks) {
            val off = b * 34
            F16_ONE_LE.copyInto(bytes, off)
            for (i in 0 until 32) {
                val v = values[b * 32 + i].toInt()
                require(v in -128..127) { "packQ8_0_unitScale: value $v out of int8 range" }
                bytes[off + 2 + i] = v.toByte()
            }
        }
        return bytes
    }

    /**
     * Build a Q8_0 MemorySegment tensor of shape `[rows, cols]` with the
     * provided integer-valued FP32 payload. The caller owns the [arena].
     */
    private fun q8Tensor(
        rows: Int,
        cols: Int,
        values: FloatArray,
        arena: Arena
    ): Tensor<FP32, Float> {
        require(values.size == rows * cols) { "values.size=${values.size} != rows*cols=${rows * cols}" }
        val bytes = packQ8_0_unitScale(values)
        val segment = arena.allocate(bytes.size.toLong())
        for ((i, b) in bytes.withIndex()) {
            segment.set(ValueLayout.JAVA_BYTE, i.toLong(), b)
        }
        val data = Q8MemorySegmentTensorData(Shape(rows, cols), segment)
        @Suppress("UNCHECKED_CAST")
        return ctx.fromData(data as sk.ainet.lang.tensor.data.TensorData<FP32, Float>, FP32::class)
    }

    private fun fp32Tensor(shape: Shape, values: FloatArray): Tensor<FP32, Float> =
        ctx.fromFloatArray(shape, FP32::class, values)

    private fun ones(shape: Shape): Tensor<FP32, Float> =
        fp32Tensor(shape, FloatArray(shape.volume) { 1.0f })

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

    /**
     * Produce two parallel weight maps (FP32 baseline and Q8_0-for-projections)
     * that share identical logical values, so any divergence in output comes
     * from the matmul dispatch, not weight initialisation drift.
     */
    private fun buildWeights(arena: Arena): Pair<Gemma4Weights<FP32, Float>, Gemma4Weights<FP32, Float>> {
        // Small integer weights → Q8 round-to-int exact with scale=1.
        val tokenEmb = FloatArray(vocabSize * dim) { ((it % 5) - 2).toFloat() }
        val finalNormVals = FloatArray(dim) { 1f }
        val lmHeadVals = FloatArray(vocabSize * dim) { ((it % 5) - 2).toFloat() }

        val attnNormVals = FloatArray(dim) { 1f }
        val qVals = FloatArray(nHeads * headDim * dim) { (((it * 3) % 7) - 3).toFloat() }
        val kVals = FloatArray(nKvHeads * headDim * dim) { (((it * 5) % 7) - 3).toFloat() }
        val vVals = FloatArray(nKvHeads * headDim * dim) { (((it * 7) % 7) - 3).toFloat() }
        val oVals = FloatArray(dim * nHeads * headDim) { (((it * 11) % 7) - 3).toFloat() }
        val postNormVals = FloatArray(dim) { 1f }
        val gateVals = FloatArray(ffnDim * dim) { (((it * 3) % 5) - 2).toFloat() }
        val upVals = FloatArray(ffnDim * dim) { (((it * 5) % 5) - 2).toFloat() }
        val downVals = FloatArray(dim * ffnDim) { (((it * 7) % 5) - 2).toFloat() }

        fun fp32Map() = linkedMapOf<String, Tensor<FP32, Float>>(
            Gemma4TensorNames.TOKEN_EMBEDDINGS to fp32Tensor(Shape(vocabSize, dim), tokenEmb),
            Gemma4TensorNames.OUTPUT_NORM to fp32Tensor(Shape(dim), finalNormVals),
            Gemma4TensorNames.OUTPUT_WEIGHT to fp32Tensor(Shape(vocabSize, dim), lmHeadVals),
            Gemma4TensorNames.inputLayernorm(0) to fp32Tensor(Shape(dim), attnNormVals),
            Gemma4TensorNames.attnQ(0) to fp32Tensor(Shape(nHeads * headDim, dim), qVals),
            Gemma4TensorNames.attnK(0) to fp32Tensor(Shape(nKvHeads * headDim, dim), kVals),
            Gemma4TensorNames.attnV(0) to fp32Tensor(Shape(nKvHeads * headDim, dim), vVals),
            Gemma4TensorNames.attnOut(0) to fp32Tensor(Shape(dim, nHeads * headDim), oVals),
            Gemma4TensorNames.postAttentionLayernorm(0) to fp32Tensor(Shape(dim), postNormVals),
            Gemma4TensorNames.ffnGate(0) to fp32Tensor(Shape(ffnDim, dim), gateVals),
            Gemma4TensorNames.ffnUp(0) to fp32Tensor(Shape(ffnDim, dim), upVals),
            Gemma4TensorNames.ffnDown(0) to fp32Tensor(Shape(dim, ffnDim), downVals),
        )

        val fp32 = Gemma4Weights(metadata, fp32Map())

        // Quantized map: reuse FP32 for norms, embeddings, and lm_head (all
        // non-matmul or sub-32-block). Swap Q/K/V/O/Gate/Up/Down for Q8 MemSeg
        // — these all have volume divisible by 32 given dim=8, ffnDim=32.
        val q8 = linkedMapOf<String, Tensor<FP32, Float>>(
            Gemma4TensorNames.TOKEN_EMBEDDINGS to fp32Tensor(Shape(vocabSize, dim), tokenEmb),
            Gemma4TensorNames.OUTPUT_NORM to fp32Tensor(Shape(dim), finalNormVals),
            Gemma4TensorNames.OUTPUT_WEIGHT to fp32Tensor(Shape(vocabSize, dim), lmHeadVals),
            Gemma4TensorNames.inputLayernorm(0) to fp32Tensor(Shape(dim), attnNormVals),
            Gemma4TensorNames.attnQ(0) to q8Tensor(nHeads * headDim, dim, qVals, arena),
            Gemma4TensorNames.attnK(0) to q8Tensor(nKvHeads * headDim, dim, kVals, arena),
            Gemma4TensorNames.attnV(0) to q8Tensor(nKvHeads * headDim, dim, vVals, arena),
            Gemma4TensorNames.attnOut(0) to q8Tensor(dim, nHeads * headDim, oVals, arena),
            Gemma4TensorNames.postAttentionLayernorm(0) to fp32Tensor(Shape(dim), postNormVals),
            Gemma4TensorNames.ffnGate(0) to q8Tensor(ffnDim, dim, gateVals, arena),
            Gemma4TensorNames.ffnUp(0) to q8Tensor(ffnDim, dim, upVals, arena),
            Gemma4TensorNames.ffnDown(0) to q8Tensor(dim, ffnDim, downVals, arena),
        )

        return fp32 to Gemma4Weights(metadata, q8)
    }

    @Test
    fun `Q8 MemSeg weights flow through the DSL path and match FP32 within tolerance`() {
        val arena = Arena.ofConfined()
        try {
            val (fp32Weights, q8Weights) = buildWeights(arena)

            val fp32Model = GemmaNetworkLoader.fromWeights(ctx, fp32Weights)
            val q8Model = GemmaNetworkLoader.fromWeights(ctx, q8Weights)

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
                        "step=$step logit[$i] diverged: fp32=${a[i]} q8=${b[i]} diff=$d"
                    )
                }
            }
            println("Q8-DSL probe PASSED. Max |Δlogit| across ${tokens.size} tokens: $maxDiff")
        } finally {
            arena.close()
        }
    }
}
