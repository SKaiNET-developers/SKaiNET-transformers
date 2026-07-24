package sk.ainet.lang.nn.transformer

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.KspTensorOps
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin

/**
 * RoPE pairing strategy for how dimension pairs are laid out in the head vector.
 */
public enum class RoPEMode {
    /**
     * Split-half: first half and second half form pairs.
     * Pairs: (x[0], x[halfDim]), (x[1], x[halfDim+1]), ...
     * Used by some implementations (e.g., HuggingFace GPT-NeoX style).
     */
    SPLIT_HALF,

    /**
     * Interleaved: adjacent elements form pairs.
     * Pairs: (x[0], x[1]), (x[2], x[3]), ...
     * Used by LLaMA, Mistral, and most GGUF models.
     */
    INTERLEAVED
}

/**
 * RoPE frequency-scaling strategy. For long-context models trained with a
 * shorter `originalMaxPosEmb` and extrapolated to a larger [maxSeqLen].
 */
public enum class RoPEScaling {
    /** No scaling. Frequencies use the raw [RoPE.base]. */
    NONE,

    /**
     * Uniform frequency scaling: `inv_freq /= factor` on top of the standard
     * per-index inv_freq — matches Gemma 4's "proportional" RoPE reference in
     * HF transformers (`_compute_proportional_rope_parameters`). At `factor=1`
     * this reduces to [NONE]. Gemma 4 global-attention layers use this.
     */
    PROPORTIONAL
}

/**
 * Rotary Position Embedding (RoPE) module.
 *
 * Precomputes cos/sin frequency tables and applies rotary embeddings to input tensors.
 * Used by Llama, Apertus, Gemma, and other decoder architectures.
 *
 * Input shape: [seqLen, dim] or [nHeads, seqLen, dim]
 * The last dimension is split into pairs for rotation.
 *
 * @param headDim dimension of each attention head (must be even)
 * @param maxSeqLen maximum sequence length for precomputed tables
 * @param base RoPE base frequency (default 10000.0; Gemma 4 uses 1e6 for global layers)
 * @param mode pairing strategy: [RoPEMode.INTERLEAVED] (LLaMA default) or [RoPEMode.SPLIT_HALF]
 * @param scaling frequency-scaling strategy; defaults to [RoPEScaling.NONE]
 * @param scalingFactor scaling factor for [RoPEScaling.PROPORTIONAL] (1.0 = identity; only
 *   values > 1 are meaningful for position extrapolation). Ignored when scaling is NONE.
 * @param partialRotaryFactor fraction of [headDim] that receives rotation (rest passes through
 *   unchanged). Defaults to 1.0 = rotate the full head. Must produce an even rotary dim.
 *   Gemma 4 global-attention layers use 0.5.
 * @param name module name
 */
public class RoPE<T : DType, V>(
    public val headDim: Int,
    public val maxSeqLen: Int,
    private val base: Float = 10000.0f,
    public val mode: RoPEMode = RoPEMode.INTERLEAVED,
    public val scaling: RoPEScaling = RoPEScaling.NONE,
    public val scalingFactor: Float = 1.0f,
    public val partialRotaryFactor: Float = 1.0f,
    // inv_freq denominator convention. HF "proportional" rope uses headDim even under partial
    // rotary (default, false). The classic/original partial rope (e.g. Moonshine) uses the
    // rotated width itself, rotaryDim — set true. Verified against Moonshine's ONNX (freqs match
    // base^(-2i/32) with rotaryDim=32, not base^(-2i/36)).
    public val freqDenomRotaryDim: Boolean = false,
    override val name: String = "RoPE"
) : Module<T, V>() {

    /** Number of head dims that actually get rotated. The trailing `headDim - rotaryDim` pass through unchanged. */
    public val rotaryDim: Int = run {
        val raw = (headDim * partialRotaryFactor).toInt()
        // Round down to even: each rotation consumes a pair.
        raw - (raw % 2)
    }

    init {
        require(headDim % 2 == 0) { "RoPE headDim must be even, got $headDim" }
        require(partialRotaryFactor > 0f && partialRotaryFactor <= 1f) {
            "RoPE partialRotaryFactor must be in (0, 1], got $partialRotaryFactor"
        }
        require(rotaryDim >= 2) {
            "RoPE rotaryDim (headDim=$headDim × partialRotaryFactor=$partialRotaryFactor) must be at least 2"
        }
        require(scalingFactor > 0f) { "RoPE scalingFactor must be > 0, got $scalingFactor" }
    }

    override val modules: List<Module<T, V>> = emptyList()

    // Precomputed frequency tables: [maxSeqLen, rotaryDim/2]
    private val halfRotary: Int = rotaryDim / 2
    private val cosTable: FloatArray = FloatArray(maxSeqLen * halfRotary)
    private val sinTable: FloatArray = FloatArray(maxSeqLen * halfRotary)

    init {
        // Reference formula (transformers 5.6.0 _compute_proportional_rope_parameters):
        //   inv_freq[i] = 1 / base ^ (2i / head_dim)        for i in 0 until halfRotary
        //   inv_freq /= factor                              (iff PROPORTIONAL)
        //
        // Note the exponent denominator is headDim, NOT rotaryDim — matters when
        // partialRotaryFactor < 1. Positions beyond rotaryDim pass through
        // unchanged (handled by the rotate-then-concat/slice in applyRoPE*).
        val applyFactor = scaling == RoPEScaling.PROPORTIONAL && scalingFactor != 1.0f
        val freqDenom = if (freqDenomRotaryDim) rotaryDim else headDim
        for (pos in 0 until maxSeqLen) {
            for (i in 0 until halfRotary) {
                var freq = 1.0f / base.pow(2.0f * i / freqDenom)
                if (applyFactor) freq /= scalingFactor
                val angle = pos * freq
                cosTable[pos * halfRotary + i] = cos(angle)
                sinTable[pos * halfRotary + i] = sin(angle)
            }
        }
    }

    /**
     * Apply rotary embeddings to [input] starting at [position].
     *
     * @param input tensor to rotate, last dim = headDim
     * @param position the starting position index (for autoregressive decoding)
     * @param ctx execution context
     * @return rotated tensor with same shape
     */
    public fun forward(input: Tensor<T, V>, position: Int, ctx: ExecutionContext): Tensor<T, V> {
        return sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, input) {
            applyRoPE(input, position, ctx)
        }
    }

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        // Default forward with position=0 (for tracing / non-autoregressive use)
        return applyRoPE(input, 0, ctx)
    }

    private fun applyRoPE(input: Tensor<T, V>, position: Int, ctx: ExecutionContext): Tensor<T, V> {
        return when (mode) {
            RoPEMode.SPLIT_HALF -> applyRoPESplitHalf(input, position, ctx)
            RoPEMode.INTERLEAVED -> applyRoPEInterleaved(input, position, ctx)
        }
    }

    /**
     * Split-half mode: pair `dim_i` with `dim_{i + headDim/2}` (full split-half),
     * and apply rotation only to pairs whose `i < halfRotary`. The remaining
     * pairs (where `i >= halfRotary`) are left untouched.
     *
     * This matches HF's "proportional RoPE" semantics: `inv_freq` has length
     * `headDim/2`, with the first `halfRotary` entries non-zero and the rest
     * zero (yielding cos=1, sin=0 → no rotation). The pairing offset is
     * always `halfHead = headDim/2`, NOT `halfRotary`. See HF transformers
     * `_compute_proportional_rope_parameters` and `apply_rotary_pos_emb`.
     */
    private fun applyRoPESplitHalf(input: Tensor<T, V>, position: Int, ctx: ExecutionContext): Tensor<T, V> {
        val lastDim = input.shape[input.rank - 1]
        require(lastDim == headDim) { "RoPE input last dim ($lastDim) != headDim ($headDim)" }

        if (rotaryDim == headDim) {
            return applyRoPESplitHalfFull(input, position, ctx)
        }

        // Partial rotary, HF-compatible. Split input by headDim/2, then within
        // each half pull out the rotated subset of length halfRotary:
        //   first half:  A = dims [0, halfRotary)        | B = dims [halfRotary, halfHead)
        //   second half: C = dims [halfHead, halfHead+halfRotary) | D = dims [halfHead+halfRotary, headDim)
        // Rotate (A, C) as a pair; leave B and D unchanged.
        val ops = ctx.ops
        val featureAxis = input.rank - 1
        val halfHead = headDim / 2
        val A = ops.narrow(input, featureAxis, 0, halfRotary)
        val B = ops.narrow(input, featureAxis, halfRotary, halfHead - halfRotary)
        val C = ops.narrow(input, featureAxis, halfHead, halfRotary)
        val D = ops.narrow(input, featureAxis, halfHead + halfRotary, headDim - halfHead - halfRotary)

        val seqLen = input.shape[input.rank - 2]
        val cosData = FloatArray(seqLen * halfRotary)
        val sinData = FloatArray(seqLen * halfRotary)
        for (s in 0 until seqLen) {
            val pos = position + s
            for (i in 0 until halfRotary) {
                cosData[s * halfRotary + i] = cosTable[pos * halfRotary + i]
                sinData[s * halfRotary + i] = sinTable[pos * halfRotary + i]
            }
        }
        val cosShape = Shape(seqLen, halfRotary)
        // Heap-backed wrap, NOT ctx.fromFloatArray — fromFloatArray would
        // copy these transient cos/sin tables into fresh MemorySegments
        // from Arena.ofAuto(). RoPE runs twice per MHA (Q, K) × every
        // layer × every forward, and direct-memory pressure doesn't trigger
        // GC, so the auto-arenas accumulate until -XX:MaxDirectMemorySize
        // is exhausted. Same root-cause class as the sliceView leak
        // (commit 319c394). Heap arrays follow normal GC.
        val cosTensor: Tensor<T, V> = ctx.fromData(
            sk.ainet.lang.tensor.data.DenseFloatArrayTensorData<T>(cosShape, cosData) as sk.ainet.lang.tensor.data.TensorData<T, V>,
            input.dtype
        )
        val sinTensor: Tensor<T, V> = ctx.fromData(
            sk.ainet.lang.tensor.data.DenseFloatArrayTensorData<T>(cosShape, sinData) as sk.ainet.lang.tensor.data.TensorData<T, V>,
            input.dtype
        )

        // Standard 2D rotation: (a, b) -> (a*cos - b*sin, a*sin + b*cos)
        val rotA = ops.subtract(ops.multiply(A, cosTensor), ops.multiply(C, sinTensor))
        val rotC = ops.add(ops.multiply(C, cosTensor), ops.multiply(A, sinTensor))

        return ops.concat(listOf(rotA, B, rotC, D), dim = featureAxis)
    }

    private fun applyRoPESplitHalfFull(input: Tensor<T, V>, position: Int, ctx: ExecutionContext): Tensor<T, V> {
        val ops = ctx.ops
        val splits = ops.split(input, halfRotary, dim = input.rank - 1)
        val even = splits[0]
        val odd = splits[1]

        val seqLen = input.shape[input.rank - 2]
        val cosData = FloatArray(seqLen * halfRotary)
        val sinData = FloatArray(seqLen * halfRotary)
        for (s in 0 until seqLen) {
            val pos = position + s
            for (i in 0 until halfRotary) {
                cosData[s * halfRotary + i] = cosTable[pos * halfRotary + i]
                sinData[s * halfRotary + i] = sinTable[pos * halfRotary + i]
            }
        }

        val cosShape = Shape(seqLen, halfRotary)
        // Heap-backed wrap — see applyRoPESplitHalf for why fromFloatArray
        // is poison on the hot path (direct-memory leak via Arena.ofAuto).
        val cosTensor: Tensor<T, V> = ctx.fromData(
            sk.ainet.lang.tensor.data.DenseFloatArrayTensorData<T>(cosShape, cosData) as sk.ainet.lang.tensor.data.TensorData<T, V>,
            input.dtype
        )
        val sinTensor: Tensor<T, V> = ctx.fromData(
            sk.ainet.lang.tensor.data.DenseFloatArrayTensorData<T>(cosShape, sinData) as sk.ainet.lang.tensor.data.TensorData<T, V>,
            input.dtype
        )

        val rotEven = ops.subtract(ops.multiply(even, cosTensor), ops.multiply(odd, sinTensor))
        val rotOdd = ops.add(ops.multiply(odd, cosTensor), ops.multiply(even, sinTensor))

        return ops.concat(listOf(rotEven, rotOdd), dim = input.rank - 1)
    }

    /**
     * Interleaved mode: adjacent elements form pairs.
     * [x0, x1, x2, x3, ...] → pairs (x0, x1), (x2, x3), ...
     *
     * This is the format used by LLaMA, Mistral, and GGUF models.
     * Operates on the raw float data for efficiency since tensor ops don't
     * natively support stride-2 element selection. The trailing
     * `headDim - rotaryDim` floats of every head are left untouched.
     */
    private fun applyRoPEInterleaved(input: Tensor<T, V>, position: Int, ctx: ExecutionContext): Tensor<T, V> {
        // Graph tracing: the raw-array path below reads input.data and rebuilds via
        // fromFloatArray, which records the rotated Q/K as a DISCONNECTED CONSTANT —
        // severing the link to the projection weights. Post-GQA-broadcast that lowers
        // to a slice-into-empty const cascade that crashes iree-compile. Under the
        // tracing wrapper (KspTensorOps), take the traceable op-based path so the
        // rotation is recorded as tensor ops. Full-rotary only (TinyLlama/Llama/
        // Mistral); partial rotary keeps the raw path (no GGUF model needs it traced).
        if (input.ops is KspTensorOps) {
            // Traceable ops path for BOTH full and partial rotary. The raw-array path
            // below bakes a disconnected constant under void/graph tracing (it reads
            // input.data), so anything traced (incl. Moonshine's partial rotary) must
            // use the op-based rotation to stay wired to the projection weights.
            return applyRoPEInterleavedOps(input, position, ctx)
        }
        val data = input.data.copyToFloatArray()
        val lastDim = input.shape[input.rank - 1]
        require(lastDim == headDim) { "RoPE input last dim ($lastDim) != headDim ($headDim)" }
        val seqLen = input.shape[input.rank - 2]

        // Number of independent vectors (heads or batch*heads)
        val nVectors = data.size / (seqLen * lastDim)

        for (vec in 0 until nVectors) {
            for (s in 0 until seqLen) {
                val pos = position + s
                val baseIdx = (vec * seqLen + s) * lastDim
                for (i in 0 until halfRotary) {
                    val idx0 = baseIdx + i * 2
                    val idx1 = baseIdx + i * 2 + 1
                    val cosVal = cosTable[pos * halfRotary + i]
                    val sinVal = sinTable[pos * halfRotary + i]
                    val v0 = data[idx0]
                    val v1 = data[idx1]
                    data[idx0] = v0 * cosVal - v1 * sinVal
                    data[idx1] = v0 * sinVal + v1 * cosVal
                }
                // Indices baseIdx + rotaryDim ... baseIdx + lastDim - 1 are left unchanged.
            }
        }

        return ctx.fromFloatArray(input.shape, input.dtype, data)
    }

    /**
     * Traceable interleaved RoPE: pure tensor ops, numerically identical to
     * [applyRoPEInterleaved] but recordable to a compute graph. Used under
     * void/graph tracing where the raw-array path bakes a disconnected constant.
     *
     * Interleaved pairing `(x[2i], x[2i+1])` is realized by reshaping the rotated
     * subspace `[rotaryDim] -> [halfRotary, 2]` (row-major: `[i,0]=x[2i]`, `[i,1]=x[2i+1]`),
     * rotating the even/odd planes, then reshaping back. Supports PARTIAL rotary:
     * the leading `rotaryDim` head dims are rotated, the trailing `headDim - rotaryDim`
     * pass through unchanged (Moonshine rotates 32 of 36). Traceable (pure tensor ops).
     */
    private fun applyRoPEInterleavedOps(input: Tensor<T, V>, position: Int, ctx: ExecutionContext): Tensor<T, V> {
        val seqLen = input.shape[input.rank - 2]
        val (cosFull, sinFull) = buildInterleavedCosSin(position, seqLen)
        val fullShape = Shape(seqLen, headDim)
        @Suppress("UNCHECKED_CAST")
        val cosTensor: Tensor<FP32, V> = ctx.fromData(
            sk.ainet.lang.tensor.data.DenseFloatArrayTensorData<FP32>(fullShape, cosFull) as sk.ainet.lang.tensor.data.TensorData<FP32, V>,
            FP32::class,
        )
        @Suppress("UNCHECKED_CAST")
        val sinTensor: Tensor<FP32, V> = ctx.fromData(
            sk.ainet.lang.tensor.data.DenseFloatArrayTensorData<FP32>(fullShape, sinFull) as sk.ainet.lang.tensor.data.TensorData<FP32, V>,
            FP32::class,
        )
        return applyInterleavedRotation(input, cosTensor, sinTensor, ctx)
    }

    /**
     * Build the sign-baked, pair-repeated cos/sin tables `[seqLen, headDim]` for interleaved
     * full-head RoPE at [position] (see [applyInterleavedRotation] for the layout: pair (2i, 2i+1)
     * shares cos/sin[i]; the rotation's sign is folded into the even lane of sin; tail pairs
     * i >= halfRotary pass through with cos=1/sin=0).
     *
     * Public so a runtime that supplies cos/sin as graph INPUTS — e.g. Moonshine's
     * `decoder_with_past`, whose position is a runtime value, not a compile-time constant — can
     * produce byte-identical tables host-side and feed them to [forwardWithCosSin]. (No in-graph
     * gather op is needed; this mirrors how token embeddings are looked up host-side.)
     */
    public fun buildInterleavedCosSin(position: Int, seqLen: Int): Pair<FloatArray, FloatArray> {
        val halfHead = headDim / 2
        val cosFull = FloatArray(seqLen * headDim)
        val sinFull = FloatArray(seqLen * headDim)
        for (s in 0 until seqLen) {
            val pos = position + s
            for (i in 0 until halfHead) {
                val c = if (i < halfRotary) cosTable[pos * halfRotary + i] else 1.0f
                val sn = if (i < halfRotary) sinTable[pos * halfRotary + i] else 0.0f
                cosFull[s * headDim + 2 * i] = c; cosFull[s * headDim + 2 * i + 1] = c
                sinFull[s * headDim + 2 * i] = -sn; sinFull[s * headDim + 2 * i + 1] = sn
            }
        }
        return cosFull to sinFull
    }

    /**
     * Build the sign-baked, half-repeated cos/sin tables `[seqLen, headDim]` for split-half
     * (NEOX / HF) RoPE at [position] — the SPLIT_HALF analogue of [buildInterleavedCosSin].
     *
     * Split-half pairs `dim_i` with `dim_{i + headDim/2}`; the rotation is
     * `out = x * cosFull + swapHalves(x) * sinFull` where `swapHalves(x)` concatenates the
     * second half then the first half (see [applySplitHalfRotation]). The rotation's sign is
     * folded into the FIRST-half lane of sin (negated) so no separate negate node is needed —
     * matching HF `apply_rotary_pos_emb` with `rotate_half`. Tail pairs `i >= halfRotary`
     * (partial rotary) pass through with `cos=1, sin=0`; for FunctionGemma (`partialRotary=1.0`)
     * every pair rotates.
     *
     * Public so a runtime that feeds cos/sin as graph INPUTS (the gemma `decoder_with_past`,
     * whose decode position is a runtime value) can produce byte-identical tables host-side and
     * pass them to [forwardWithCosSin]. Numerically identical to [applyRoPESplitHalfFull] /
     * [applyRoPESplitHalf] at the same position (both read the same `cosTable`/`sinTable`).
     */
    public fun buildSplitHalfCosSin(position: Int, seqLen: Int): Pair<FloatArray, FloatArray> {
        val half = headDim / 2
        val cosFull = FloatArray(seqLen * headDim)
        val sinFull = FloatArray(seqLen * headDim)
        for (s in 0 until seqLen) {
            val pos = position + s
            for (i in 0 until half) {
                val c = if (i < halfRotary) cosTable[pos * halfRotary + i] else 1.0f
                val sn = if (i < halfRotary) sinTable[pos * halfRotary + i] else 0.0f
                cosFull[s * headDim + i] = c
                cosFull[s * headDim + half + i] = c
                sinFull[s * headDim + i] = -sn
                sinFull[s * headDim + half + i] = sn
            }
        }
        return cosFull to sinFull
    }

    /**
     * RoPE with cos/sin supplied as TENSORS (sign-baked, per-mode layout: [buildInterleavedCosSin]
     * for INTERLEAVED, [buildSplitHalfCosSin] for SPLIT_HALF) rather than derived from a
     * compile-time position — so the position can be a runtime graph input. Routes by [mode] so
     * callers stay mode-agnostic. Numerically identical to [forward]; the rotation runs in f32.
     */
    public fun forwardWithCosSin(
        input: Tensor<T, V>,
        cosFull: Tensor<T, V>,
        sinFull: Tensor<T, V>,
        ctx: ExecutionContext,
    ): Tensor<T, V> {
        val ops = ctx.ops
        @Suppress("UNCHECKED_CAST")
        val cosF: Tensor<FP32, V> = if (cosFull.dtype == FP32::class) cosFull as Tensor<FP32, V> else ops.convert(cosFull, FP32)
        @Suppress("UNCHECKED_CAST")
        val sinF: Tensor<FP32, V> = if (sinFull.dtype == FP32::class) sinFull as Tensor<FP32, V> else ops.convert(sinFull, FP32)
        return sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, input) {
            when (mode) {
                RoPEMode.SPLIT_HALF -> applySplitHalfRotation(input, cosF, sinF, ctx)
                RoPEMode.INTERLEAVED -> applyInterleavedRotation(input, cosF, sinF, ctx)
            }
        }
    }

    /**
     * The position-independent interleaved rotation `out = x * cos_full + swap(x) * sin_full`.
     * `swap(x)` maps each pair (x0, x1) -> (x1, x0); the rotation's sign lives in `sin_full`
     * (even lane negated) so this needs only a pair-SWAP, no negate node (the OPTIMIZED DAG
     * runtime drops a mulScalar here to 0-outputs on interleaved models). This is the FULL-HEAD
     * form the reference ONNX uses — numerically identical to rotating even/odd separately and
     * re-interleaving, but the recombine reshape mis-lays-out on the Torq NPU while this compiles
     * bit-exact (SL2610 sim). Extracted so the compile-time-position and runtime-cos/sin paths
     * share identical math. cos/sin are `[seqLen, headDim]`, broadcasting over the leading dims.
     */
    private fun applyInterleavedRotation(
        input: Tensor<T, V>,
        cosTensor: Tensor<FP32, V>,
        sinTensor: Tensor<FP32, V>,
        ctx: ExecutionContext,
    ): Tensor<T, V> {
        val ops = ctx.ops
        val rank = input.rank
        require(input.shape[rank - 1] == headDim) { "RoPE input last dim (${input.shape[rank - 1]}) != headDim ($headDim)" }
        val halfHead = headDim / 2
        val isF32 = input.dtype == FP32::class
        @Suppress("UNCHECKED_CAST")
        val xF: Tensor<FP32, V> = if (isF32) input as Tensor<FP32, V> else ops.convert(input, FP32)
        val leading = IntArray(rank - 1) { input.shape[it] }
        val pairedShape = Shape(*leading, halfHead, 2)
        val planeShape = Shape(*leading, halfHead)
        val pairAxis = rank
        val paired = ops.reshape(xF, pairedShape)
        val even = ops.reshape(ops.narrow(paired, pairAxis, 0, 1), planeShape)
        val odd = ops.reshape(ops.narrow(paired, pairAxis, 1, 1), planeShape)
        val swapped = ops.reshape(
            ops.concat(listOf(ops.unsqueeze(odd, rank), ops.unsqueeze(even, rank)), dim = rank),
            input.shape,
        )
        val outF = ops.add(ops.multiply(xF, cosTensor), ops.multiply(swapped, sinTensor))
        val modelDtype: DType = if (input.dtype == FP16::class) FP16 else BF16
        @Suppress("UNCHECKED_CAST")
        return if (isF32) outF as Tensor<T, V> else ops.convert(outF, modelDtype) as Tensor<T, V>
    }

    /**
     * The position-independent split-half rotation `out = x * cos_full + swapHalves(x) * sin_full`.
     * `swapHalves(x)` concatenates the second half then the first half of the feature axis; the
     * rotation's sign lives in `sin_full` (first-half lane negated by [buildSplitHalfCosSin]) so no
     * negate node is needed. This is the SPLIT_HALF (NEOX / HF `rotate_half`) analogue of
     * [applyInterleavedRotation] — numerically identical to [applyRoPESplitHalfFull] and (for the
     * partial case) [applyRoPESplitHalf], but with cos/sin fed in so the position is a runtime
     * value. cos/sin are `[seqLen, headDim]`, broadcasting over the leading dims.
     */
    private fun applySplitHalfRotation(
        input: Tensor<T, V>,
        cosTensor: Tensor<FP32, V>,
        sinTensor: Tensor<FP32, V>,
        ctx: ExecutionContext,
    ): Tensor<T, V> {
        val ops = ctx.ops
        val rank = input.rank
        require(input.shape[rank - 1] == headDim) { "RoPE input last dim (${input.shape[rank - 1]}) != headDim ($headDim)" }
        val half = headDim / 2
        val featureAxis = rank - 1
        val isF32 = input.dtype == FP32::class
        @Suppress("UNCHECKED_CAST")
        val xF: Tensor<FP32, V> = if (isF32) input as Tensor<FP32, V> else ops.convert(input, FP32)
        val first = ops.narrow(xF, featureAxis, 0, half)
        val second = ops.narrow(xF, featureAxis, half, half)
        val swapped = ops.concat(listOf(second, first), dim = featureAxis)
        val outF = ops.add(ops.multiply(xF, cosTensor), ops.multiply(swapped, sinTensor))
        val modelDtype: DType = if (input.dtype == FP16::class) FP16 else BF16
        @Suppress("UNCHECKED_CAST")
        return if (isF32) outF as Tensor<T, V> else ops.convert(outF, modelDtype) as Tensor<T, V>
    }
}
