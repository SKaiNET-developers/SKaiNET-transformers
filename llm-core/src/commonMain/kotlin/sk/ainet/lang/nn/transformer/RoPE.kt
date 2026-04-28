package sk.ainet.lang.nn.transformer

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
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
        for (pos in 0 until maxSeqLen) {
            for (i in 0 until halfRotary) {
                var freq = 1.0f / base.pow(2.0f * i / headDim)
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
}
