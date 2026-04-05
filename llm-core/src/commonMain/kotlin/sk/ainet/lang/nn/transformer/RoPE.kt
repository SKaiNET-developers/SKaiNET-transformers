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
 * Rotary Position Embedding (RoPE) module.
 *
 * Precomputes cos/sin frequency tables and applies rotary embeddings to input tensors.
 * Used by Llama, Apertus, and other decoder architectures.
 *
 * Input shape: [seqLen, dim] or [nHeads, seqLen, dim]
 * The last dimension is split into pairs for rotation.
 *
 * @param headDim dimension of each attention head (must be even)
 * @param maxSeqLen maximum sequence length for precomputed tables
 * @param base RoPE base frequency (default 10000.0)
 * @param mode pairing strategy: [RoPEMode.INTERLEAVED] (LLaMA default) or [RoPEMode.SPLIT_HALF]
 * @param name module name
 */
public class RoPE<T : DType, V>(
    public val headDim: Int,
    public val maxSeqLen: Int,
    private val base: Float = 10000.0f,
    public val mode: RoPEMode = RoPEMode.INTERLEAVED,
    override val name: String = "RoPE"
) : Module<T, V>() {

    init {
        require(headDim % 2 == 0) { "RoPE headDim must be even, got $headDim" }
    }

    override val modules: List<Module<T, V>> = emptyList()

    // Precomputed frequency tables: [maxSeqLen, headDim/2]
    private val cosTable: FloatArray = FloatArray(maxSeqLen * (headDim / 2))
    private val sinTable: FloatArray = FloatArray(maxSeqLen * (headDim / 2))

    init {
        val halfDim = headDim / 2
        for (pos in 0 until maxSeqLen) {
            for (i in 0 until halfDim) {
                val freq = 1.0f / base.pow(2.0f * i / headDim)
                val angle = pos * freq
                cosTable[pos * halfDim + i] = cos(angle)
                sinTable[pos * halfDim + i] = sin(angle)
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
     * Split-half mode: first half and second half of the last dimension form pairs.
     * [x0, x1, ..., x_{n/2-1}, x_{n/2}, ..., x_{n-1}] → pairs (x_i, x_{i+n/2})
     */
    private fun applyRoPESplitHalf(input: Tensor<T, V>, position: Int, ctx: ExecutionContext): Tensor<T, V> {
        val ops = ctx.ops
        val halfDim = headDim / 2
        val splits = ops.split(input, halfDim, dim = input.rank - 1)
        val even = splits[0]
        val odd = splits[1]

        val seqLen = input.shape[input.rank - 2]
        val cosData = FloatArray(seqLen * halfDim)
        val sinData = FloatArray(seqLen * halfDim)
        for (s in 0 until seqLen) {
            val pos = position + s
            for (i in 0 until halfDim) {
                cosData[s * halfDim + i] = cosTable[pos * halfDim + i]
                sinData[s * halfDim + i] = sinTable[pos * halfDim + i]
            }
        }

        val cosShape = Shape(seqLen, halfDim)
        val cosTensor: Tensor<T, V> = ctx.fromFloatArray(cosShape, input.dtype, cosData)
        val sinTensor: Tensor<T, V> = ctx.fromFloatArray(cosShape, input.dtype, sinData)

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
     * natively support stride-2 element selection.
     */
    private fun applyRoPEInterleaved(input: Tensor<T, V>, position: Int, ctx: ExecutionContext): Tensor<T, V> {
        val halfDim = headDim / 2
        val data = input.data.copyToFloatArray()
        val lastDim = input.shape[input.rank - 1]
        val seqLen = input.shape[input.rank - 2]

        // Number of independent vectors (heads or batch*heads)
        val nVectors = data.size / (seqLen * lastDim)

        for (vec in 0 until nVectors) {
            for (s in 0 until seqLen) {
                val pos = position + s
                val baseIdx = (vec * seqLen + s) * lastDim
                for (i in 0 until halfDim) {
                    val idx0 = baseIdx + i * 2
                    val idx1 = baseIdx + i * 2 + 1
                    val cosVal = cosTable[pos * halfDim + i]
                    val sinVal = sinTable[pos * halfDim + i]
                    val v0 = data[idx0]
                    val v1 = data[idx1]
                    data[idx0] = v0 * cosVal - v1 * sinVal
                    data[idx1] = v0 * sinVal + v1 * cosVal
                }
            }
        }

        return ctx.fromFloatArray(input.shape, input.dtype, data)
    }
}
