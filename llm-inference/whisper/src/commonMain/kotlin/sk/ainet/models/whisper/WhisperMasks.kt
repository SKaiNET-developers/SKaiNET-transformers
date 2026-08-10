package sk.ainet.models.whisper

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Host-side constants and per-step masks for the fixed-masked-KV contract.
 * All masks are ADDITIVE f32 (0 = attend, [neg] = blocked) — never booleans —
 * because i1 compare/select does not codegen on Vulkan/SPIR-V.
 */
public object WhisperMasks {
    public const val NEG: Float = -1e4f

    /** Additive causal mask `[1, 1, seq, seq]` for the prefill prompt. */
    public fun <T : DType, V> causal(seq: Int, ctx: ExecutionContext, dtype: KClass<T>): Tensor<T, V> {
        val m = FloatArray(seq * seq)
        for (i in 0 until seq) for (j in 0 until seq) if (j > i) m[i * seq + j] = NEG
        return ctx.fromFloatArray(Shape(1, 1, seq, seq), dtype, m)
    }

    /** Constant zero-pad `[1, maxPositions - seq, dim]` appended to prefill self K/V. */
    public fun <T : DType, V> zeroPad(seq: Int, maxPositions: Int, dim: Int, ctx: ExecutionContext, dtype: KClass<T>): Tensor<T, V> =
        ctx.fromFloatArray(Shape(1, maxPositions - seq, dim), dtype, FloatArray((maxPositions - seq) * dim))

    /** Host per-step additive mask values `[maxP]`: 0 for i<=pos, NEG beyond. */
    public fun stepAddMask(pos: Int, maxPositions: Int): FloatArray =
        FloatArray(maxPositions) { i -> if (i <= pos) 0f else NEG }

    /** Host per-step one-hot write vector `[maxP]`: 1 at pos. */
    public fun stepWriteVector(pos: Int, maxPositions: Int): FloatArray =
        FloatArray(maxPositions) { i -> if (i == pos) 1f else 0f }
}
