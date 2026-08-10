package sk.ainet.models.whisper

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.transformer.VoidDense
import sk.ainet.lang.nn.transformer.linearProject
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.math.sqrt
import kotlin.reflect.KClass

/**
 * Whisper attention projections, hand-wired for full control over the export
 * shape rules (additive f32 masks only — never `causal=true`/i1 select, the
 * Vulkan/SPIR-V constraint from the validated ONNX pipeline).
 *
 * HF whisper projections are biased EXCEPT `k_proj` (verified against the
 * checkpoint and `export_static_kv.py`). Scale semantics: the reference applies
 * `headDim^-0.25` to both q and k; SDPA's single `scale = headDim^-0.5` on the
 * q·kᵀ product is mathematically identical.
 */
public class WhisperAttentionProjections<T : DType, V>(
    namePrefix: String,
    private val dim: Int,
    public val nHeads: Int,
    public val headDim: Int,
    dtype: KClass<T>,
) : Module<T, V>() {

    override val name: String = namePrefix

    public val qProj: VoidDense<T, V> = VoidDense("$namePrefix.q_proj", dim, dim, dtype, addBias = true)
    public val kProj: VoidDense<T, V> = VoidDense("$namePrefix.k_proj", dim, dim, dtype, addBias = false)
    public val vProj: VoidDense<T, V> = VoidDense("$namePrefix.v_proj", dim, dim, dtype, addBias = true)
    public val oProj: VoidDense<T, V> = VoidDense("$namePrefix.o_proj", dim, dim, dtype, addBias = true)

    override val modules: List<Module<T, V>> = listOf(qProj, kProj, vProj, oProj)

    public val scale: Float = 1f / sqrt(headDim.toFloat())

    /** `[1, S, dim]` → heads-first `[1, nHeads, S, headDim]`. */
    public fun toHeads(x: Tensor<T, V>, seq: Int, ctx: ExecutionContext): Tensor<T, V> {
        val ops = ctx.ops
        val r = ops.reshape(x, Shape(1, seq, nHeads, headDim))
        return ops.permute(r, intArrayOf(0, 2, 1, 3))
    }

    /** heads-first `[1, nHeads, S, headDim]` → `[1, S, dim]`. */
    public fun mergeHeads(x: Tensor<T, V>, seq: Int, ctx: ExecutionContext): Tensor<T, V> {
        val ops = ctx.ops
        val p = ops.permute(x, intArrayOf(0, 2, 1, 3))
        return ops.reshape(p, Shape(1, seq, dim))
    }

    /**
     * Full attention over flat (`[1, ·, dim]`) q-input / k / v with an optional
     * additive mask `[1, 1, qSeq, kvSeq]`. Returns `[1, qSeq, dim]` after o_proj.
     */
    public fun attend(
        qInput: Tensor<T, V>,
        kFlat: Tensor<T, V>,
        vFlat: Tensor<T, V>,
        qSeq: Int,
        kvSeq: Int,
        mask: Tensor<T, V>?,
        ctx: ExecutionContext,
    ): Tensor<T, V> {
        val ops = ctx.ops
        val q = toHeads(linearProjectBias(qProj, qInput, ctx), qSeq, ctx)
        val k = toHeads(kFlat, kvSeq, ctx)
        val v = toHeads(vFlat, kvSeq, ctx)
        val o = ops.scaledDotProductAttention(q, k, v, mask = mask, scale = scale, causal = false)
        return linearProjectBias(oProj, mergeHeads(o, qSeq, ctx), ctx)
    }

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> =
        error("WhisperAttentionProjections is driven explicitly; use attend(...)")
}

/** `x @ Wᵀ (+ bias)` through a [VoidDense]'s params (its own forward would work too;
 *  this keeps the call sites explicit about which projection weight is used). */
public fun <T : DType, V> linearProjectBias(
    dense: VoidDense<T, V>,
    x: Tensor<T, V>,
    ctx: ExecutionContext,
): Tensor<T, V> {
    val ops = ctx.ops
    var out = linearProject(ops, x, dense.params[0].value)
    if (dense.params.size > 1) out = ops.add(out, dense.params[1].value)
    return out
}
