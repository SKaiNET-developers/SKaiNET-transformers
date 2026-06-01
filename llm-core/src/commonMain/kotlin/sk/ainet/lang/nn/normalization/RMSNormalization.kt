package sk.ainet.lang.nn.normalization

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.*
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Optional interface that [TensorOps] implementations can provide to support
 * fused RMS normalization without intermediate tensor allocations.
 */
public interface FusedRmsNormOps {
    /**
     * Compute fused RMS normalization:
     *   output[i] = (input[i] / rms(input)) * weight[i]
     *
     * @return result tensor, or `null` if this implementation cannot handle the input
     */
    public fun <T : DType, V> fusedRmsNorm(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        eps: Float,
    ): Tensor<T, V>?
}

/**
 * RMS (Root Mean Square) Normalization layer.
 * Unlike LayerNormalization, RMSNorm has no bias and normalizes using only the
 * root mean square of the input, making it simpler and faster.
 *
 * Used extensively in LLaMA-family models.
 *
 * @param normalizedShape The shape of the normalization (typically the last dimension)
 * @param eps Small value added to the denominator for numerical stability
 * @param name Name of the module
 * @param initWeight Initial weight (scale) parameter
 */
public class RMSNormalization<T : DType, V>(
    private val normalizedShape: IntArray,
    private val eps: Double = 1e-5,
    override val name: String = "RMSNormalization",
    initWeight: Tensor<T, V>? = null,
    /**
     * When `true`, the gain is computed as `(1 + weight)` instead of `weight`.
     * Gemma family checkpoints store the RMSNorm gain centered at zero — the
     * trained value at initialisation is the deviation from identity. Without
     * this offset, an unset (zero-initialised) weight zeros out the post-norm
     * signal entirely, collapsing trunk activations to ~0 and producing
     * uniform output logits. Default `false` matches LLaMA / Mistral / GPT
     * conventions.
     */
    private val unitOffset: Boolean = false,
    // Logical element type prescribed by the DSL; keeps the placeholder weight
    // typed so the module traces before real weights load.
    private val dtype: KClass<T>? = null,
) : Module<T, V>(), ModuleParameters<T, V> {

    override val params: List<ModuleParameter<T, V>> = listOf(
        ModuleParameter.WeightParameter("$name.weight", initWeight ?: createIdentityPlaceholder())
    )

    override val modules: List<Module<T, V>>
        get() = emptyList()

    @Suppress("UNCHECKED_CAST")
    private fun createIdentityPlaceholder(): Tensor<T, V> {
        // Picks the value that makes the gain (weight or 1+weight) equal to 1
        // when the checkpoint hasn't loaded the weight yet. With unitOffset
        // we expect the trained weight near 0, so the placeholder is 0 too.
        // Without unitOffset, the trained weight is near 1 (LLaMA convention),
        // and the placeholder is 1.
        val placeholder: Float = if (unitOffset) 0.0f else 1.0f
        return VoidOpsTensor(
            object : sk.ainet.lang.tensor.data.TensorData<T, V> {
                override val shape = Shape(*normalizedShape)
                override fun get(vararg indices: Int): V = placeholder as V
                override fun set(vararg indices: Int, value: V) {}
            },
            (dtype ?: Any::class) as KClass<T>
        )
    }

    override fun forward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> =
        sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, input) {
            // Try fused path if the ops backend supports it. Skip for Gemma's
            // unit-offset variant — backends don't know about (1 + weight).
            val w = params[0].value
            if (!unitOffset) {
                val fusedOps = ctx.ops as? FusedRmsNormOps
                if (fusedOps != null) {
                    val result = fusedOps.fusedRmsNorm(input, w, eps.toFloat())
                    if (result != null) return@withForwardHooks result
                }
            }
            // Fallback: decomposed path
            val squared = input * input
            val mean = squared.mean(dim = input.rank - 1)
            // Unsqueeze so broadcasting works for batched input (e.g. [B, dim] / [B, 1])
            val rmsRaw = (mean + eps).sqrt()
            val rms = if (rmsRaw.rank < input.rank) rmsRaw.unsqueeze(rmsRaw.rank) else rmsRaw
            val normalized = input / rms
            val gain = if (unitOffset) ctx.ops.addScalar(w, 1f) else w
            val gainBcast = if (gain.rank == 1) gain.reshape(Shape(1, gain.shape[0])) else gain
            normalized * gainBcast
        }
}
