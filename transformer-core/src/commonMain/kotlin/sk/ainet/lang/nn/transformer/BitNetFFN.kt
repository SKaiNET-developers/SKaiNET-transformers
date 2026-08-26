package sk.ainet.lang.nn.transformer

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.normalization.RMSNormalization
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * BitNet b1.58 Feed-Forward Network (squared-ReLU gated FFN with a sub-layer norm).
 *
 * Computes: `down_proj(subNorm(relu(gate_proj(x))² * up_proj(x)))`
 *
 * The structure is verified against NeoGPU's reference driver (`hs_ml_infer.c`, the working
 * BitNet-2B4T inference this port tracks): squared ReLU on the gate instead of SiLU, and an
 * RMSNorm (`ffn_sub_norm`, over the FFN hidden dim) between the gated product and the down
 * projection.
 *
 * Weight parameters (no bias):
 * - gate_proj.weight: [hiddenDim, dim]
 * - up_proj.weight:   [hiddenDim, dim]
 * - down_proj.weight: [dim, hiddenDim]
 * - sub_norm.weight:  [hiddenDim]  (child [RMSNormalization], GGUF `blk.N.ffn_sub_norm.weight`)
 *
 * @param dim model dimension
 * @param hiddenDim FFN hidden dimension
 * @param subNormEps RMSNorm epsilon for the sub-layer norm (1e-5 for BitNet-2B4T)
 */
public class BitNetFFN<T : DType, V>(
    public val dim: Int,
    public val hiddenDim: Int,
    public val subNormEps: Double = 1e-5,
    override val name: String = "BitNetFFN",
    private val dtype: KClass<T>? = null,
) : Module<T, V>(), ModuleParameters<T, V> {

    @Suppress("UNCHECKED_CAST")
    private fun voidWeight(paramName: String, rows: Int, cols: Int): ModuleParameter<T, V> {
        val tensor = VoidOpsTensor(
            object : sk.ainet.lang.tensor.data.TensorData<T, V> {
                override val shape = Shape(rows, cols)
                override fun get(vararg indices: Int): V = 0.0f as V
                override fun set(vararg indices: Int, value: V) {}
            },
            Any::class as KClass<T>
        )
        return ModuleParameter.WeightParameter(paramName, tensor)
    }

    override val params: List<ModuleParameter<T, V>> = listOf(
        voidWeight("$name.gate_proj.weight", hiddenDim, dim),
        voidWeight("$name.up_proj.weight", hiddenDim, dim),
        voidWeight("$name.down_proj.weight", dim, hiddenDim)
    )

    /** The BitNet `ffn_sub_norm` over the FFN hidden dimension. */
    public val subNorm: RMSNormalization<T, V> =
        RMSNormalization(intArrayOf(hiddenDim), eps = subNormEps, name = "$name.sub_norm", dtype = dtype)

    override val modules: List<Module<T, V>> = listOf(subNorm)

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        val ops = ctx.ops
        val gateW = params[0].value
        val upW = params[1].value
        val downW = params[2].value

        // gate = relu(input @ gate_proj^T)²  — BitNet's squared ReLU
        val gateLin = PhaseProfile.time("ffn.proj") { linearProject(ops, input, gateW) }
        val gate = PhaseProfile.time("ffn.eltwise") {
            val r = ops.relu(gateLin)
            ops.multiply(r, r)
        }
        // up = input @ up_proj^T
        val up = PhaseProfile.time("ffn.proj") { linearProject(ops, input, upW) }
        // gated = gate² * up, then the ffn_sub_norm
        val gated = PhaseProfile.time("ffn.eltwise") { ops.multiply(gate, up) }
        val normed = subNorm.forward(gated, ctx)
        // output = normed @ down_proj^T
        return PhaseProfile.time("ffn.proj") { linearProject(ops, normed, downW) }
    }
}
