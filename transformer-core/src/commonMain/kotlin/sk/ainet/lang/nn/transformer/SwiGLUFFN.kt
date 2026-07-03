package sk.ainet.lang.nn.transformer

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * SwiGLU Feed-Forward Network (Llama-style gated FFN).
 *
 * Computes: down_proj(silu(gate_proj(x)) * up_proj(x))
 *
 * Weight parameters (no bias):
 * - gate_proj.weight: [hiddenDim, dim]
 * - up_proj.weight:   [hiddenDim, dim]
 * - down_proj.weight: [dim, hiddenDim]
 *
 * @param dim model dimension
 * @param hiddenDim FFN hidden dimension (typically ~2.67 * dim)
 * @param name module name
 */
public class SwiGLUFFN<T : DType, V>(
    public val dim: Int,
    public val hiddenDim: Int,
    override val name: String = "SwiGLUFFN"
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

    override val modules: List<Module<T, V>> = emptyList()

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        val ops = ctx.ops
        val gateW = params[0].value
        val upW = params[1].value
        val downW = params[2].value

        // gate = silu(input @ gate_proj^T)
        val gateLin = PhaseProfile.time("ffn.proj") { linearProject(ops, input, gateW) }
        val gate = PhaseProfile.time("ffn.eltwise") { ops.silu(gateLin) }
        // up = input @ up_proj^T
        val up = PhaseProfile.time("ffn.proj") { linearProject(ops, input, upW) }
        // gated = gate * up
        val gated = PhaseProfile.time("ffn.eltwise") { ops.multiply(gate, up) }
        // output = gated @ down_proj^T
        return PhaseProfile.time("ffn.proj") { linearProject(ops, gated, downW) }
    }
}
