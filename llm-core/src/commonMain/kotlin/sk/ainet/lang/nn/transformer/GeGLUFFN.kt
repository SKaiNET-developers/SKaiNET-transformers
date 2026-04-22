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
 * GeGLU Feed-Forward Network (Gemma-style gated FFN).
 *
 * Computes: down_proj(gelu(gate_proj(x)) * up_proj(x))
 *
 * Identical parameter layout to [SwiGLUFFN] so [sk.ainet.apps.llm.weights.LlamaGGUFNameResolver]
 * maps the same GGUF tensor names. The only difference is the activation
 * (GELU instead of SiLU).
 *
 * Weight parameters (no bias):
 * - gate_proj.weight: [hiddenDim, dim]
 * - up_proj.weight:   [hiddenDim, dim]
 * - down_proj.weight: [dim, hiddenDim]
 */
public class GeGLUFFN<T : DType, V>(
    public val dim: Int,
    public val hiddenDim: Int,
    override val name: String = "GeGLUFFN"
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

        val gate = ops.gelu(linearProject(ops, input, gateW))
        val up = linearProject(ops, input, upW)
        val gated = ops.multiply(gate, up)
        return linearProject(ops, gated, downW)
    }
}
