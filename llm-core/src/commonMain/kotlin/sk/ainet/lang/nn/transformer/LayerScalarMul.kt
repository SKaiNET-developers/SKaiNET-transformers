package sk.ainet.lang.nn.transformer

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Scalar-broadcast multiply by a `[1]`-shaped learned weight.
 *
 * Corresponds to Gemma 4's `self.layer_scalar` buffer (GGUF name:
 * `blk.N.layer_output_scale.weight`) — a single scalar per transformer
 * block applied at the very end of the block forward:
 *
 * ```python
 * # Gemma4TextDecoderLayer.forward tail
 * hidden_states *= self.layer_scalar
 * return hidden_states
 * ```
 *
 * Placeholder weight initialises to ones so an unmapped parameter is a
 * no-op, matching HF's `torch.ones(1)` default. The tensor backing is a
 * [VoidOpsTensor] with a shape-only [TensorData] — WeightMapper replaces
 * it with the loaded checkpoint weight during load.
 *
 * @param name module name. The single parameter is registered as
 *   `$name.weight` so resolver rules matching `layer_output_scale.weight`
 *   find it.
 */
@Suppress("UNCHECKED_CAST")
public class LayerScalarMul<T : DType, V>(
    override val name: String = "LayerScalarMul"
) : Module<T, V>(), ModuleParameters<T, V> {

    private fun onesPlaceholder(): VoidOpsTensor<T, V> = VoidOpsTensor(
        object : TensorData<T, V> {
            override val shape: Shape = Shape(1)
            override fun get(vararg indices: Int): V = 1.0f as V
            override fun set(vararg indices: Int, value: V) {}
        },
        Any::class as KClass<T>
    )

    override val params: List<ModuleParameter<T, V>> = listOf(
        ModuleParameter.WeightParameter("$name.weight", onesPlaceholder())
    )

    override val modules: List<Module<T, V>> = emptyList()

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        return ctx.ops.multiply(input, params[0].value)
    }
}
