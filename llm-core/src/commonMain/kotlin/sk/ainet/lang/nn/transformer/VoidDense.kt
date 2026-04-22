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
 * Lightweight dense (linear) projection whose weight and bias are void
 * placeholders — a shape-only [TensorData] wrapped in a [VoidOpsTensor].
 *
 * Intended for large-vocab `lm_head` projections (e.g. Gemma 4 E2B at
 * 262 144 × 1536) where the default [sk.ainet.lang.nn.dsl.NeuralNetworkDsl.dense]
 * builder would eagerly allocate a gigabytes-scale zero tensor before any
 * weights are loaded. Using [VoidDense] defers all storage until the weight
 * mapper populates the parameter.
 *
 * Follows the same convention as the transformer primitives in this package
 * ([SwiGLUFFN], [GeGLUFFN]): `$name.weight` + `$name.bias` parameter names so
 * standard weight-name resolvers pick them up.
 *
 * Forward pass computes `input @ weight^T` with no bias (bias parameter is
 * declared so resolvers that expect it do not fail to map, but is not added
 * to the output).
 */
@Suppress("UNCHECKED_CAST")
public class VoidDense<T : DType, V>(
    override val name: String,
    public val outDim: Int,
    public val inDim: Int
) : Module<T, V>(), ModuleParameters<T, V> {

    private fun voidTensor(shape: Shape): VoidOpsTensor<T, V> = VoidOpsTensor(
        object : TensorData<T, V> {
            override val shape: Shape = shape
            override fun get(vararg indices: Int): V = 0.0f as V
            override fun set(vararg indices: Int, value: V) {}
        },
        Any::class as KClass<T>
    )

    override val params: List<ModuleParameter<T, V>> = listOf(
        ModuleParameter.WeightParameter("$name.weight", voidTensor(Shape(outDim, inDim))),
        ModuleParameter.BiasParameter("$name.bias", voidTensor(Shape(outDim)))
    )

    override val modules: List<Module<T, V>> = emptyList()

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        val ops = ctx.ops
        val weight = params[0].value
        val wShape = weight.shape
        return if (wShape.rank == 2 && wShape[0] > wShape[1]) {
            ops.matmul(input, ops.transpose(weight))
        } else {
            ops.matmul(input, weight)
        }
    }
}
