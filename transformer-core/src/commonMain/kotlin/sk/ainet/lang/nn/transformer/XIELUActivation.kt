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
 * xIELU activation with per-layer learned parameters (Apertus architecture).
 *
 * For x > 0: softplus(alpha_p) * x^2 + beta * x
 * For x ≤ 0: (expm1(clamp(x, eps)) - x) * (beta + softplus(alpha_n)) + beta * x
 *
 * Parameters: alpha_p, alpha_n, beta, eps — all scalar, learned per layer.
 *
 * @param name module name
 */
public class XIELUActivation<T : DType, V>(
    override val name: String = "XIELUActivation"
) : Module<T, V>(), ModuleParameters<T, V> {

    @Suppress("UNCHECKED_CAST")
    private fun scalarParam(paramName: String, initValue: Float): ModuleParameter<T, V> {
        val tensor = VoidOpsTensor(
            object : sk.ainet.lang.tensor.data.TensorData<T, V> {
                override val shape = Shape(1)
                override fun get(vararg indices: Int): V = initValue as V
                override fun set(vararg indices: Int, value: V) {}
            },
            Any::class as KClass<T>
        )
        return ModuleParameter.WeightParameter("$name.$paramName", tensor)
    }

    override val params: List<ModuleParameter<T, V>> = listOf(
        scalarParam("alpha_p", 1.0f),
        scalarParam("alpha_n", 1.0f),
        scalarParam("beta", 1.0f),
        scalarParam("eps", -10.0f)
    )

    override val modules: List<Module<T, V>> = emptyList()

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        val ops = ctx.ops

        // The four learned parameters are frozen per-layer scalars, so their softplus
        // transforms are computed host-side in float — exactly, with the large-x guard.
        // The previous in-graph version used exp() as a "simplified softplus approx":
        // exp(166) (a real Apertus-8B alpha_p value) is Inf, and Inf * 0-mask = NaN, so
        // every logit came out NaN and greedy decode produced <unk> forever. Even for
        // small alphas exp(x) != ln(1+exp(x)) — the math was never faithful.
        val alphaP = scalarValue(params[0].value)
        val alphaN = scalarValue(params[1].value)
        val beta = scalarValue(params[2].value)
        val eps = scalarValue(params[3].value)

        val alphaPEff = softplus(alphaP)
        val alphaNEff = beta + softplus(alphaN)

        // Compute masks
        val posMask = ops.ge(input, 0.0f)    // 1.0 where x >= 0, 0.0 otherwise
        val negMask = ops.lt(input, 0.0f)    // 1.0 where x < 0, 0.0 otherwise

        // Positive branch: softplus(αp) * x² + β * x
        val xSquared = ops.multiply(input, input)
        val posResult = ops.add(ops.mulScalar(xSquared, alphaPEff), ops.mulScalar(input, beta))
        val posContrib = ops.multiply(posResult, posMask)

        // Negative branch: (expm1(min(x, eps)) - x) * (β + softplus(αn)) + β * x.
        // The lower clamp only keeps exp() in float range: expm1 saturates at -1 long
        // before -88, so it does not perturb the reference math.
        val clamped = ops.clamp(input, -88.0f, eps)
        val expm1MinusX = ops.subtract(ops.expm1(clamped), input)
        val negResult = ops.add(ops.mulScalar(expm1MinusX, alphaNEff), ops.mulScalar(input, beta))
        val negContrib = ops.multiply(negResult, negMask)

        return ops.add(posContrib, negContrib)
    }

    private fun scalarValue(tensor: Tensor<T, V>): Float =
        (tensor.data.get(0) as Number).toFloat()

    /** `softplus(x) = ln(1 + exp(x))` with the standard large-x shortcut (exp overflow). */
    private fun softplus(x: Float): Float =
        if (x > 20f) x else kotlin.math.ln(1f + kotlin.math.exp(x))
}
