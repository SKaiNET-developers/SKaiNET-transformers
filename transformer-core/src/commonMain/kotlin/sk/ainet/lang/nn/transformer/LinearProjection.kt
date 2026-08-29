package sk.ainet.lang.nn.transformer

import sk.ainet.lang.nn.quant.PreTransposedWeight
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.types.DType

/**
 * Linear projection `y = x @ W.t()` for a weight stored in the stock
 * `[out, in]` checkpoint layout.
 *
 * Intended as the single place every transformer DSL module goes through
 * when projecting against a weight parameter.
 *
 * MIGRATION NOTE (#338 arc): the unmarked branch becomes the engine
 * primitive [TensorOps.matmulWeightTransposed] once the legacy MemSeg
 * converter lane is gone (the decoder/gemma migration phases). The flip
 * cannot be validated before then: the legacy lane's tensor data does not
 * declare a truthful block order, and the lane itself is broken at baseline
 * for pure-Q4_K/Q6_K models (transformers develop + engine 0.40.1 crashes
 * with the #993 ClassCastException on the first decode step; engines >= 0.49
 * turn that crash into silent garbage via matmulGeneric's raw-code
 * copyToFloatArray fallback — measured on Qwen2.5-1.5B). The engine-loader
 * migration replaces the lane wholesale, which is the actual fix.
 *
 * The [PreTransposedWeight] branch remains only while the legacy packing
 * layer exists (deleted with it in the gemma migration phase).
 *
 * @param ops active tensor operations (usually `ctx.ops`)
 * @param input input tensor of shape `[..., in]`
 * @param weight projection weight in `[out, in]` layout — or `[in, out]`
 *   when its data carries the [PreTransposedWeight] marker
 * @return `input @ W.t()`
 */
public fun <T : DType, V> linearProject(
    ops: TensorOps,
    input: Tensor<T, V>,
    weight: Tensor<T, V>
): Tensor<T, V> =
    if (weight.data is PreTransposedWeight) ops.matmul(input, weight)
    else ops.matmul(input, ops.transpose(weight))
