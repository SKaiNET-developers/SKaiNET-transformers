package sk.ainet.lang.nn.transformer

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.types.DType

/**
 * Linear projection `y = x @ W.t()` for a weight stored in the stock
 * `[out, in]` checkpoint layout.
 *
 * Intended as the single place every transformer DSL module goes through
 * when projecting against a weight parameter. Delegates to the engine
 * primitive [TensorOps.matmulWeightTransposed], which lets backends serve
 * the projection without materializing a transposed copy — packed block
 * weights (heap or mapped) dispatch to their transpose-aware kernels
 * directly.
 *
 * Every weight reaching this seam is engine-loader-produced
 * (`StreamingGgufParametersLoader` / SafeTensors) with a truthful logical
 * `[out, in]` shape; the legacy MemSeg converter lane and its
 * pre-transposed-marker branch were deleted in the #338 migration arc
 * (gemma phase, #341).
 *
 * @param ops active tensor operations (usually `ctx.ops`)
 * @param input input tensor of shape `[..., in]`
 * @param weight projection weight in `[out, in]` layout
 * @return `input @ W.t()`
 */
public fun <T : DType, V> linearProject(
    ops: TensorOps,
    input: Tensor<T, V>,
    weight: Tensor<T, V>
): Tensor<T, V> = ops.matmulWeightTransposed(input, weight)
