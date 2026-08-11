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
 * This is Solution C from `ISSUE-skainet-8b-oom.md` (#184 hoist 3): a
 * converter that already delivers the weight in the transposed `[in, out]`
 * layout marks its tensor data with [PreTransposedWeight] (e.g. via
 * [sk.ainet.lang.nn.quant.BlockQuantPacking.packPreTransposed]), and this
 * helper skips `ops.transpose` for it, dispatching `ops.matmul(x, W)`
 * directly — for packed quant weights that avoids even the engine's lazy
 * shape-swap transpose wrapper on every forward. Unmarked weights take the
 * classic `ops.matmul(x, ops.transpose(W))` path, where the engine's packed
 * `ops.transpose` support remains the fallback.
 *
 * A shape-only heuristic (`W.shape[0] == x.shape[-1]`) is unsafe for square
 * projections (e.g. `dim == qDim` on GQA-free configs), hence the
 * explicit-marker requirement.
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
