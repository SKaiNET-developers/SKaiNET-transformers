package sk.ainet.lang.nn.transformer

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.types.DType

/**
 * Linear projection `y = x @ W.t()` for a weight stored in the stock
 * `[out, in]` checkpoint layout.
 *
 * Intended as the single place every transformer DSL module goes through
 * when projecting against a weight parameter. Today it just materialises
 * the transpose and forwards to `ops.matmul`, so it's a drop-in rename
 * for `ops.matmul(x, ops.transpose(W))`.
 *
 * The future for this helper is Solution C from `ISSUE-skainet-8b-oom.md`:
 * when a MemSeg-style converter pre-transposes a quantized weight (Q4_K /
 * Q6_K dequant-and-transpose, or Q4_0 / Q8_0 via a transpose-on-the-fly
 * kernel), that conversion step will set an explicit pre-transposed
 * marker; this helper will read the marker and skip the transpose in the
 * pre-transposed case. A shape-only heuristic (`W.shape[0] == x.shape[-1]`)
 * is unsafe for square projections (e.g. `dim == qDim` on GQA-free
 * configs), hence the explicit-marker requirement.
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
): Tensor<T, V> = ops.matmul(input, ops.transpose(weight))
