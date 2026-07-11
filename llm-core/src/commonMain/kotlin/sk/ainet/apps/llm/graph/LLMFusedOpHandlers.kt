package sk.ainet.apps.llm.graph

import sk.ainet.lang.graph.exec.ComputeGraphExecutor
import sk.ainet.lang.graph.exec.FusedOpHandler
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.types.DType

/**
 * CPU fallback implementations for fused LLM operations.
 *
 * These handlers decompose fused ops back into sequences of [TensorOps] calls.
 * They produce correct results on any backend but don't provide the performance
 * benefit of a true fused kernel. Platform-specific backends (Metal, CUDA) should
 * register their own handlers to override these.
 */
public object LLMFusedOpHandlers {

    public fun registerAll() {
        ComputeGraphExecutor.registerFusedOp("fused_rms_norm", RmsNormHandler)
        ComputeGraphExecutor.registerFusedOp("fused_swiglu_ffn", SwiGluFFNHandler)
        ComputeGraphExecutor.registerFusedOp("fused_qkv_proj", QKVProjHandler)
        ComputeGraphExecutor.registerFusedOp("permute", PermuteHandler)
    }

    /**
     * Replays `permute` with its recorded `axes` parameter.
     *
     * Works around the executor's builtin dispatch (engine ≤ 0.35.0), which
     * replays "permute" as a plain last-two-dims [TensorOps.transpose] and
     * silently drops the axes — wrong for any rank-3 permutation like MHA's
     * heads/sequence swap `[1, 0, 2]` on multi-token (encoder / prefill)
     * traces. Single-token decode never hits it (the seqLen-1 shortcut
     * reshapes instead), which is why generation paths didn't surface this.
     * Remove once the engine's builtin honors `axes`.
     */
    private object PermuteHandler : FusedOpHandler<DType, Any> {
        override fun execute(
            ops: TensorOps,
            inputs: List<Tensor<DType, Any>>,
            params: Map<String, Any>
        ): List<Tensor<DType, Any>> {
            val axesParam = params["axes"] ?: params["dims"]
            val axes = when (axesParam) {
                is IntArray -> axesParam
                is List<*> -> axesParam.mapNotNull { (it as? Number)?.toInt() }.toIntArray()
                else -> null
            }
            return if (axes != null && axes.size == inputs[0].rank) {
                listOf(ops.permute(inputs[0], axes))
            } else {
                // No usable axes recorded — preserve the executor's legacy behavior.
                listOf(ops.transpose(inputs[0]))
            }
        }
    }

    private object RmsNormHandler : FusedOpHandler<DType, Any> {
        override fun execute(
            ops: TensorOps,
            inputs: List<Tensor<DType, Any>>,
            params: Map<String, Any>
        ): List<Tensor<DType, Any>> {
            require(inputs.size >= 1) { "fused_rms_norm requires at least 1 input (x), got ${inputs.size}" }
            val x = inputs[0]
            val eps = (params["eps"] as? Number)?.toFloat() ?: 1e-5f

            val xSquared = ops.multiply(x, x)
            val meanSquared = ops.mean(xSquared, -1)
            val meanPlusEps = ops.addScalar(meanSquared, eps)
            val rms = ops.sqrt(meanPlusEps)
            val normalized = ops.divide(x, rms)

            return if (inputs.size >= 2) {
                val weight = inputs[1]
                listOf(ops.multiply(normalized, weight))
            } else {
                listOf(normalized)
            }
        }
    }

    private object SwiGluFFNHandler : FusedOpHandler<DType, Any> {
        override fun execute(
            ops: TensorOps,
            inputs: List<Tensor<DType, Any>>,
            params: Map<String, Any>
        ): List<Tensor<DType, Any>> {
            val x: Tensor<DType, Any>
            val gateWeight: Tensor<DType, Any>
            val upWeight: Tensor<DType, Any>
            val downWeight: Tensor<DType, Any>

            if (inputs.size >= 5) {
                x = inputs[0]
                gateWeight = inputs[2]
                upWeight = inputs[3]
                downWeight = inputs[4]
            } else {
                require(inputs.size >= 4) { "fused_swiglu_ffn requires at least 4 inputs, got ${inputs.size}" }
                x = inputs[0]
                gateWeight = inputs[1]
                upWeight = inputs[2]
                downWeight = inputs[3]
            }

            // Use transposeB flags propagated by LLMFusionPass from the original
            // matmul nodes. Falls back to shape-based detection for graphs not
            // processed by TransposeEliminationPass.
            val transposeGate = params["transposeGate"] as? Boolean ?: false
            val transposeUp = params["transposeUp"] as? Boolean ?: false
            val transposeDown = params["transposeDown"] as? Boolean ?: false

            fun applyTranspose(w: Tensor<DType, Any>, shouldTranspose: Boolean, input: Tensor<DType, Any>): Tensor<DType, Any> {
                if (shouldTranspose) return ops.transpose(w)
                // Fallback: shape-based detection for unfused graphs
                val inCols = input.shape[input.rank - 1]
                val wRows = w.shape[0]
                return if (wRows != inCols && w.rank == 2 && w.shape[1] == inCols) {
                    ops.transpose(w)
                } else {
                    w
                }
            }

            val gateOut = ops.matmul(x, applyTranspose(gateWeight, transposeGate, x))
            val gateActivated = ops.silu(gateOut)
            val upOut = ops.matmul(x, applyTranspose(upWeight, transposeUp, x))
            val gated = ops.multiply(gateActivated, upOut)
            val result = ops.matmul(gated, applyTranspose(downWeight, transposeDown, gated))

            return listOf(result)
        }
    }

    private object QKVProjHandler : FusedOpHandler<DType, Any> {
        override fun execute(
            ops: TensorOps,
            inputs: List<Tensor<DType, Any>>,
            params: Map<String, Any>
        ): List<Tensor<DType, Any>> {
            val x: Tensor<DType, Any>
            val qWeight: Tensor<DType, Any>
            val kWeight: Tensor<DType, Any>
            val vWeight: Tensor<DType, Any>

            if (inputs.size >= 6) {
                x = inputs[0]
                qWeight = inputs[3]
                kWeight = inputs[4]
                vWeight = inputs[5]
            } else {
                require(inputs.size >= 4) { "fused_qkv_proj requires at least 4 inputs, got ${inputs.size}" }
                x = inputs[0]
                qWeight = inputs[1]
                kWeight = inputs[2]
                vWeight = inputs[3]
            }

            // Use transposeB flags propagated by LLMFusionPass from the original
            // matmul nodes. Falls back to shape-based detection.
            val transposeQ = params["transposeQ"] as? Boolean ?: false
            val transposeK = params["transposeK"] as? Boolean ?: false
            val transposeV = params["transposeV"] as? Boolean ?: false

            fun applyTranspose(w: Tensor<DType, Any>, shouldTranspose: Boolean): Tensor<DType, Any> {
                if (shouldTranspose) return ops.transpose(w)
                val xCols = x.shape[x.rank - 1]
                val wRows = w.shape[0]
                return if (wRows != xCols && w.rank == 2 && w.shape[1] == xCols) {
                    ops.transpose(w)
                } else {
                    w
                }
            }

            val q = ops.matmul(x, applyTranspose(qWeight, transposeQ))
            val k = ops.matmul(x, applyTranspose(kWeight, transposeK))
            val v = ops.matmul(x, applyTranspose(vWeight, transposeV))

            return listOf(q, k, v)
        }
    }
}
