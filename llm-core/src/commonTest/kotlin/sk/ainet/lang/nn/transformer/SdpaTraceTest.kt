package sk.ainet.lang.nn.transformer

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.types.FP32
import sk.ainet.tape.Execution
import kotlin.test.Test

/**
 * Verifies the dtype fix: a MultiHeadAttention built with `dtype = FP32::class`
 * traces to a ComputeGraph WITHOUT loaded weights (no more "Unsupported dtype:
 * Object"), and reports whether scaledDotProductAttention is an atomic op node
 * or decomposes into matmul/softmax/transpose — which decides where its
 * StableHLO converter lives.
 */
class SdpaTraceTest {
    @Test
    fun traceStandaloneAttention() {
        val dim = 64
        val seqLen = 4
        val mha = MultiHeadAttention<FP32, Float>(
            dim = dim,
            nHeads = 2,
            nKVHeads = 1,
            causal = true,
            dtype = FP32::class,
        )

        val input = VoidOpsTensor(
            object : TensorData<FP32, Float> {
                override val shape = Shape(seqLen, dim)
                override fun get(vararg indices: Int): Float = 0.0f
                override fun set(vararg indices: Int, value: Float) {}
            },
            FP32::class,
        )

        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val (tape, _) = ctx.record {
            val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
            Execution.tapeStack.pushTape(ct)
            try {
                mha.forward(input, this as ExecutionContext)
            } finally {
                Execution.tapeStack.popTape()
            }
        }
        val graph = (tape as DefaultExecutionTape).toComputeGraph()
        val opTypes = graph.nodes.map { it.operationName }
        println("=== MHA traced: ${graph.nodes.size} op nodes ===")
        opTypes.groupingBy { it }.eachCount().entries.sortedBy { it.key }.forEach { (k, v) -> println("  $k x$v") }
        println("SDPA atomic? -> " + opTypes.any { it.contains("ScaledDot", true) || it.contains("Attention", true) })
    }
}
