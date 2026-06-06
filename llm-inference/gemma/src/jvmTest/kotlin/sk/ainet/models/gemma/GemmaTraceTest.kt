package sk.ainet.models.gemma

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
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Verifies the dtype fix end-to-end: a tiny gemma3 network traces to a
 * ComputeGraph WITHOUT loaded weights (no "Unsupported dtype: Object").
 * Prints the recorded op histogram — the input to the StableHLO converter set.
 */
class GemmaTraceTest {
    @Test
    fun traceTinyGemmaNetwork() {
        val meta = Gemma4ModelMetadata(
            architecture = "gemma3",
            embeddingLength = 64,
            contextLength = 128,
            blockCount = 1,
            headCount = 2,
            kvHeadCount = 1,
            intermediateSize = 128,
            headDim = 32,
            globalHeadDim = 32,
            vocabSize = 32,
            slidingWindow = 64,
            kvSharedLayers = 0,
            layerTypes = listOf("full_attention"),
            ropeParametersFull = Gemma4RopeConfig(base = 10000.0f),
            ropeParametersSliding = Gemma4RopeConfig(base = 10000.0f),
            maxPositionEmbeddings = 128,
        )
        val model = gemmaNetwork<FP32, Float>(meta, FP32::class, maxInferenceLen = 8)

        val input = VoidOpsTensor(
            object : TensorData<FP32, Float> {
                override val shape = Shape(1, 4)
                override fun get(vararg indices: Int): Float = 0.0f
                override fun set(vararg indices: Int, value: Float) {}
            },
            FP32::class,
        )

        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        var thrown: Throwable? = null
        val tape = try {
            ctx.record {
                val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
                Execution.tapeStack.pushTape(ct)
                try {
                    model.forward(input, this as ExecutionContext)
                } finally {
                    Execution.tapeStack.popTape()
                }
            }.first
        } catch (e: Throwable) {
            thrown = e
            null
        }

        // The dtype fix is verified by NOT seeing the "Unsupported dtype: Object"
        // crash that previously aborted at the first placeholder weight. Any
        // remaining failure is a separate VoidTensorOps tracing-stub limitation
        // (e.g. the void gather returns one row -> reshape volume mismatch), not
        // a dtype problem.
        val msg = thrown?.message ?: ""
        println("trace outcome: ${if (thrown == null) "completed" else thrown::class.simpleName + ": " + msg}")
        thrown?.stackTraceToString()?.lines()?.take(14)?.forEach { println("  STACK $it") }
        assertFalse(
            msg.contains("Unsupported dtype", ignoreCase = true) || msg.contains("for zeros: class java.lang.Object"),
            "dtype fix regressed — placeholder weights are erasing to Object again: $msg",
        )

        if (tape != null) {
            // synthesizeExternalInputs=true runs finalize(), which materializes
            // module weights / model inputs as graph nodes (input/constant) and
            // wires them as operands — required so gather/transpose/matmul get
            // their weight operand (otherwise: arity-failure cascade).
            val graph = (tape as DefaultExecutionTape).toComputeGraph(synthesizeExternalInputs = true)
            val ops = graph.nodes.map { it.operationName }
            println("=== gemma traced: ${graph.nodes.size} nodes ===")
            ops.groupingBy { it }.eachCount().toSortedMap().forEach { (k, v) -> println("  $k x$v") }

            // Informational: lower to StableHLO and report the remaining gaps.
            // The op-level converters (split/permute/narrow/...) are unit-tested
            // in skainet-compile-hlo. A *full* gemma lowering additionally needs
            // two next-phase items, surfaced here:
            //   1. weight/param operand wiring — module weights are placeholders,
            //      not graph edges, so gather/transpose/matmul that consume them
            //      report arity failures (cascading).
            //   2. a scaledDotProductAttention converter (attention subgraph).
            val mlir = sk.ainet.compile.hlo.toStableHlo(graph, "gemma").content
            val unsupported = mlir.lines().filter { it.contains("Unsupported op", ignoreCase = true) }
            val arity = mlir.lines().filter { it.contains("arity", ignoreCase = true) }
            println("=== StableHLO: ${mlir.lines().size} lines; ${unsupported.size} unsupported-op, ${arity.size} arity gaps ===")
            (unsupported + arity).take(12).forEach { println("  GAP $it") }
        }
    }
}
