package sk.ainet.models.moonshine

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.types.BF16
import sk.ainet.tape.Execution
import java.io.File
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Traces the Moonshine encoder transformer stack to StableHLO and asserts the
 * weights land as **bf16** in the MLIR — the Torq NPU requirement (fp32 weights
 * crash the torq compiler; see the demo's docs/torq-npu-weight-crash.md).
 * Dumps the MLIR so it can be compiled with the torq-fork iree-compile.
 */
class MoonshineEncoderMlirDumpTest {
    @Test
    fun dumpEncoderBf16Mlir() {
        val cfg = MoonshineConfig()
        val model = moonshineEncoder<BF16, Float>(cfg, BF16::class)

        // Encoder input = conv-frontend output [batch, frames, dim].
        val input = VoidOpsTensor(
            object : TensorData<BF16, Float> {
                override val shape = Shape(1, cfg.maxFrames, cfg.dim)
                override fun get(vararg indices: Int): Float = 0.0f
                override fun set(vararg indices: Int, value: Float) {}
            },
            BF16::class,
        )
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val tape = ctx.record {
            val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
            Execution.tapeStack.pushTape(ct)
            try {
                model.forward(input, this as ExecutionContext)
            } finally {
                Execution.tapeStack.popTape()
            }
        }.first
        val rawGraph = (tape as DefaultExecutionTape).toComputeGraph(synthesizeExternalInputs = true)
        // Unify edge dtypes: bf16-native traces record reductions/norms with a
        // stale FP32 dtype while their producers emit bf16, which would render the
        // same SSA value with two types. See DtypeForwardPropagationPass.
        val graph = sk.ainet.compile.opt.passes.DtypeForwardPropagationPass(targetFloatDtype = "BF16").apply(rawGraph).graph
        val mlir = sk.ainet.compile.hlo.toStableHlo(graph, "moonshine_encoder").content

        val out = File(System.getProperty("moonshineMlirOut") ?: "build/build-mlir/moonshine-encoder.mlir")
        out.parentFile?.mkdirs()
        out.writeText(mlir)
        println("WROTE_MLIR ${out.absolutePath} (${mlir.lines().size} lines)")

        assertTrue(mlir.contains("bf16"), "encoder MLIR must carry bf16 weights for the Torq NPU path")
    }
}
