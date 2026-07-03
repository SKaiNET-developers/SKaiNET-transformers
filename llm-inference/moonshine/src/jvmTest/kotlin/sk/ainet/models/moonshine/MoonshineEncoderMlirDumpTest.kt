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
        val cfg = MoonshineConfig(encoderLayers = System.getProperty("encLayers")?.toInt() ?: 6)
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
        // Target-pluggable optimization: the Torq NPU backend registers its own
        // graph-lowering pass (attention head-tiling) via the core TargetOptimizers
        // registry — no HW knowledge in core or in the model. dagPipelineFor("torq")
        // applies whatever is plugged in for that target; it produces standard,
        // still-portable StableHLO (compiles on llvm-cpu too).
        sk.ainet.compile.opt.TargetOptimizers.registerDagPasses("torq") {
            listOf(TorqAttentionTilingPass(maxHeadsPerTile = 4))
        }
        val tiled = sk.ainet.compile.opt.dagPipelineFor("torq").optimize(rawGraph).graph
        // Then unify edge dtypes to bf16 (bf16-native traces record reductions/norms
        // as FP32 while producers emit bf16). HW-agnostic.
        val graph = sk.ainet.compile.opt.passes.DtypeForwardPropagationPass(targetFloatDtype = "BF16").apply(tiled).graph
        val mlir = sk.ainet.compile.hlo.toStableHlo(graph, "moonshine_encoder").content

        val out = File(System.getProperty("moonshineMlirOut") ?: "build/build-mlir/moonshine-encoder.mlir")
        out.parentFile?.mkdirs()
        out.writeText(mlir)
        println("WROTE_MLIR ${out.absolutePath} (${mlir.lines().size} lines)")

        assertTrue(mlir.contains("bf16"), "encoder MLIR must carry bf16 weights for the Torq NPU path")
    }
}
