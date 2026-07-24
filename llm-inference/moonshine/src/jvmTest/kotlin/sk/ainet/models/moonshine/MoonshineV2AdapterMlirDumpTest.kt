package sk.ainet.models.moonshine

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import sk.ainet.tape.Execution
import java.io.File
import kotlin.reflect.KClass
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Traces the Moonshine **v2 adapter** (learned positional embedding add, no norm) to StableHLO — the P6 structure
 * milestone for the encoder→decoder bridge. Proves the two-input forward (encoder memory + frame positions)
 * builds and lowers. Portable FP32 by default; numeric validation vs a v2 reference needs a v2 checkpoint.
 */
class MoonshineV2AdapterMlirDumpTest {
    @Test
    fun dumpV2AdapterMlir() {
        val cfg = MoonshineV2Config()
        val frames = System.getenv("ADP_FRAMES")?.toInt() ?: 64
        val mlir = traceAdapter(cfg, frames, FP32::class)

        val out = File(System.getenv("MOONSHINE_V2_ADAPTER_MLIR_OUT") ?: "build/build-mlir/moonshine-v2-adapter.mlir")
        out.parentFile?.mkdirs()
        out.writeText(mlir)
        println("WROTE_MLIR ${out.absolutePath} (${mlir.lines().size} lines)")

        assertTrue(mlir.isNotBlank(), "v2 adapter MLIR must be non-empty")
        assertTrue(mlir.contains("moonshine_v2_adapter"), "must emit the v2 adapter function")
        assertTrue(mlir.contains("f32"), "portable FP32 trace must carry f32 tensors")
    }

    private fun <T : DType> traceAdapter(cfg: MoonshineV2Config, frames: Int, dtypeClass: KClass<T>): String {
        val adapter = MoonshineV2Adapter<T, Float>(cfg, maxFrames = frames, dtype = dtypeClass)
        fun void(shape: Shape) = VoidOpsTensor(
            object : TensorData<T, Float> {
                override val shape = shape
                override fun get(vararg indices: Int): Float = 0.0f
                override fun set(vararg indices: Int, value: Float) {}
            },
            dtypeClass,
        )
        val memory = void(Shape(1, frames, cfg.dim))   // position-free encoder output
        val positions = void(Shape(1, frames))         // float-encoded frame indices 0..frames-1
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val tape = ctx.record {
            val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
            Execution.tapeStack.pushTape(ct)
            try {
                adapter.forward(memory, positions, this as ExecutionContext)
            } finally {
                Execution.tapeStack.popTape()
            }
        }.first
        val rawGraph = (tape as DefaultExecutionTape).toComputeGraph(synthesizeExternalInputs = true)
        val graph = sk.ainet.compile.opt.passes.DtypeForwardPropagationPass(targetFloatDtype = "FP32").apply(rawGraph).graph
        return sk.ainet.compile.hlo.toStableHlo(graph, "moonshine_v2_adapter").content
    }
}
