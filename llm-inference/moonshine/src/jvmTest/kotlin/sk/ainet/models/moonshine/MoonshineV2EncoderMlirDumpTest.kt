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
 * Traces the Moonshine **v2** streaming encoder (position-free + bounded-lookahead sliding window) to
 * StableHLO. This is the P6 structure milestone: it proves the v2 encoder builds and lowers, and it
 * exercises the new transformer-core `rightContext` windowed-mask path end to end (the (16,4)/(16,0)
 * bands materialize an additive mask that the attention converter lowers before softmax).
 *
 * Portable by default (`FP32`) — the v2 encoder is dtype-portable like v1. Numeric validation vs a v2
 * reference is a follow-up (needs a released v2 checkpoint).
 */
class MoonshineV2EncoderMlirDumpTest {
    @Test
    fun dumpV2EncoderMlir() {
        val cfg = MoonshineV2Config(
            encoderLayers = System.getenv("ENC_LAYERS")?.toInt() ?: 2,
        )
        val frames = System.getenv("ENC_FRAMES")?.toInt() ?: 64
        val mlir = traceV2Encoder(cfg, frames, FP32::class, "FP32")

        val out = File(System.getenv("MOONSHINE_V2_MLIR_OUT") ?: "build/build-mlir/moonshine-v2-encoder.mlir")
        out.parentFile?.mkdirs()
        out.writeText(mlir)
        println("WROTE_MLIR ${out.absolutePath} (${mlir.lines().size} lines)")

        // Structure milestone: valid StableHLO for the v2 encoder function, in the portable dtype.
        assertTrue(mlir.isNotBlank(), "v2 encoder MLIR must be non-empty")
        assertTrue(mlir.contains("moonshine_v2_encoder"), "must emit the v2 encoder function")
        assertTrue(mlir.contains("f32"), "portable FP32 trace must carry f32 tensors")
    }

    private fun <T : DType> traceV2Encoder(
        cfg: MoonshineV2Config,
        frames: Int,
        dtypeClass: KClass<T>,
        floatDtype: String,
    ): String {
        val model = moonshineV2Encoder<T, Float>(cfg, dtypeClass)
        val input = VoidOpsTensor(
            object : TensorData<T, Float> {
                override val shape = Shape(1, frames, cfg.dim)
                override fun get(vararg indices: Int): Float = 0.0f
                override fun set(vararg indices: Int, value: Float) {}
            },
            dtypeClass,
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
        val graph = sk.ainet.compile.opt.passes.DtypeForwardPropagationPass(targetFloatDtype = floatDtype).apply(rawGraph).graph
        return sk.ainet.compile.hlo.toStableHlo(graph, "moonshine_v2_encoder").content
    }
}
