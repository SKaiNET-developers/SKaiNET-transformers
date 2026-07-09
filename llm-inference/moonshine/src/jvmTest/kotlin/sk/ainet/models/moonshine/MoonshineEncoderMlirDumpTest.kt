package sk.ainet.models.moonshine

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import sk.ainet.tape.Execution
import java.io.File
import kotlin.reflect.KClass
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
        val cfg = MoonshineConfig(encoderLayers = System.getProperty("encLayers")?.toInt() ?: (System.getenv("ENC_LAYERS")?.toInt() ?: 6))
        // ENC_DTYPE=FP32 traces an f32 encoder (numeric validation against the reference,
        // isolating model conventions from bf16 rounding/overflow); default is bf16 for the
        // Torq NPU path. The dtype flows explicitly through the DSL — no forced conversions.
        val useF32 = System.getenv("ENC_DTYPE") == "FP32"
        val mlir = if (useF32) traceEncoder(cfg, FP32::class, "FP32") else traceEncoder(cfg, BF16::class, "BF16")

        val out = File(System.getProperty("moonshineMlirOut") ?: (System.getenv("MOONSHINE_MLIR_OUT") ?: "build/build-mlir/moonshine-encoder.mlir"))
        out.parentFile?.mkdirs()
        out.writeText(mlir)
        println("WROTE_MLIR ${out.absolutePath} (${mlir.lines().size} lines)")

        assertTrue(mlir.contains(if (useF32) "f32" else "bf16"), "encoder MLIR must carry $useF32-consistent weights")
    }

    private fun <T : DType> traceEncoder(cfg: MoonshineConfig, dtypeClass: KClass<T>, floatDtype: String): String {
        val model = moonshineEncoder<T, Float>(cfg, dtypeClass)
        // Encoder input = conv-frontend output [batch, frames, dim].
        val inShape = if (System.getenv("ENC_2D") == "1") Shape(cfg.maxFrames, cfg.dim) else Shape(1, cfg.maxFrames, cfg.dim)
        val input = VoidOpsTensor(
            object : TensorData<T, Float> {
                override val shape = inShape
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
        // Emit canonical, HW-agnostic StableHLO. Target-specific DAG passes (e.g. the Torq
        // attention/FFN tiling and bf16-trace numeric workarounds) are NOT applied here — they
        // live in the vendor plugin (skainet-embedded-vendors :synaptics-torq) and are applied
        // downstream by the app/build tool that has chosen the target. The model stays portable.
        //
        // Edge-dtype unification (bf16-native traces record reductions/norms as FP32 while
        // producers emit the model dtype) is a core, HW-agnostic pass. ENC_SKIP_DTYPE=1 leaves
        // the trace's mixed precision intact.
        val dtypeTarget = if (System.getenv("ENC_SKIP_DTYPE") == "1") null else floatDtype
        val graph = sk.ainet.compile.opt.passes.DtypeForwardPropagationPass(targetFloatDtype = dtypeTarget).apply(rawGraph).graph
        return sk.ainet.compile.hlo.toStableHlo(graph, "moonshine_encoder").content
    }
}
