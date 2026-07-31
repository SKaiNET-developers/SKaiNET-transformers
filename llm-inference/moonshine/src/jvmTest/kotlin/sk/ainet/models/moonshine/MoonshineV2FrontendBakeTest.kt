package sk.ainet.models.moonshine

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
import kotlin.test.assertTrue

/**
 * Bake the real v2 frontend weights into [moonshineV2Frontend], trace `forward(audio[1,samples])` with
 * `embedConstants=true` (weights fold to `stablehlo.constant`, leaving only the audio input), and write the
 * StableHLO to `MOONSHINE_V2_FE_MLIR_OUT` — the self-compiled DSL frontend graph (`@moonshine_v2_frontend`),
 * replacing the vendor `frontend.onnx`. Gated on `FE_DIR` (the fe_* .bin dir); `FE_SAMPLES` sets the fixed
 * audio length (multiple of 80; default 80000 = 5 s @ 16 kHz).
 */
class MoonshineV2FrontendBakeTest {
    @Test
    fun emitFrontendMlir() {
        val dir = System.getenv("FE_DIR") ?: run { println("SKIP emitFrontendMlir: set FE_DIR"); return }
        val out = System.getenv("MOONSHINE_V2_FE_MLIR_OUT") ?: run { println("SKIP emitFrontendMlir: set MOONSHINE_V2_FE_MLIR_OUT"); return }
        val samples = System.getenv("FE_SAMPLES")?.toInt() ?: 80000

        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val fe = moonshineV2Frontend<FP32, Float>(FP32::class)
        val baked = bakeMoonshineWeights(fe, DecDirBinWeightSource(dir), { DecMap(it, false) }, FP32::class, ctx as ExecutionContext)
        println("baked $baked v2 frontend params for export")

        val audio = voidF32(Shape(1, samples))
        val tape = ctx.record {
            val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
            Execution.tapeStack.pushTape(ct)
            try { fe.forward(audio, this as ExecutionContext) } finally { Execution.tapeStack.popTape() }
        }.first
        val graph = (tape as DefaultExecutionTape).toComputeGraph(synthesizeExternalInputs = true, embedConstants = true)
        // createExtended (not toStableHlo/createBasic) — the conv1d converter lives only in the extended registry.
        val mlir = sk.ainet.compile.hlo.StableHloConverterFactory.createExtended().convert(graph, "moonshine_v2_frontend").content
        assertTrue(mlir.contains("stablehlo.constant"), "baked frontend weights should fold to constants")
        java.io.File(out).apply { parentFile?.mkdirs() }.writeText(mlir)
        println("WROTE_MLIR $out (${mlir.lines().size} lines)")
    }

    private fun voidF32(shape: Shape): VoidOpsTensor<FP32, Float> = VoidOpsTensor(
        object : TensorData<FP32, Float> {
            override val shape: Shape = shape
            override fun get(vararg indices: Int): Float = 0.0f
            override fun set(vararg indices: Int, value: Float) {}
        },
        FP32::class,
    )
}
