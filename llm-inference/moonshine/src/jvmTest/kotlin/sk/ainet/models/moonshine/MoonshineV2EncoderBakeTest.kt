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
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Bake the REAL Moonshine **v2** tiny-streaming ENCODER + ADAPTER weights (from
 * `scripts/convert_moonshine_v2_weights.py`) and confirm every DSL param maps to a baked tensor with a
 * matching shape — the name/shape half that catches almost all mapping bugs — then trace with
 * `embedConstants=true` and assert the weights fold to `stablehlo.constant`, leaving only the data inputs.
 *
 * Gated on `MOONSHINE_V2_CHECKPOINT=<dir of per-tensor .bin>`; a no-op skip when unset so CI stays green
 * without the checkpoint. Numeric parity vs the ONNX reference needs onnxruntime and is tracked separately.
 */
class MoonshineV2EncoderBakeTest {
    @Test
    fun bakeRealV2EncoderAndAdapter() {
        val dir = System.getenv("MOONSHINE_V2_CHECKPOINT") ?: run {
            println("SKIP bakeRealV2EncoderAndAdapter: set MOONSHINE_V2_CHECKPOINT to the .bin weights dir")
            return
        }
        val cfg = MoonshineV2Config()
        val src = DecDirBinWeightSource(dir)

        // ---- encoder: every param maps + shape matches, then folds to constants ----
        val enc = moonshineV2Encoder<FP32, Float>(cfg, FP32::class)
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val bakedEnc = bakeV2EncoderWeights(enc, src, FP32::class, ctx as ExecutionContext)
        assertEquals(countParams(enc), bakedEnc, "every v2 encoder param must bake")
        println("BAKED $bakedEnc v2 encoder params from $dir")

        val frames = System.getenv("ENC_FRAMES")?.toInt() ?: 64
        val features = voidF32(Shape(1, frames, cfg.dim))
        val tape = ctx.record {
            val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
            Execution.tapeStack.pushTape(ct)
            try {
                enc.forward(features, this as ExecutionContext)
            } finally {
                Execution.tapeStack.popTape()
            }
        }.first
        val graph = (tape as DefaultExecutionTape).toComputeGraph(
            synthesizeExternalInputs = true,
            embedConstants = true,
        )
        val mlir = sk.ainet.compile.hlo.toStableHlo(graph, "moonshine_v2_encoder").content
        val argCount = Regex("""%arg\d+""").findAll(mlir.substringBefore(") ->")).map { it.value }.toSet().size
        println("v2 encoder MLIR entry args after embedConstants: $argCount")
        assertTrue(mlir.contains("stablehlo.constant"), "baked weights should fold to constants")
        assertEquals(1, argCount, "only the features input should remain as an arg")

        // ---- adapter: the single learned pos_embed table bakes (maxFrames from the checkpoint) ----
        val maxFrames = System.getenv("ADP_MAX_FRAMES")?.toInt() ?: 4096
        val adapter = MoonshineV2Adapter<FP32, Float>(cfg, maxFrames = maxFrames, dtype = FP32::class)
        val ctx2 = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val bakedAda = bakeV2EncoderWeights(adapter, src, FP32::class, ctx2 as ExecutionContext)
        assertEquals(countParams(adapter), bakedAda, "every v2 adapter param must bake")
        println("BAKED $bakedAda v2 adapter params (pos_embed [$maxFrames, ${cfg.dim}])")
    }

    private fun countParams(m: sk.ainet.lang.nn.Module<*, *>): Int =
        m.params.size + m.modules.sumOf { countParams(it) }

    private fun voidF32(shape: Shape): VoidOpsTensor<FP32, Float> =
        VoidOpsTensor(
            object : TensorData<FP32, Float> {
                override val shape = shape
                override fun get(vararg indices: Int): Float = 0.0f
                override fun set(vararg indices: Int, value: Float) {}
            },
            FP32::class,
        )
}
