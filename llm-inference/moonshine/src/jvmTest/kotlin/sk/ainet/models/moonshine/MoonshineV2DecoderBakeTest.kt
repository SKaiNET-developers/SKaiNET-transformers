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
 * Bake the REAL Moonshine **v2** decoder weights (`scripts/bake_moonshine_v2_decoder.py`) into
 * [moonshineV2Decoder] and confirm every DSL param maps with a matching shape (name/shape half of the check),
 * then trace `forward(embeds, memory)` with `embedConstants=true` so the weights fold to `stablehlo.constant`
 * — leaving only the two data inputs — and (optionally) write the MLIR for the self-compiled decoder vmfb.
 *
 * The decoder's numeric equivalence to the ONNX is proven separately (cos-sim 1.0,
 * `validate_moonshine_v2_decoder.py`); this locks the Kotlin bake + emits the compile input.
 *
 * Gated on `MOONSHINE_V2_DEC_CHECKPOINT=<dir of .bin>`; no-op skip when unset so CI stays green.
 * `MOONSHINE_V2_DEC_MLIR_OUT=<file>` writes the folded MLIR (for `iree-compile`).
 */
class MoonshineV2DecoderBakeTest {
    @Test
    fun bakeRealV2Decoder() {
        val dir = System.getenv("MOONSHINE_V2_DEC_CHECKPOINT") ?: run {
            println("SKIP bakeRealV2Decoder: set MOONSHINE_V2_DEC_CHECKPOINT to the .bin weights dir")
            return
        }
        val cfg = MoonshineV2Config()
        val dec = moonshineV2Decoder<FP32, Float>(cfg, FP32::class)
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val baked = bakeV2DecoderWeights(dec, DecDirBinWeightSource(dir), FP32::class, ctx as ExecutionContext)
        assertEquals(countParams(dec), baked, "every v2 decoder param must bake")
        println("BAKED $baked v2 decoder params from $dir")

        val seq = System.getenv("DEC_SEQ")?.toInt() ?: 2   // seq≥2 → clean single-logits trace
        val frames = System.getenv("DEC_FRAMES")?.toInt() ?: 64
        val embeds = voidF32(Shape(1, seq, cfg.dim))
        val memory = voidF32(Shape(1, frames, cfg.dim))
        val tape = ctx.record {
            val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
            Execution.tapeStack.pushTape(ct)
            try {
                dec.forward(embeds, memory, this as ExecutionContext)
            } finally {
                Execution.tapeStack.popTape()
            }
        }.first
        val graph = (tape as DefaultExecutionTape).toComputeGraph(
            synthesizeExternalInputs = true,
            embedConstants = true,
        )
        val mlir = sk.ainet.compile.hlo.toStableHlo(graph, "moonshine_v2_decoder").content
        val argCount = Regex("""%arg\d+""").findAll(mlir.substringBefore(") ->")).map { it.value }.toSet().size
        println("v2 decoder MLIR entry args after embedConstants: $argCount")
        assertTrue(mlir.contains("stablehlo.constant"), "baked weights should fold to constants")
        assertEquals(2, argCount, "only inputs_embeds + encoder memory should remain as args")

        System.getenv("MOONSHINE_V2_DEC_MLIR_OUT")?.let {
            val out = java.io.File(it); out.parentFile?.mkdirs(); out.writeText(mlir)
            println("WROTE_MLIR ${out.absolutePath} (${mlir.lines().size} lines)")
        }
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
