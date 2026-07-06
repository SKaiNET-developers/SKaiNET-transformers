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
 * Phase B verification: bake the REAL moonshine-tiny decoder weights into
 * [MoonshineDecoderModel] and confirm every DSL param maps to a checkpoint tensor with a
 * matching shape (the name/shape half of Phase B — what catches almost all mapping bugs).
 * Then trace with `embedConstants=true` and assert the weights fold to `stablehlo.constant`,
 * leaving only the two data inputs (`inputs_embeds` + encoder memory).
 *
 * Gated on `DEC_CHECKPOINT=<dir of per-tensor .bin>` (from `convert_moonshine_weights.py`);
 * the test is a no-op skip when it is unset so CI without the checkpoint stays green.
 *
 * NOTE: the full Phase-B "done means" — f32 CPU cos ≈ 1.0 vs the HF decoder on injected
 * encoder memory — additionally needs a `transformers` reference and is tracked separately.
 */
class MoonshineDecoderBakeTest {
    @Test
    fun bakeRealDecoderWeights() {
        val dir = System.getenv("DEC_CHECKPOINT") ?: run {
            println("SKIP bakeRealDecoderWeights: set DEC_CHECKPOINT to the .bin weights dir")
            return
        }
        val cfg = MoonshineConfig()
        val model = moonshineDecoder<FP32, Float>(cfg, FP32::class)

        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val baked = bakeDecoderWeights(model, DecDirBinWeightSource(dir), FP32::class, ctx as ExecutionContext)

        // Every param baked (bakeDecoderWeights fail-fasts on any missing tensor / shape mismatch).
        val paramCount = countParams(model)
        assertEquals(paramCount, baked, "every decoder param must bake")
        println("BAKED $baked decoder params from $dir")

        // Trace with the baked constants folded in; only the 2 data inputs should remain.
        val embeds = voidF32(Shape(1, 2, cfg.dim))   // seq=2 → general SDPA path (see decoder-dump caveat)
        val memory = voidF32(Shape(1, cfg.maxFrames, cfg.dim))
        val tape = ctx.record {
            val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
            Execution.tapeStack.pushTape(ct)
            try {
                model.forward(embeds, memory, this as ExecutionContext)
            } finally {
                Execution.tapeStack.popTape()
            }
        }.first
        val graph = (tape as DefaultExecutionTape).toComputeGraph(
            synthesizeExternalInputs = true,
            embedConstants = true,
        )
        val mlir = sk.ainet.compile.hlo.toStableHlo(graph, "moonshine_decoder").content
        val argCount = Regex("""%arg\d+""").findAll(mlir.substringBefore(") ->")).map { it.value }.toSet().size
        println("decoder MLIR entry args after embedConstants: $argCount")
        assertTrue(mlir.contains("stablehlo.constant"), "baked weights should fold to constants")
        assertEquals(2, argCount, "only inputs_embeds + encoder memory should remain as args")
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
