package sk.ainet.models.moonshine

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DType
import sk.ainet.tape.Execution
import java.io.File
import kotlin.reflect.KClass
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Phase D — the `decoder_with_past` KV-cache export graph.
 *
 * Traces [MoonshineDecoderModel.forwardWithPast] and asserts the board contract: inputs = one token
 * `[1,1,dim]` + per-layer self cache in `[1,nHeads,pos,headDim]` + cross cache in `[1,nHeads,frames,headDim]`;
 * outputs = logits `[1,1,vocab]` + per-layer extended self cache `[1,nHeads,pos+1,headDim]` (6 layers →
 * 12 self-KV outputs, 13 total; cross cache is passed through, not re-emitted).
 *
 * RoPE position is a RUNTIME input: the per-step cos/sin tables `[1, headDim]` enter as graph inputs
 * (host-built via `buildInterleavedCosSin`, mirroring host-side token embedding) — so ONE vmfb serves
 * every position, no in-graph gather. The graph therefore also has `cos`/`sin` `[1,headDim]` inputs.
 *
 * Env: DEC_PAST (incoming self-cache length, default 1), DEC_FRAMES (default 207), DEC_DTYPE=FP32.
 */
class MoonshineDecoderWithPastMlirDumpTest {
    @Test
    fun dumpWithPastKvGraph() {
        val cfg = MoonshineConfig()
        val past = System.getenv("DEC_PAST")?.toInt() ?: 1
        val frames = System.getenv("DEC_FRAMES")?.toInt() ?: 207
        val useF32 = System.getenv("DEC_DTYPE") == "FP32"

        val mlir = if (useF32) trace(cfg, past, frames, sk.ainet.lang.types.FP32::class, "FP32")
        else trace(cfg, past, frames, BF16::class, "BF16")

        val out = File(System.getenv("MOONSHINE_MLIR_OUT") ?: "build/build-mlir/moonshine-decoder-with-past.mlir")
        out.parentFile?.mkdirs()
        out.writeText(mlir)
        println("WROTE_MLIR ${out.absolutePath} (${mlir.lines().size} lines)")

        val returns = Regex("""\)\s*->\s*\(([^)]*)\)""").find(mlir)?.groupValues?.get(1) ?: ""
        val results = returns.split(",").map { it.trim() }.filter { it.contains("tensor<") }
        val nHeads = cfg.nHeads; val hd = cfg.headDim
        val logits = results.count { it.contains("x${cfg.vocabSize}x") }
        val selfOut = results.count { it.contains("x${nHeads}x${past + 1}x${hd}x") }
        println("with_past outputs: total=${results.size} logits=$logits extendedSelfKV=$selfOut")

        assertEquals(1 + 2 * cfg.decoderLayers, results.size, "expect logits + 2·layers extended self-KV")
        assertEquals(1, logits, "one logits output")
        assertEquals(2 * cfg.decoderLayers, selfOut, "extended self k+v per layer, [1,$nHeads,${past + 1},$hd]")
        // cross-cache inputs must be present (consumed, not re-projected): [1,nHeads,frames,headDim].
        assertTrue(mlir.contains("x${nHeads}x${frames}x${hd}x"), "cross cache [1,$nHeads,$frames,$hd] must be a graph input")
    }

    private fun <T : DType> trace(cfg: MoonshineConfig, past: Int, frames: Int, dtype: KClass<T>, f: String): String {
        val model = moonshineDecoder<T, Float>(cfg, dtype)
        val token = void(Shape(1, 1, cfg.dim), dtype)
        val cos = void(Shape(1, cfg.headDim), dtype) // runtime-position RoPE tables, graph inputs
        val sin = void(Shape(1, cfg.headDim), dtype)
        val selfK = List(cfg.decoderLayers) { void(Shape(1, cfg.nHeads, past, cfg.headDim), dtype) }
        val selfV = List(cfg.decoderLayers) { void(Shape(1, cfg.nHeads, past, cfg.headDim), dtype) }
        val crossK = List(cfg.decoderLayers) { void(Shape(1, cfg.nHeads, frames, cfg.headDim), dtype) }
        val crossV = List(cfg.decoderLayers) { void(Shape(1, cfg.nHeads, frames, cfg.headDim), dtype) }
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val tape = ctx.record {
            val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
            Execution.tapeStack.pushTape(ct)
            try {
                model.forwardWithPast(token, cos, sin, selfK, selfV, crossK, crossV, this as ExecutionContext)
            } finally { Execution.tapeStack.popTape() }
        }.first
        val graph = (tape as DefaultExecutionTape).toComputeGraph(synthesizeExternalInputs = true)
        val dt = if (System.getenv("DEC_SKIP_DTYPE") == "1") null else f
        val g = sk.ainet.compile.opt.passes.DtypeForwardPropagationPass(targetFloatDtype = dt).apply(graph).graph
        return sk.ainet.compile.hlo.toStableHlo(g, "moonshine_decoder_with_past").content
    }

    private fun <T : DType> void(shape: Shape, dtype: KClass<T>): Tensor<T, Float> =
        VoidOpsTensor(object : TensorData<T, Float> {
            override val shape = shape
            override fun get(vararg indices: Int): Float = 0.0f
            override fun set(vararg indices: Int, value: Float) {}
        }, dtype)
}
