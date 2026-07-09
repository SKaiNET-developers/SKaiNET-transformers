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
import sk.ainet.tape.Execution
import java.io.File
import kotlin.reflect.KClass
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Phase D — the `decoder` PREFILL KV-cache export graph.
 *
 * Traces [MoonshineDecoderModel.forwardPrefill] and asserts it emits the board's `decoder`
 * contract: logits `[1, seq, 32768]` + per layer `self_k/v [1, nHeads, seq, headDim]` and
 * `cross_k/v [1, nHeads, frames, headDim]` (6 layers → 24 KV outputs, 25 total). The returned
 * KV tensors, being unconsumed, surface as graph outputs — enabled by the fused-decode trace
 * fix (seq==1 now traces the symbolic SDPA path, so K/V are real graph values not buffer reads).
 *
 * Env: DEC_FRAMES (encoder memory frames; default 207 = board), DEC_SEQ (default 1 = START token),
 * DEC_DTYPE=FP32, MOONSHINE_MLIR_OUT.
 */
class MoonshineDecoderPrefillMlirDumpTest {
    @Test
    fun dumpPrefillKvGraph() {
        val cfg = MoonshineConfig()
        val seq = System.getenv("DEC_SEQ")?.toInt() ?: 1
        val frames = System.getenv("DEC_FRAMES")?.toInt() ?: 207
        val useF32 = System.getenv("DEC_DTYPE") == "FP32"

        val mlir = if (useF32) trace(cfg, seq, frames, sk.ainet.lang.types.FP32::class, "FP32")
        else trace(cfg, seq, frames, BF16::class, "BF16")

        val out = File(System.getenv("MOONSHINE_MLIR_OUT") ?: "build/build-mlir/moonshine-decoder-prefill.mlir")
        out.parentFile?.mkdirs()
        out.writeText(mlir)
        println("WROTE_MLIR ${out.absolutePath} (${mlir.lines().size} lines)")

        val returns = Regex("""\)\s*->\s*\(([^)]*)\)""").find(mlir)?.groupValues?.get(1) ?: ""
        val results = returns.split(",").map { it.trim() }.filter { it.contains("tensor<") }
        val nHeads = cfg.nHeads; val hd = cfg.headDim
        val logits = results.count { it.contains("x${cfg.vocabSize}x") }
        val selfKv = results.count { it.contains("x${nHeads}x${seq}x${hd}x") }
        val crossKv = results.count { it.contains("x${nHeads}x${frames}x${hd}x") }
        println("prefill outputs: total=${results.size} logits=$logits selfKV=$selfKv crossKV=$crossKv")

        assertEquals(1 + 4 * cfg.decoderLayers, results.size, "expect logits + 4·layers KV outputs")
        assertEquals(1, logits, "one logits output [1,$seq,${cfg.vocabSize}]")
        assertEquals(2 * cfg.decoderLayers, selfKv, "self k+v per layer, [1,$nHeads,$seq,$hd]")
        assertEquals(2 * cfg.decoderLayers, crossKv, "cross k+v per layer, [1,$nHeads,$frames,$hd]")
        assertTrue(mlir.contains("bf16") || useF32, "prefill weights should be bf16 by default")
    }

    private fun <T : DType> trace(cfg: MoonshineConfig, seq: Int, frames: Int, dtype: KClass<T>, f: String): String {
        val model = moonshineDecoder<T, Float>(cfg, dtype)
        val embeds = voidTensor(Shape(1, seq, cfg.dim), dtype)
        val memory = voidTensor(Shape(1, frames, cfg.dim), dtype)
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val tape = ctx.record {
            val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
            Execution.tapeStack.pushTape(ct)
            try { model.forwardPrefill(embeds, memory, this as ExecutionContext) } finally { Execution.tapeStack.popTape() }
        }.first
        val graph = (tape as DefaultExecutionTape).toComputeGraph(synthesizeExternalInputs = true)
        val dt = if (System.getenv("DEC_SKIP_DTYPE") == "1") null else f
        val g = sk.ainet.compile.opt.passes.DtypeForwardPropagationPass(targetFloatDtype = dt).apply(graph).graph
        return sk.ainet.compile.hlo.toStableHlo(g, "moonshine_decoder_prefill").content
    }

    private fun <T : DType> voidTensor(shape: Shape, dtype: KClass<T>): VoidOpsTensor<T, Float> =
        VoidOpsTensor(object : TensorData<T, Float> {
            override val shape = shape
            override fun get(vararg indices: Int): Float = 0.0f
            override fun set(vararg indices: Int, value: Float) {}
        }, dtype)
}
