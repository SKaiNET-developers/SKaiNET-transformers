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
 * Traces the Moonshine **v2** DECODER (`moonshineV2Decoder`, the v1 decode architecture at v2 tiny dims) to
 * StableHLO — the P6 structure milestone for the last v2 model piece. Proves the two-input cross-attention
 * forward (decoder embeds `[·,seq,dim]` + adapter memory `[·,frames,dim]`) builds and lowers with the real
 * v2 config (dim=320, 6 layers, 8 heads, head_dim=40, ffn=1280, rotary_dim=32). Portable FP32; numeric
 * validation vs the ONNX (RoPE base/factor, tie) is a follow-up.
 *
 * Uses `DEC_SEQ ≥ 2` (default 2) so the trace takes the symbolic SDPA path — at seq=1 the self-attention's
 * fused-decode fast path leaks per-layer K/V as dangling outputs (see [MoonshineDecoderMlirDumpTest]).
 */
class MoonshineV2DecoderMlirDumpTest {
    @Test
    fun dumpV2DecoderMlir() {
        val cfg = MoonshineV2Config(
            decoderLayers = System.getenv("DEC_LAYERS")?.toInt() ?: 6,
        )
        val seq = System.getenv("DEC_SEQ")?.toInt() ?: 2
        val frames = System.getenv("DEC_FRAMES")?.toInt() ?: 64
        val mlir = traceV2Decoder(cfg, seq, frames, FP32::class)

        val out = File(System.getenv("MOONSHINE_V2_DEC_MLIR_OUT") ?: "build/build-mlir/moonshine-v2-decoder.mlir")
        out.parentFile?.mkdirs()
        out.writeText(mlir)
        println("WROTE_MLIR ${out.absolutePath} (${mlir.lines().size} lines)")

        assertTrue(mlir.isNotBlank(), "v2 decoder MLIR must be non-empty")
        assertTrue(mlir.contains("moonshine_decoder"), "must emit the decoder function")
        assertTrue(mlir.contains("f32"), "portable FP32 trace must carry f32 tensors")
        // seq≥2 traces cleanly → a single logits output (no dangling per-head K/V).
        val returns = Regex("""\)\s*->\s*\(([^)]*)\)""").find(mlir)?.groupValues?.get(1) ?: ""
        val nResults = returns.split(",").count { it.contains("tensor<") }
        assertTrue(nResults == 1, "seq=$seq decoder must have a single logits output, got $nResults: $returns")
    }

    private fun <T : DType> traceV2Decoder(
        cfg: MoonshineV2Config,
        seq: Int,
        frames: Int,
        dtypeClass: KClass<T>,
    ): String {
        val model = moonshineV2Decoder<T, Float>(cfg, dtypeClass)
        val embeds = voidTensor(Shape(1, seq, cfg.dim), dtypeClass)
        val memory = voidTensor(Shape(1, frames, cfg.dim), dtypeClass)

        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val tape = ctx.record {
            val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
            Execution.tapeStack.pushTape(ct)
            try {
                model.forward(embeds, memory, this as ExecutionContext)
            } finally {
                Execution.tapeStack.popTape()
            }
        }.first
        val rawGraph = (tape as DefaultExecutionTape).toComputeGraph(synthesizeExternalInputs = true)
        val graph = sk.ainet.compile.opt.passes.DtypeForwardPropagationPass(targetFloatDtype = "FP32")
            .apply(rawGraph).graph
        return sk.ainet.compile.hlo.toStableHlo(graph, "moonshine_decoder").content
    }

    private fun <T : DType> voidTensor(shape: Shape, dtypeClass: KClass<T>): VoidOpsTensor<T, Float> =
        VoidOpsTensor(
            object : TensorData<T, Float> {
                override val shape = shape
                override fun get(vararg indices: Int): Float = 0.0f
                override fun set(vararg indices: Int, value: Float) {}
            },
            dtypeClass,
        )
}
