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
 * Phase A stub test: traces the Moonshine DECODER stack to StableHLO with TWO graph
 * inputs — `inputs_embeds` `[·, seq, dim]` and encoder memory `[·, frames, dim]` — and
 * asserts the wiring compiles (self-attn → cross-attn(memory) → GELU MLP → norm → lm_head)
 * and the weights land as the configured dtype (bf16 by default, the Torq NPU requirement).
 *
 * This proves cross-attention threading only; numeric validation (weights, Phase B) and
 * the two KV-cached decode graphs (Phase D) are not covered here.
 *
 * Phase D fix (landed): at DEC_SEQ=1 (the real one-token decode shape) the self-attention
 * used to take [MultiHeadAttention]'s fused-decode fast path, whose buffer-direct ops don't
 * record on the trace tape — leaking per-layer self K/V as dangling `[8,1,36]` outputs. That
 * path is now guarded by `!ctx.isRecording`, so tracing falls through to the symbolic SDPA
 * path and seq=1 exports one clean logits output. This test asserts that single output.
 *
 * Env knobs (mirror the encoder dump test):
 *   DEC_LAYERS   decoder layer count      (default cfg.decoderLayers = 6)
 *   DEC_SEQ      decoder query length     (default 1 — the real prefill/decode shape)
 *   DEC_FRAMES   encoder memory frames    (default cfg.maxFrames = 165; board layout = 207)
 *   DEC_DTYPE=FP32   trace an f32 decoder (numeric bring-up); default bf16
 *   DEC_SKIP_DTYPE=1 leave the trace's mixed precision intact
 *   MOONSHINE_MLIR_OUT  output path       (default build/build-mlir/moonshine-decoder.mlir)
 */
class MoonshineDecoderMlirDumpTest {
    @Test
    fun dumpDecoderBf16Mlir() {
        val cfg = MoonshineConfig(
            decoderLayers = System.getProperty("decLayers")?.toInt()
                ?: (System.getenv("DEC_LAYERS")?.toInt() ?: 6),
        )
        val seq = System.getenv("DEC_SEQ")?.toInt() ?: 1
        val frames = System.getenv("DEC_FRAMES")?.toInt() ?: cfg.maxFrames
        val useF32 = System.getenv("DEC_DTYPE") == "FP32"

        val mlir = if (useF32) {
            traceDecoder(cfg, seq, frames, FP32::class, "FP32")
        } else {
            traceDecoder(cfg, seq, frames, BF16::class, "BF16")
        }

        val out = File(
            System.getProperty("moonshineMlirOut")
                ?: (System.getenv("MOONSHINE_MLIR_OUT") ?: "build/build-mlir/moonshine-decoder.mlir"),
        )
        out.parentFile?.mkdirs()
        out.writeText(mlir)
        println("WROTE_MLIR ${out.absolutePath} (${mlir.lines().size} lines)")

        assertTrue(mlir.contains(if (useF32) "f32" else "bf16"), "decoder MLIR must carry $useF32-consistent weights")
        assertTrue(mlir.contains("moonshine_decoder"), "decoder MLIR must name the entry function")
        // The fused-decode guard means seq=1 traces cleanly: exactly ONE result (logits),
        // no dangling per-head self K/V outputs.
        val returns = Regex("""\)\s*->\s*\(([^)]*)\)""").find(mlir)?.groupValues?.get(1) ?: ""
        val nResults = returns.split(",").count { it.contains("tensor<") }
        assertTrue(nResults == 1, "seq=$seq decoder must have a single logits output, got $nResults: $returns")
    }

    private fun <T : DType> traceDecoder(
        cfg: MoonshineConfig,
        seq: Int,
        frames: Int,
        dtypeClass: KClass<T>,
        floatDtype: String,
    ): String {
        val model = moonshineDecoder<T, Float>(cfg, dtypeClass)
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
        // Same HW-agnostic edge-dtype unification as the encoder dump; Torq tiling/bf16
        // passes are applied downstream by the target build, not here.
        val dtypeTarget = if (System.getenv("DEC_SKIP_DTYPE") == "1") null else floatDtype
        val graph = sk.ainet.compile.opt.passes.DtypeForwardPropagationPass(targetFloatDtype = dtypeTarget)
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
