package sk.ainet.models.gemma

import org.junit.jupiter.api.Tag
import kotlinx.coroutines.runBlocking
import sk.ainet.compile.hlo.ConstantMaterializationPolicy
import sk.ainet.compile.hlo.StableHloConverterFactory
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.tensor.storage.BufferHandle
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.transformer.MultiHeadAttention
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.types.FP32
import sk.ainet.tape.Execution
import java.io.File
import kotlin.test.Test

/**
 * Bake the REAL FunctionGemma-270M to a self-contained IREE artifact:
 * fromWeights (real topology + real weights) -> trace (embedConstants=true,
 * resolves the bound weights) -> StableHLO with ExternalAlways so every weight
 * becomes a `util.global.load` + ExternalParameterRef -> IrpaWriter writes the
 * .irpa. The emitted vmfb then takes ONLY the token input; weights resolve from
 * the archive via `iree-run-module --parameters=model=gemma.irpa`. No 361-arg
 * mapping, no host-side RoPE reproduction. Boxing-free (FloatArray) path.
 */
@Tag("integration")
class RealGemmaBakeIrpaTest {
    @Test
    fun bakeRealGemmaToIrpa() = runBlocking {
        val path = "/home/miso/projects/coral/sl2610-voice-cc-kt/models/functiongemma-physical-ai-v10-Q5_K_M.gguf"
        val ctx = DirectCpuExecutionContext.create()
        val weights = Gemma4WeightLoader(
            randomAccessProvider = { JvmRandomAccessSource.open(path) },
            quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
        ).loadToMapStreaming<FP32, Float>(ctx, FP32::class)
        // gemma3 uses FULL rotary; the gguf omits rope.partial_rotary_factor so
        // the loader defaulted to 0.25 (a Gemma-4 convention) which mis-rotates
        // global layers. Override via -DpartialRotary (default 1.0).
        val partial = (System.getProperty("partialRotary") ?: "1.0").toFloat()
        val patched = weights.copy(
            metadata = weights.metadata.copy(
                ropeParametersFull = weights.metadata.ropeParametersFull.copy(partialRotaryFactor = partial),
            ),
        )
        println("PARTIAL_ROTARY $partial")
        val model = GemmaNetworkLoader.fromWeights(ctx, patched, FP32::class)

        // Disable per-layer KV caches before tracing. KVCache.update() does raw
        // copyToFloatArray (non-traceable); under VoidTensorOps it returns a
        // zero [1,seq,headDim] leaf for K and V -> 36 zero "frozen params" that
        // kill RoPE/attention in the export. A single prefill pass needs no
        // cache: K/V are computed fresh for all positions and stay traceable.
        fun stripKvCache(m: Module<*, *>) {
            if (m is MultiHeadAttention<*, *>) m.kvCache = null
            m.modules.forEach { stripKvCache(it) }
        }
        stripKvCache(model)

        val seqLen = (System.getProperty("seqLen") ?: "4").toInt()
        val input = VoidOpsTensor(
            object : TensorData<FP32, Float> {
                override val shape = Shape(1, seqLen)
                override fun get(vararg indices: Int): Float = 0.0f
                override fun set(vararg indices: Int, value: Float) {}
            },
            FP32::class,
        )
        val tapeCtx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val tape = tapeCtx.record {
            val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
            Execution.tapeStack.pushTape(ct)
            try { model.forward(input, this as ExecutionContext) } finally { Execution.tapeStack.popTape() }
        }.first
        // embedConstants=true: finalize resolves the bound weights (now stored
        // as primitive FloatArray, no .toList() boxing). ExternalAlways lifts
        // every weight into the .irpa.
        val graph = (tape as DefaultExecutionTape).toComputeGraph(
            synthesizeExternalInputs = true, embedConstants = true,
        )
        val module = StableHloConverterFactory
            .createBasic(ConstantMaterializationPolicy.ExternalAlways(scope = "model"))
            .convert(graph, "gemma")

        val outDir = File("/home/miso/projects/coral/build-mlir").apply { mkdirs() }
        File(outDir, "gemma-baked.mlir").writeText(module.content)

        val ext = module.externalParameters
        val funcArgs = module.content.lineSequence().firstOrNull { it.contains("func.func @gemma(") }
            ?.let { Regex("%arg\\d+").findAll(it).count() } ?: -1
        val globals = module.content.lineSequence().count { it.trimStart().startsWith("util.global ") }
        val totalBytes = ext.sumOf { it.source.sizeInBytes }
        println("EXTPARAMS ${ext.size} totalMiB=${totalBytes / (1024 * 1024)}")
        println("FUNCARGS $funcArgs UTILGLOBALS $globals MLIRlines=${module.content.lines().size}")

        // Write a safetensors keyed t0..tN (1-D, size-equivalent; IREE accepts
        // size-equivalent params and the util.global carries the real shape).
        // iree-convert-parameters turns this into a valid .irpa — bypassing
        // SKaiNET's IrpaWriter, whose header is not yet IREE-v0 compatible.
        var off = 0L
        val hdr = StringBuilder("{")
        ext.forEachIndexed { i, e ->
            val len = e.source.sizeInBytes
            val count = len / 4
            if (i > 0) hdr.append(",")
            hdr.append("\"${e.key}\":{\"dtype\":\"F32\",\"shape\":[$count],\"data_offsets\":[$off,${off + len}]}")
            off += len
        }
        hdr.append("}")
        val headerBytes = hdr.toString().encodeToByteArray()
        val st = File(outDir, "gemma.safetensors")
        java.io.BufferedOutputStream(java.io.FileOutputStream(st), 1 shl 20).use { os ->
            val lenBuf = java.nio.ByteBuffer.allocate(8).order(java.nio.ByteOrder.LITTLE_ENDIAN)
                .putLong(headerBytes.size.toLong())
            os.write(lenBuf.array())
            os.write(headerBytes)
            for (e in ext) {
                val src = e.source as BufferHandle.Owned
                os.write(src.data, src.offset, src.sizeInBytes.toInt())
            }
        }
        println("WROTE_SAFETENSORS ${st.absolutePath} sizeMiB=${st.length() / (1024 * 1024)}")
    }
}
