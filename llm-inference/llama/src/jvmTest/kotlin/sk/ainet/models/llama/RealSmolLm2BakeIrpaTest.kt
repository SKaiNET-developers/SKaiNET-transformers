package sk.ainet.models.llama

import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Tag
import sk.ainet.compile.hlo.ConstantMaterializationPolicy
import sk.ainet.compile.hlo.StableHloConverterFactory
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.transformer.MultiHeadAttention
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.tensor.storage.BufferHandle
import sk.ainet.lang.types.FP32
import sk.ainet.tape.Execution
import java.io.File
import kotlin.test.Test

/**
 * Bake the REAL SmolLM2-135M-Instruct to a self-contained IREE artifact — the
 * host-side half of transformers#305 (the compiled leg of #272's SmolLM2
 * cross-target reproducer): `fromWeights` (real topology + real weights) ->
 * trace (embedConstants=true, resolves the bound weights) -> StableHLO with
 * ExternalAlways so every weight becomes a `util.global.load` +
 * ExternalParameterRef -> a safetensors readable by `iree-convert-parameters`.
 * The emitted vmfb then takes ONLY the token input; weights resolve from the
 * archive via `iree-run-module --parameters=model=smollm2.irpa`.
 *
 * Mirrors `:llm-inference:gemma`'s `RealGemmaBakeIrpaTest` — same shared
 * `decoderTransformerNetwork` builder (llama and gemma3 both go through it),
 * so the same trace recipe applies: strip per-layer `KVCache` before tracing
 * (a single prefill pass needs none — K/V are computed fresh for every
 * position and stay traceable, whereas `KVCache.update()`'s raw
 * `copyToFloatArray` is non-traceable under `VoidTensorOps` and would freeze
 * 2N "params" of zeros). No RoPE partial-rotary quirk here — SmolLM2 is a
 * standard llama-architecture model, so the loader's defaults apply as-is.
 *
 * Gated like the reproducer spike (transformers#272): skips cleanly unless
 * `SMOLLM2_MODEL` points at an existing `.gguf`.
 *
 * ```
 * export SMOLLM2_MODEL=/abs/path/SmolLM2-135M-Instruct-Q8_0.gguf
 * ./gradlew :llm-inference:llama:jvmTest --tests '*RealSmolLm2BakeIrpaTest*' -PincludeIntegration
 * ```
 */
@Tag("integration")
class RealSmolLm2BakeIrpaTest {
    @Test
    fun bakeRealSmolLm2ToIrpa() = runBlocking {
        val path = System.getenv("SMOLLM2_MODEL")?.trim().orEmpty()
        assumeTrue(path.isNotEmpty() && File(path).isFile) {
            "[skip] SMOLLM2_MODEL not set (or not an existing file) — skipping SmolLM2 bake-to-irpa."
        }

        val ctx = DirectCpuExecutionContext.create()
        val weights = DecoderGgufWeightLoader(
            randomAccessProvider = { JvmRandomAccessSource.open(path) },
            weightForm = DECODER_DEQUANTIZE_ALL,
            acceptedArchitectures = setOf("llama", "mistral"),
        ).loadToMapStreaming<FP32, Float>(ctx, FP32::class)
        val model = LlamaNetworkLoader.fromWeights(weights)

        // Disable per-layer KV caches before tracing — see class doc.
        fun stripKvCache(m: Module<*, *>) {
            if (m is MultiHeadAttention<*, *>) m.kvCache = null
            m.modules.forEach { stripKvCache(it) }
        }
        stripKvCache(model)

        val seqLen = (System.getenv("SMOLLM2_SEQLEN") ?: "4").toInt()
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
            val ct = this.currentTape ?: error("no tape")
            Execution.tapeStack.pushTape(ct)
            try { model.forward(input, this as ExecutionContext) } finally { Execution.tapeStack.popTape() }
        }.first
        // embedConstants=true: finalize resolves the bound weights (now stored
        // as primitive FloatArray, no .toList() boxing). ExternalAlways lifts
        // every weight into the safetensors below.
        val graph = (tape as DefaultExecutionTape).toComputeGraph(
            synthesizeExternalInputs = true, embedConstants = true,
        )
        val module = StableHloConverterFactory
            .createBasic(ConstantMaterializationPolicy.ExternalAlways(scope = "model"))
            .convert(graph, "smollm2")

        val outDir = File(System.getenv("SMOLLM2_MLIR_OUT") ?: "build/mlir-export").apply { mkdirs() }
        File(outDir, "smollm2-baked.mlir").writeText(module.content)

        val ext = module.externalParameters
        val funcArgs = module.content.lineSequence().firstOrNull { it.contains("func.func @smollm2(") }
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
        val st = File(outDir, "smollm2.safetensors")
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
