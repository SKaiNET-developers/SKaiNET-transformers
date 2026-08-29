package sk.ainet.models.smollm2

import kotlinx.coroutines.runBlocking
import sk.ainet.compile.hlo.ConstantMaterializationPolicy
import sk.ainet.compile.hlo.ExternalParameterRef
import sk.ainet.compile.hlo.StableHloConverterFactory
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.plan.EncodingRequest
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.plan.WeightShapeOrientation
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.transformer.MultiHeadAttention
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.Bf16TensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.tensor.storage.BufferHandle
import sk.ainet.lang.types.FP32
import sk.ainet.models.llama.DecoderGgufWeightLoader
import sk.ainet.models.llama.LlamaNetworkLoader
import sk.ainet.tape.Execution
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * SmolLM2 compiled-export harness — the module's ONE export surface
 * (whisper/functiongemma pattern). Mirrors `:llm-inference:functiongemma`'s
 * `FunctionGemmaExportHarness.export` (the redecode graph only — see the
 * module doc for why the two-graph KV-cache decode is follow-up scope):
 * author `llamaNetwork()` from the real GGUF checkpoint, trace ONE fixed
 * `[1,seq]` prefill pass that ends in the DSL argMax tail, and emit portable
 * StableHLO with EXTERNAL params. The emitted `func @smollm2` returns
 * `tensor<seqxi32>` directly — small per-step output, no host-side argmax
 * over a `[seq, vocab]` logits tensor.
 *
 * Unlike gemma3, SmolLM2 is a standard llama-architecture model: no
 * partial-rotary override, no sliding/global RoPE-base split.
 *
 * Writes `<outDir>/smollm2-gen.mlir` + `<outDir>/smollm2.safetensors`.
 */
public object SmolLm2ExportHarness {

    public data class RedecodeResult(
        val mlirPath: String,
        val safetensorsPath: String,
        val externalParamCount: Int,
        val weightMiB: Long,
        val seq: Int,
    )

    /**
     * Trace `llamaNetwork()`, drop per-layer KV caches (a single fixed-seq
     * prefill pass needs none — `KVCache.update()`'s raw `copyToFloatArray`
     * is non-traceable under `VoidTensorOps`), append the DSL argMax tail,
     * and emit StableHLO with every weight lifted to an external param.
     *
     * @param bf16 halve the archive by emitting weights as bf16 externals
     *   (a `stablehlo.convert bf16->f32` on load; compute stays f32) — a
     *   bit-exact truncation of the same weights, not a lossy requantization.
     */
    public fun export(
        gguf: String,
        outDir: String,
        seq: Int = 24,
        bf16: Boolean = true,
    ): RedecodeResult = runBlocking {
        val ctx = DirectCpuExecutionContext.create()
        val weights = DecoderGgufWeightLoader(
            randomAccessProvider = { JvmRandomAccessSource.open(gguf) },
            acceptedArchitectures = setOf("llama", "mistral"),
            weightForm = WeightForm(
                encoding = EncodingRequest.DequantizeTo(FP32),
                shape = WeightShapeOrientation.OUT_IN
            ),
        ).loadToMapStreaming<FP32, Float>(ctx, FP32::class)
        val model = LlamaNetworkLoader.fromWeights(weights)

        fun stripKvCache(m: Module<*, *>) {
            if (m is MultiHeadAttention<*, *>) m.kvCache = null
            m.modules.forEach { stripKvCache(it) }
        }
        stripKvCache(model)

        val input = VoidOpsTensor(
            object : TensorData<FP32, Float> {
                override val shape = Shape(1, seq)
                override fun get(vararg indices: Int): Float = 0.0f
                override fun set(vararg indices: Int, value: Float) {}
            },
            FP32::class,
        )
        val tapeCtx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val tape = tapeCtx.record {
            val ct = currentTape ?: error("no tape")
            Execution.tapeStack.pushTape(ct)
            try {
                val ectx = this as ExecutionContext
                val logits = model.forward(input, ectx)      // [1, seq, vocab] f32
                val idx = ectx.ops.argMax(logits, dim = -1)  // [1, seq] i32
                ectx.ops.squeeze(idx, 0)                     // [seq] i32 — the smollm2-gen runtime contract
            } finally {
                Execution.tapeStack.popTape()
            }
        }.first

        val graph = (tape as DefaultExecutionTape).toComputeGraph(
            synthesizeExternalInputs = true, embedConstants = true,
        )
        val module = StableHloConverterFactory
            .createBasic(ConstantMaterializationPolicy.ExternalAlways(scope = "model"))
            .convert(graph, "smollm2")

        val out = File(outDir).apply { mkdirs() }
        val ext = module.externalParameters

        val mlir = if (bf16) rewriteGlobalsToBf16(module.content) else module.content
        val mlirFile = File(out, "smollm2-gen.mlir").apply { writeText(mlir) }

        val stFile = File(out, "smollm2.safetensors")
        writeSafetensors(ext, stFile, bf16)

        val totalF32 = ext.sumOf { it.source.sizeInBytes }
        RedecodeResult(
            mlirPath = mlirFile.absolutePath,
            safetensorsPath = stFile.absolutePath,
            externalParamCount = ext.size,
            weightMiB = (if (bf16) totalF32 / 2 else totalF32) / (1024 * 1024),
            seq = seq,
        )
    }

    /** Copied from `FunctionGemmaExportHarness.writeSafetensors` (identical contract). */
    private fun writeSafetensors(ext: List<ExternalParameterRef>, stFile: File, bf16: Boolean) {
        val dtype = if (bf16) "BF16" else "F32"
        val bpe = if (bf16) 2 else 4
        var off = 0L
        val hdr = StringBuilder("{")
        ext.forEachIndexed { i, e ->
            val count = e.source.sizeInBytes / 4
            val len = count * bpe
            if (i > 0) hdr.append(",")
            hdr.append("\"${e.key}\":{\"dtype\":\"$dtype\",\"shape\":[$count],\"data_offsets\":[$off,${off + len}]}")
            off += len
        }
        hdr.append("}")
        val headerBytes = hdr.toString().encodeToByteArray()
        BufferedOutputStream(FileOutputStream(stFile), 1 shl 20).use { os ->
            os.write(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(headerBytes.size.toLong()).array())
            os.write(headerBytes)
            for (e in ext) {
                val src = e.source as BufferHandle.Owned
                if (bf16) {
                    val data = src.data
                    val base = src.offset
                    val n = src.sizeInBytes.toInt() / 4
                    val obuf = ByteArray(n * 2)
                    for (j in 0 until n) {
                        val o = base + j * 4
                        val fb = (data[o].toInt() and 0xFF) or
                            ((data[o + 1].toInt() and 0xFF) shl 8) or
                            ((data[o + 2].toInt() and 0xFF) shl 16) or
                            ((data[o + 3].toInt() and 0xFF) shl 24)
                        val bf = Bf16TensorData.floatToBf16Bits(Float.fromBits(fb))
                        obuf[j * 2] = (bf and 0xFF).toByte()
                        obuf[j * 2 + 1] = ((bf ushr 8) and 0xFF).toByte()
                    }
                    os.write(obuf)
                } else {
                    os.write(src.data, src.offset, src.sizeInBytes.toInt())
                }
            }
        }
    }

    /** Copied from `FunctionGemmaExportHarness.rewriteGlobalsToBf16` (identical contract). */
    private fun rewriteGlobalsToBf16(mlir: String): String {
        var m = mlir
        m = Regex("""(util\.global private @\w+ = #flow\.parameter\.named<"[^"]*"::"[^"]*"> : tensor<[0-9x]*x)f32>""")
            .replace(m) { it.groupValues[1] + "bf16>" }
        m = Regex("""(%\w+) = util\.global\.load @(\w+) : tensor<([0-9x]*)xf32>""")
            .replace(m) { r ->
                val ssa = r.groupValues[1]
                val g = r.groupValues[2]
                val shape = r.groupValues[3]
                "${ssa}_h = util.global.load @$g : tensor<${shape}xbf16>\n" +
                    "    $ssa = stablehlo.convert ${ssa}_h : (tensor<${shape}xbf16>) -> tensor<${shape}xf32>"
            }
        return m
    }
}
