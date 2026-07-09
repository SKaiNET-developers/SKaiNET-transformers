package sk.ainet.apps.kgemma

import kotlinx.coroutines.runBlocking
import sk.ainet.compile.hlo.ConstantMaterializationPolicy
import sk.ainet.compile.hlo.StableHloConverterFactory
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.model.QuantPolicy
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
import sk.ainet.models.gemma.Gemma4WeightLoader
import sk.ainet.models.gemma.GemmaNetworkLoader
import sk.ainet.tape.Execution
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * FunctionGemma compiled-export: author `gemmaNetwork()` from the real GGUF checkpoint,
 * trace ONE fixed-prefill pass that ends in the DSL argMax tail, and emit portable
 * StableHLO with EXTERNAL params (the 270M weights live in an `.irpa`, not baked).
 *
 * Promotes the former `RealGemmaBakeIrpaTest` into a runnable library step and folds the
 * ex-Python rewrites INTO the DSL/export (skainet philosophy — no Python):
 *  - the per-position argmax tail is now `ops.argMax(logits, -1)` (a real DSL op), so the
 *    emitted `func @gemma` already returns `tensor<seqxi32>` — retires `add_argmax_perpos.py`.
 *  - weights are emitted as **bf16** externals (globals + a bf16 safetensors, f32 compute via a
 *    `stablehlo.convert bf16->f32` on load), halving the archive — retires `make_f16.py`. bf16 is
 *    a bit-exact drop-in for the f16 vmfb (verified board A/B).
 *
 * Writes `<outDir>/gemma-gen.mlir` + `<outDir>/gemma.safetensors`. `scripts/compile-gemma.sh`
 * turns these into `gemma-gen.vmfb` (iree-compile llvm-cpu) + `gemma-gen.irpa`
 * (iree-convert-parameters). Entry function stays `gemma` (no `@main` rename).
 */
public object FunctionGemmaExport {

    public data class Result(
        val mlirPath: String,
        val safetensorsPath: String,
        val externalParamCount: Int,
        val weightMiB: Long,
        val seq: Int,
    )

    public fun export(
        gguf: String,
        outDir: String,
        seq: Int = 24,
        partialRotary: Float = 1.0f,
        bf16: Boolean = true,
    ): Result = runBlocking {
        val ctx = DirectCpuExecutionContext.create()
        val weights = Gemma4WeightLoader(
            randomAccessProvider = { JvmRandomAccessSource.open(gguf) },
            quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
        ).loadToMapStreaming<FP32, Float>(ctx, FP32::class)
        // gemma3 uses FULL rotary; the gguf omits rope.partial_rotary_factor (loader defaults to a
        // Gemma-4 0.25 that mis-rotates global layers). Force it.
        val patched = weights.copy(
            metadata = weights.metadata.copy(
                ropeParametersFull = weights.metadata.ropeParametersFull.copy(partialRotaryFactor = partialRotary),
            ),
        )
        val model = GemmaNetworkLoader.fromWeights(ctx, patched, FP32::class)

        // Drop per-layer KV caches before tracing (KVCache.update() is non-traceable under
        // VoidTensorOps -> 36 zero frozen params that kill RoPE/attention). One prefill pass
        // needs no cache: K/V are computed fresh for every position.
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
            val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
            Execution.tapeStack.pushTape(ct)
            try {
                val ectx = this as ExecutionContext
                val logits = model.forward(input, ectx)      // [1, seq, vocab] f32
                val idx = ectx.ops.argMax(logits, dim = -1)  // [1, seq] i32  (the real DSL argMax op)
                ectx.ops.squeeze(idx, 0)                     // [seq] i32 — the gemma-gen runtime contract
            } finally {
                Execution.tapeStack.popTape()
            }
        }.first

        val graph = (tape as DefaultExecutionTape).toComputeGraph(
            synthesizeExternalInputs = true, embedConstants = true,
        )
        val module = StableHloConverterFactory
            .createBasic(ConstantMaterializationPolicy.ExternalAlways(scope = "model"))
            .convert(graph, "gemma")

        val out = File(outDir).apply { mkdirs() }
        val ext = module.externalParameters
        val mlir = if (bf16) rewriteGlobalsToBf16(module.content) else module.content
        val mlirFile = File(out, "gemma-gen.mlir").apply { writeText(mlir) }

        val stFile = File(out, "gemma.safetensors")
        val dtype = if (bf16) "BF16" else "F32"
        val bpe = if (bf16) 2 else 4
        var off = 0L
        val hdr = StringBuilder("{")
        ext.forEachIndexed { i, e ->
            val count = e.source.sizeInBytes / 4          // f32 element count
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
                        val bf = Bf16TensorData.floatToBf16Bits(Float.fromBits(fb)) // truncation = core parity
                        obuf[j * 2] = (bf and 0xFF).toByte()
                        obuf[j * 2 + 1] = ((bf ushr 8) and 0xFF).toByte()
                    }
                    os.write(obuf)
                } else {
                    os.write(src.data, src.offset, src.sizeInBytes.toInt())
                }
            }
        }

        val totalF32 = ext.sumOf { it.source.sizeInBytes }
        Result(
            mlirPath = mlirFile.absolutePath,
            safetensorsPath = stFile.absolutePath,
            externalParamCount = ext.size,
            weightMiB = (if (bf16) totalF32 / 2 else totalF32) / (1024 * 1024),
            seq = seq,
        )
    }

    /** f32 weight `util.global`s -> bf16 + a `stablehlo.convert bf16->f32` on each load (compute stays f32). */
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
