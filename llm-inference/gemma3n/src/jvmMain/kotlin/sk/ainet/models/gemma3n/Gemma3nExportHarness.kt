package sk.ainet.models.gemma3n

import kotlinx.coroutines.runBlocking
import sk.ainet.compile.hlo.ConstantMaterializationPolicy
import sk.ainet.compile.hlo.ExternalParameterRef
import sk.ainet.compile.hlo.StableHloConverterFactory
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.plan.EncodingRequest
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.plan.WeightShapeOrientation
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.transformer.MultiHeadAttention
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.Bf16TensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.tensor.storage.BufferHandle
import sk.ainet.lang.types.FP32
import sk.ainet.tape.Execution
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Gemma 3n compiled-export harness — the StableHLO → IREE mobile path for the #377 DSL
 * lane (SmolLM2/FunctionGemma redecode pattern): author `gemma3nNetwork()` from the real
 * GGUF, trace ONE fixed `[1, seq]` prefill pass ending in the DSL argMax tail, and emit
 * portable StableHLO with every weight lifted to an EXTERNAL parameter
 * (`scope = "model"`). `func @gemma3n` returns `tensor<seqxi32>` — the
 * `llm-runtime/iree-android` `IreeRedecodeSession` contract, so a compiled
 * `(vmfb, irpa, "gemma3n")` triple runs on-device unchanged.
 *
 * Everything gemma3n-specific traces through `ctx.ops`: the four AltUp streams with the
 * tanh router, Laurel, Gaussian-top-k sparsity, and the PLE token gather
 * (`PerLayerEmbedding` switches to an `indexSelect` graph op while recording — the eager
 * packed row-dequant path is host-side and would bake constants). KV caches are stripped
 * before tracing (a single fixed-seq pass needs none; `OwnerReadOnlyKVCache` followers
 * would otherwise record the stateful eager copy path).
 *
 * **The PLE table is NOT a graph parameter.** `per_layer_inputs` (`[1, seq, L, pleDim]`)
 * is the graph's SECOND INPUT, computed on the CPU from the packed PLE table at runtime —
 * PLE's design point per Google's Gemma 3n guide: per-layer embeddings live
 * off-accelerator. That keeps the 262k×7680 table (2 GB packed, 8 GB dense) out of the
 * parameter archive entirely: the archive carries the trunk + token embedding only
 * (~4 GB bf16 for E2B; int8 narrowing is follow-up scope, FunctionGemma
 * `rewriteGlobalsToInt8` precedent). The host load must be dense FP32 (packed tensors
 * cannot become graph constants), and trace zeros + graph copies sit alongside it —
 * E2B peaks ~44 GB transient heap; run on a large-memory host with
 * `-PexportMaxHeap=46g` (the all-zero trace pages compress well under macOS).
 *
 * Writes `<outDir>/gemma3n-gen.mlir` + `<outDir>/gemma3n.safetensors` + `manifest.json`.
 * vmfb compilation happens outside Kotlin (iree-compile; the repo pins a Torq fork for
 * `iree-run-module` — see `llm-runtime/gemma-iree`).
 */
@OptIn(ExperimentalMemoryApi::class)
public object Gemma3nExportHarness {

    public const val FN_REDECODE: String = "gemma3n"
    public const val PARAMETER_SCOPE: String = "model"

    public data class RedecodeResult(
        val mlirPath: String,
        val safetensorsPath: String,
        val manifestPath: String,
        val externalParamCount: Int,
        val weightMiB: Long,
        val seq: Int,
        val vocabSize: Int,
    )

    public fun export(
        gguf: String,
        outDir: String,
        seq: Int = 24,
        bf16: Boolean = true,
        /**
         * Truncate the exported trunk to the first N layers. Full-E2B export needs a
         * ≥64 GB host (the engine's trace/export pipeline keeps dense weights, per-op
         * zero buffers and graph constant copies co-resident — engine issue filed);
         * a truncated export verifies the whole pipeline end-to-end on smaller hosts
         * and is what the model-gated smoke uses.
         */
        layers: Int? = null,
    ): RedecodeResult = runBlocking {
        val ctx = DirectCpuExecutionContext.create()
        // DENSE load is REQUIRED for export: packed tensors cannot be extracted as graph
        // constants — the tracer silently turns them into opaque function ARGUMENTS
        // (measured: 190+ weight args, zero dot ops — an unservable module). The sanity
        // check below makes that failure loud if it ever regresses.
        val weights = Gemma3nWeightLoader(
            randomAccessProvider = { JvmRandomAccessSource.open(gguf) },
            weightForm = WeightForm(
                encoding = EncodingRequest.DequantizeTo(FP32),
                shape = WeightShapeOrientation.OUT_IN,
            ),
        ).loadToMapStreaming<FP32, Float>(ctx)

        val fullLayers = weights.metadata.blockCount
        val n = layers?.coerceIn(1, fullLayers) ?: fullLayers
        val weightsN = if (n == fullLayers) weights else {
            val md = weights.metadata
            val firstShared = fullLayers - md.kvSharedLayers
            Gemma3nWeights(
                md.copy(
                    blockCount = n,
                    // Shared-KV followers only exist past firstShared; a truncated trunk
                    // below that point has no sharing.
                    kvSharedLayers = (n - firstShared).coerceAtLeast(0),
                    feedForwardLengths = md.feedForwardLengths.take(n),
                    layerPattern = md.layerPattern.take(n).ifEmpty { md.layerPattern },
                    activationSparsityScales = md.activationSparsityScales.take(n),
                ),
                weights.tensors.filterKeys { key ->
                    !key.startsWith("blk.") ||
                        (key.removePrefix("blk.").substringBefore('.').toIntOrNull() ?: 0) < n
                },
            )
        }
        val model = Gemma3nNetworkLoader.fromWeights(
            ctx, weightsN, maxInferenceLen = seq, pleNumLayers = fullLayers,
        )

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
        // Second graph input: per_layer_inputs, computed host-side (CPU gather over the
        // packed PLE table) by the on-device session, exactly like the eager path does.
        val md0 = weights.metadata
        val pliInput = VoidOpsTensor(
            object : TensorData<FP32, Float> {
                override val shape = Shape(1, seq, fullLayers, md0.perLayerEmbeddingLength)
                override fun get(vararg indices: Int): Float = 0.0f
                override fun set(vararg indices: Int, value: Float) {}
            },
            FP32::class,
        )
        (model as Gemma3nModel<FP32, Float>).externalPerLayerInputs = pliInput

        val tapeCtx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val tape = tapeCtx.record {
            val ct = currentTape ?: error("no tape")
            Execution.tapeStack.pushTape(ct)
            try {
                val ectx = this as ExecutionContext
                val logits = model.forward(input, ectx)      // [1, seq, vocab] f32
                val idx = ectx.ops.argMax(logits, dim = -1)  // [1, seq] i32
                ectx.ops.squeeze(idx, 0)                     // [seq] i32 — the redecode runtime contract
            } finally {
                Execution.tapeStack.popTape()
            }
        }.first

        val graph = (tape as DefaultExecutionTape).toComputeGraph(
            synthesizeExternalInputs = true, embedConstants = true,
        )
        val module = StableHloConverterFactory
            .createBasic(ConstantMaterializationPolicy.ExternalAlways(scope = PARAMETER_SCOPE))
            .convert(graph, FN_REDECODE)

        val out = File(outDir).apply { mkdirs() }
        val ext = module.externalParameters

        // Sanity: the redecode contract has exactly TWO function inputs (tokens,
        // per_layer_inputs). More means weights leaked into the signature (the
        // packed-tensor failure mode) — the module would be unservable.
        val sig = Regex("func\\.func @$FN_REDECODE\\(([^)]*)\\)").find(module.content)?.groupValues?.get(1)
            ?: error("gemma3n export: emitted module has no @$FN_REDECODE function")
        val argCount = Regex("%arg\\d+").findAll(sig).count()
        require(argCount == 2) {
            "gemma3n export: expected 2 graph inputs (tokens, per_layer_inputs) but the " +
                "function signature has $argCount — weights leaked into the signature " +
                "(packed tensors cannot become graph constants; the loader must dequantize)."
        }
        // Emission completeness: the engine converter currently reports operand-linkage
        // failures as MLIR comments and still exits 0, leaving a compute-free module
        // (SKaiNET#1247). Make that a hard error here.
        val failures = Regex("// Conversion failed for node ([^:]+):").findAll(module.content)
            .map { it.groupValues[1] }.take(5).toList()
        require(failures.isEmpty()) {
            "gemma3n export: the StableHLO converter failed on ${failures.size}+ nodes " +
                "(first: $failures) — the emitted module is not servable. See SKaiNET#1247."
        }
        require(!module.content.contains("// Warning: No output values found")) {
            "gemma3n export: emitted function has no outputs — unservable module (SKaiNET#1247)."
        }

        val mlir = if (bf16) rewriteGlobalsToBf16(module.content) else module.content
        val mlirFile = File(out, "gemma3n-gen.mlir").apply { writeText(mlir) }

        val stFile = File(out, "gemma3n.safetensors")
        writeSafetensors(ext, stFile, bf16)

        val manifestFile = File(out, "manifest.json").apply {
            writeText(
                """
                {
                  "family": "gemma3n",
                  "function": "$FN_REDECODE",
                  "parameterScope": "$PARAMETER_SCOPE",
                  "seq": $seq,
                  "layers": $n,
                  "layersTotal": $fullLayers,
                  "vocabSize": ${weights.metadata.vocabSize},
                  "dtype": "${if (bf16) "bf16" else "f32"}",
                  "inputs": ["tokens[1,$seq]i32", "per_layer_inputs[1,$seq,${weights.metadata.blockCount},${weights.metadata.perLayerEmbeddingLength}]f32"],
                  "perLayerInputsOnHost": true,
                  "mlir": "${mlirFile.name}",
                  "safetensors": "${stFile.name}"
                }
                """.trimIndent() + "\n",
            )
        }

        val totalF32 = ext.sumOf { it.source.sizeInBytes }
        RedecodeResult(
            mlirPath = mlirFile.absolutePath,
            safetensorsPath = stFile.absolutePath,
            manifestPath = manifestFile.absolutePath,
            externalParamCount = ext.size,
            weightMiB = (if (bf16) totalF32 / 2 else totalF32) / (1024 * 1024),
            seq = seq,
            vocabSize = weights.metadata.vocabSize,
        )
    }

    /**
     * FunctionGemma/SmolLM2 safetensors contract, with one difference: bf16 conversion is
     * CHUNKED — gemma3n's PLE table is 2.01B elements, and the single-`ByteArray`
     * conversion buffer the smaller models use would exceed the JVM array limit.
     */
    private fun writeSafetensors(ext: List<ExternalParameterRef>, stFile: File, bf16: Boolean) {
        val dtype = if (bf16) "BF16" else "F32"
        val bpe = if (bf16) 2L else 4L
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
            val chunkElems = 1 shl 24 // 16M floats per conversion chunk (32 MiB bf16 out)
            for (e in ext) {
                val src = e.source as BufferHandle.Owned
                if (bf16) {
                    val data = src.data
                    val n = (src.sizeInBytes / 4).toInt()
                    var done = 0
                    while (done < n) {
                        val take = minOf(chunkElems, n - done)
                        val obuf = ByteArray(take * 2)
                        for (j in 0 until take) {
                            val o = src.offset + (done + j) * 4
                            val fb = (data[o].toInt() and 0xFF) or
                                ((data[o + 1].toInt() and 0xFF) shl 8) or
                                ((data[o + 2].toInt() and 0xFF) shl 16) or
                                ((data[o + 3].toInt() and 0xFF) shl 24)
                            val bf = Bf16TensorData.floatToBf16Bits(Float.fromBits(fb))
                            obuf[j * 2] = (bf and 0xFF).toByte()
                            obuf[j * 2 + 1] = ((bf ushr 8) and 0xFF).toByte()
                        }
                        os.write(obuf)
                        done += take
                    }
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
