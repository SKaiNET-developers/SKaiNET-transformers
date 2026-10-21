package sk.ainet.models.whisper

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import sk.ainet.compile.hlo.ConstantMaterializationPolicy
import sk.ainet.compile.hlo.ExternalParameterRef
import sk.ainet.compile.hlo.StableHloConverterFactory
import sk.ainet.compile.hlo.StableHloModule
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.tensor.storage.BufferHandle
import sk.ainet.lang.tensor.storage.DefaultBufferResolver
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import sk.ainet.tape.Execution
import java.io.File

/**
 * Export options that are not part of the checkpoint.
 *
 * @property audioCtx encoder positions after the stride-2 conv (`melFrames = 2 * audioCtx`); 200 = a 4 s window,
 *   1500 = the stock 30 s. Shorter than ≈ 150 (3 s) is out of distribution and breaks decoding.
 * @property maxPositions length of the decoder's self-attention KV cache (prompt + generated tokens).
 * @property promptLength length of the prefill prompt the prefill graph is built for (`<sot> <lang> <transcribe>
 *   <notimestamps>` = 4 for a multilingual checkpoint).
 * @property language the language code written into the manifest's prompt; ignored for English-only checkpoints.
 * @property scope the IREE parameter scope of the external weights.
 */
public data class WhisperExportOptions(
    val audioCtx: Int = 200,
    val maxPositions: Int = 48,
    val promptLength: Int = 4,
    val language: String = "en",
    val scope: String = "model",
)

/**
 * The whisper "contract v2" StableHLO export: three graphs plus one merged parameter archive and a manifest.
 *
 * The three graphs are traced through ONE [DefaultGraphExecutionContext], so all three share ONE TraceSession:
 * tensor ref ids (`t<N>`) are assigned per tensor OBJECT, so the same baked weight tensor gets the same
 * external-parameter key in the prefill and step modules — the merged `params.irpa` then dedups by key with
 * byte-identity asserted (a key collision with differing bytes would mean the sessions diverged and is a hard error).
 *
 *  - `whisper_encoder`  mel `[1,nMels,melFrames]` → features `[1,audioCtx,dim]`; weights INLINE dense constants —
 *    a self-contained vmfb.
 *  - `whisper_prefill`  promptIds `[promptLength]` i32 + feat → logits `[1,promptLength,vocab]` + per layer
 *    selfK/selfV `[1,maxPositions,dim]` (zero-padded) and crossK/crossV `[1,audioCtx,dim]`. causalMask/zeroPad are
 *    embedded constants. Weights external (`ExternalAlways(scope)`).
 *  - `whisper_step`     tok `[1]` i32, pos `[1]` i32, addMask `[1,1,1,maxPositions]`, wf `[1,maxPositions,1]` + per
 *    layer selfK/selfV/crossK/crossV → logits `[1,1,vocab]` + per layer updated selfK/selfV. Weights external.
 *
 * ARGUMENT AND RESULT ORDER is pinned by "touch" identity reshapes: the graph builder synthesizes input placeholders
 * in first-unresolved-use order, so touching every input tensor at the top of the trace (and re-tapping every output
 * at the bottom) makes the func signature exactly the order touched:
 *
 *   prefill args    = promptIds, feat
 *   prefill results = logits, then per layer l: selfK_l, selfV_l, crossK_l, crossV_l
 *   step args       = tok, pos, addMask, wf, then per layer l: selfK_l, selfV_l, crossK_l, crossV_l
 *   step results    = logits, then per layer l: selfK_l, selfV_l
 *
 * The identity reshapes fold away in iree-compile.
 */
public class WhisperExportHarness(
    public val cfg: WhisperConfig,
    public val options: WhisperExportOptions = WhisperExportOptions(audioCtx = cfg.audioCtx),
    private val log: (String) -> Unit = {},
) {
    init {
        require(options.audioCtx == cfg.audioCtx) { "options.audioCtx ${options.audioCtx} != cfg.audioCtx ${cfg.audioCtx} — build the config with the export's audio context" }
        require(options.promptLength in 1 until options.maxPositions) { "promptLength ${options.promptLength} must be in 1 until maxPositions ${options.maxPositions}" }
    }

    public data class ExportResult(
        val outDir: File,
        val encoderMlir: File,
        val prefillMlir: File,
        val stepMlir: File,
        val paramsIrpa: File,
        val manifest: File,
        val prefillParamCount: Int,
        val stepParamCount: Int,
        val mergedParamCount: Int,
    )

    /** Bake the weights of [modelSafetensors], trace + convert the three graphs, write MLIR + params + manifest. */
    public fun export(modelSafetensors: File, outDir: File): ExportResult {
        require(modelSafetensors.isFile) { "no checkpoint at $modelSafetensors" }
        outDir.mkdirs()
        val maxP = options.maxPositions
        val scope = options.scope

        val cpuCtx = DirectCpuExecutionContext.create()
        val encoder = WhisperEncoderModel<FP32, Float>(cfg, FP32::class)
        val decoder = WhisperDecoderModel<FP32, Float>(cfg, FP32::class)
        SafeTensorsWeightSource({ JvmRandomAccessSource.open(modelSafetensors.path) }).use { src ->
            val nEnc = bakeWhisperWeights(encoder, src, cfg, FP32::class, cpuCtx as ExecutionContext)
            val nDec = bakeWhisperWeights(decoder, src, cfg, FP32::class, cpuCtx)
            log("baked encoder=$nEnc decoder=$nDec parameters")
        }

        // ONE tape context → ONE shared TraceSession → stable weight keys across graphs.
        val gctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())

        // ---------- whisper_encoder ----------
        val mel = voidF(1, cfg.nMels, cfg.melFrames)
        val encTape = trace(gctx) {
            encoder.forward(touch(mel), this)
        }
        val encGraph = encTape.toComputeGraph(
            synthesizeExternalInputs = true,
            inputTensorIds = setOf(idOf(gctx, mel)),
            embedConstants = true,
        )
        val encModule = StableHloConverterFactory.createExtended()
            .convert(encGraph, "whisper_encoder")

        // ---------- whisper_prefill ----------
        val seq = options.promptLength
        val promptIds = voidI(seq)
        val feat = voidF(1, cfg.audioCtx, cfg.dim)
        val causal = WhisperMasks.causal<FP32, Float>(seq, cpuCtx, FP32::class)
        val zeroPad = WhisperMasks.zeroPad<FP32, Float>(seq, maxP, cfg.dim, cpuCtx, FP32::class)
        val prefillTape = trace(gctx) {
            val p = touch(promptIds)
            val f = touch(feat)
            val r = decoder.forwardPrefill(p, f, causal, zeroPad, seq, maxP, this)
            touch(r.logits)
            for (l in 0 until cfg.decoderLayers) {
                touch(r.selfK[l]); touch(r.selfV[l]); touch(r.crossK[l]); touch(r.crossV[l])
            }
            Unit
        }
        val prefillGraph = prefillTape.toComputeGraph(
            synthesizeExternalInputs = true,
            inputTensorIds = setOf(idOf(gctx, promptIds), idOf(gctx, feat)),
            embedConstants = true,
        )
        val prefillModule = StableHloConverterFactory
            .createExtended(ConstantMaterializationPolicy.ExternalAlways(scope = scope))
            .convert(prefillGraph, "whisper_prefill")

        // ---------- whisper_step ----------
        val tok = voidI(1)
        val pos = voidI(1)
        val addMask = voidF(1, 1, 1, maxP)
        val wf = voidF(1, maxP, 1)
        val selfKIn = List(cfg.decoderLayers) { voidF(1, maxP, cfg.dim) }
        val selfVIn = List(cfg.decoderLayers) { voidF(1, maxP, cfg.dim) }
        val crossKIn = List(cfg.decoderLayers) { voidF(1, cfg.audioCtx, cfg.dim) }
        val crossVIn = List(cfg.decoderLayers) { voidF(1, cfg.audioCtx, cfg.dim) }
        val stepTape = trace(gctx) {
            val t = touch(tok)
            val p = touch(pos)
            val m = touch(addMask)
            val w = touch(wf)
            val sk = ArrayList<Tensor<FP32, Float>>(cfg.decoderLayers)
            val sv = ArrayList<Tensor<FP32, Float>>(cfg.decoderLayers)
            val ck = ArrayList<Tensor<FP32, Float>>(cfg.decoderLayers)
            val cv = ArrayList<Tensor<FP32, Float>>(cfg.decoderLayers)
            for (l in 0 until cfg.decoderLayers) {
                sk += touch(selfKIn[l]); sv += touch(selfVIn[l])
                ck += touch(crossKIn[l]); cv += touch(crossVIn[l])
            }
            val r = decoder.forwardStep(t, p, m, w, sk, sv, ck, cv, maxP, this)
            touch(r.logits)
            for (l in 0 until cfg.decoderLayers) {
                touch(r.selfK[l]); touch(r.selfV[l])
            }
            Unit
        }
        val stepInputIds = buildSet {
            add(idOf(gctx, tok)); add(idOf(gctx, pos))
            add(idOf(gctx, addMask)); add(idOf(gctx, wf))
            for (l in 0 until cfg.decoderLayers) {
                add(idOf(gctx, selfKIn[l])); add(idOf(gctx, selfVIn[l]))
                add(idOf(gctx, crossKIn[l])); add(idOf(gctx, crossVIn[l]))
            }
        }
        val stepGraph = stepTape.toComputeGraph(
            synthesizeExternalInputs = true,
            inputTensorIds = stepInputIds,
            embedConstants = true,
        )
        val stepModule = StableHloConverterFactory
            .createExtended(ConstantMaterializationPolicy.ExternalAlways(scope = scope))
            .convert(stepGraph, "whisper_step")

        // ---------- write MLIR ----------
        val encFile = File(outDir, "whisper_encoder.mlir").apply { writeText(encModule.content) }
        val preFile = File(outDir, "whisper_prefill.mlir").apply { writeText(prefillModule.content) }
        val stepFile = File(outDir, "whisper_step.mlir").apply { writeText(stepModule.content) }

        // ---------- merged weights: dedup by key, assert byte identity ----------
        val merged = mergeExternalParameters(prefillModule, stepModule)
        val irpa = File(outDir, "params.irpa")
        writeIrpa(merged, irpa)

        // ---------- manifest ----------
        val manifest = File(outDir, "manifest.json").apply { writeText(manifestJson()) }

        log(
            "exported encoder=${encFile.length() / (1 shl 20)}MiB prefill=${preFile.length() / (1 shl 20)}MiB " +
                "step=${stepFile.length() / (1 shl 20)}MiB params=${irpa.length() / (1 shl 20)}MiB " +
                "(prefillExt=${prefillModule.externalParameters.size} stepExt=${stepModule.externalParameters.size} " +
                "merged=${merged.size})",
        )
        return ExportResult(
            outDir, encFile, preFile, stepFile, irpa, manifest,
            prefillModule.externalParameters.size, stepModule.externalParameters.size, merged.size,
        )
    }

    /** The prompt the runtime prefills: `<sot> <lang> <transcribe> <notimestamps>`, or `<sot> <notimestamps>` for `*.en`. */
    public fun promptIds(): IntArray {
        val special = WhisperSpecialTokens.forVocab(cfg.vocabSize)
        return if (cfg.isMultilingual) special.transcribePrompt(options.language) else intArrayOf(special.sot, special.noTimestamps)
    }

    public fun manifestJson(): String {
        val prompt = promptIds()
        require(prompt.size == options.promptLength) { "prompt has ${prompt.size} tokens, the prefill graph is built for ${options.promptLength}" }
        fun perLayer(vararg names: String): String =
            (0 until cfg.decoderLayers).joinToString(",") { l -> names.joinToString(",") { "\"l$l.$it\"" } }
        return """
        |{
        |  "contractVersion": 2,
        |  "functions": { "encoder": "whisper_encoder", "prefill": "whisper_prefill", "step": "whisper_step" },
        |  "nLayers": ${cfg.decoderLayers},
        |  "dModel": ${cfg.dim},
        |  "audioCtx": ${cfg.audioCtx},
        |  "maxPositions": ${options.maxPositions},
        |  "vocab": ${cfg.vocabSize},
        |  "eot": ${WhisperSpecialTokens.forVocab(cfg.vocabSize).eot},
        |  "promptIds": [${prompt.joinToString(",")}],
        |  "melFrames": ${cfg.melFrames},
        |  "parameters": "params.irpa",
        |  "parameterScope": "${options.scope}",
        |  "prefillArgs": ["promptIds","feat"],
        |  "prefillOutputs": ["logits",${perLayer("selfK", "selfV", "crossK", "crossV")}],
        |  "stepArgs": ["tok","pos","addMask","wf",${perLayer("selfK", "selfV", "crossK", "crossV")}],
        |  "stepOutputs": ["logits",${perLayer("selfK", "selfV")}]
        |}
        |""".trimMargin()
    }

    public companion object {
        private val json = Json { ignoreUnknownKeys = true }

        /**
         * `config.json` of a Hugging Face Whisper checkpoint → [WhisperConfig] at [audioCtx]. Geometry and vocabulary
         * come from the checkpoint (tiny, base, small, … and `*.en` alike); only the audio context is the export's.
         */
        public fun configFromHf(configJson: String, audioCtx: Int = 200): WhisperConfig {
            val top = json.parseToJsonElement(configJson).jsonObject
            fun JsonObject.int(key: String): Int = (this[key] ?: error("config.json: missing '$key'")).jsonPrimitive.int
            require(top["model_type"]?.jsonPrimitive?.content == "whisper") { "config.json is not a whisper checkpoint (model_type=${top["model_type"]})" }
            val dim = top.int("d_model")
            val heads = top.int("encoder_attention_heads")
            require(top.int("decoder_attention_heads") == heads) { "encoder and decoder head counts differ — not supported" }
            require(top.int("encoder_ffn_dim") == top.int("decoder_ffn_dim")) { "encoder and decoder FFN widths differ — not supported" }
            val cfg = WhisperConfig(
                nMels = top.int("num_mel_bins"),
                dim = dim,
                nHeads = heads,
                headDim = dim / heads,
                encoderLayers = top.int("encoder_layers"),
                decoderLayers = top.int("decoder_layers"),
                ffnDim = top.int("encoder_ffn_dim"),
                vocabSize = top.int("vocab_size"),
                maxTextPositions = top.int("max_target_positions"),
                audioCtx = audioCtx,
            )
            require(audioCtx <= top.int("max_source_positions")) { "audioCtx $audioCtx exceeds the checkpoint's max_source_positions" }
            return cfg
        }

        /** Union of both modules' external parameters, deduped by key; byte-identity is asserted. */
        public fun mergeExternalParameters(prefill: StableHloModule, step: StableHloModule): List<ExternalParameterRef> {
            val byKey = LinkedHashMap<String, ExternalParameterRef>()
            for (e in prefill.externalParameters + step.externalParameters) {
                val prev = byKey[e.key]
                if (prev == null) {
                    byKey[e.key] = e
                } else {
                    require(bytesOf(prev.source).contentEquals(bytesOf(e.source))) {
                        "external parameter key '${e.key}' has DIFFERENT bytes in prefill vs step — " +
                            "shared-session key stability broke"
                    }
                }
            }
            return byKey.values.toList()
        }

        /** Little-endian bytes of an external parameter, whatever `BufferHandle` the engine handed over (#420). */
        private fun bytesOf(h: BufferHandle): ByteArray = when (h) {
            is BufferHandle.Owned -> h.data.copyOfRange(h.offset, h.offset + h.sizeInBytes.toInt())
            is BufferHandle.Borrowed -> h.data.copyOfRange(h.offset, h.offset + h.sizeInBytes.toInt())
            else -> DefaultBufferResolver().resolve(h).use { it.readAllBytes() }
        }

        /**
         * IREE parameter archive (v0) writer.
         *
         * `sk.ainet.io.irpa.IrpaWriter` @ 0.38.0 was tried first and is REJECTED by the IREE runtime: it writes
         * `header_size=40` and pads DATA entries to 80 bytes with a 4-byte gap after `type`, but IREE's
         * `parameter_archive.h` structs are PACKED — the runtime demands `header_size=88` ("IRPA v0 header expected
         * to be exactly 88 bytes but was reported as 40") and reads `flags` at entry offset 12, `name` at 20. So the
         * harness emits the packed v0 layout itself (88-byte header, 76-byte DATA entries advanced by align16=80,
         * storage 64-aligned) — validated end-to-end by `WhisperVmfbParityTest` via
         * `iree-run-module --parameters=model=params.irpa`.
         */
        private fun writeIrpa(entries: List<ExternalParameterRef>, out: File) {
            val headerSize = 88L
            val entrySize = 76L                          // packed DATA entry
            val entryStride = 80L                        // align16(76)
            val dataAlign = 64L
            fun alignUp(v: Long, a: Long) = (v + a - 1) and (a - 1).inv()

            val keys = entries.map { it.key.encodeToByteArray() }
            val entrySegOff = alignUp(headerSize, 16)    // 96
            val entrySegLen = entries.size * entryStride
            val metaSegOff = entrySegOff + entrySegLen
            val metaSegLen = keys.sumOf { it.size.toLong() }
            val storageSegOff = alignUp(metaSegOff + metaSegLen, dataAlign)
            val storageOffsets = LongArray(entries.size)
            var cursor = 0L
            for ((i, e) in entries.withIndex()) {
                cursor = alignUp(cursor, dataAlign)
                storageOffsets[i] = cursor
                cursor += e.source.sizeInBytes
            }
            val storageSegLen = cursor

            out.outputStream().buffered(1 shl 20).use { os ->
                var written = 0L
                fun u16(v: Int) { os.write(v and 0xff); os.write((v shr 8) and 0xff); written += 2 }
                fun u32(v: Int) { for (b in 0 until 4) os.write((v shr (8 * b)) and 0xff); written += 4 }
                fun u64(v: Long) { for (b in 0 until 8) os.write(((v shr (8 * b)) and 0xff).toInt()); written += 8 }
                fun pad(to: Long) { while (written < to) { os.write(0); written++ } }

                // header prefix (packed, 32 B) + v0 header body (56 B) = 88 B
                u32(0x41505249)          // magic "IRPA"
                u16(0); u16(0)           // version 0.0
                u64(headerSize)          // header_size INCLUDING magic = 88
                u64(0)                   // next_header_offset
                u64(0)                   // flags
                u64(entries.size.toLong())
                u64(entrySegOff); u64(entrySegLen)
                u64(metaSegOff); u64(metaSegLen)
                u64(storageSegOff); u64(storageSegLen)
                pad(entrySegOff)

                // entry table: packed data entries, each padded to the 16-aligned stride
                var nameOff = 0L
                for ((i, e) in entries.withIndex()) {
                    val start = written
                    u64(entrySize)                       // entry_size (76, excl. padding)
                    u32(2)                               // type = DATA
                    u64(0)                               // flags (packed: offset 12!)
                    u64(nameOff); u64(keys[i].size.toLong())   // name ref (metadata-relative)
                    u64(0); u64(0)                       // metadata ref (none)
                    u64(dataAlign)                       // minimum_alignment
                    u64(storageOffsets[i])               // storage.offset (storage-relative)
                    u64(e.source.sizeInBytes)            // storage.length
                    pad(start + entryStride)
                    nameOff += keys[i].size
                }

                // metadata segment: concatenated key bytes
                for (k in keys) { os.write(k); written += k.size }
                pad(storageSegOff)

                // storage segment
                for ((i, e) in entries.withIndex()) {
                    pad(storageSegOff + storageOffsets[i])
                    when (val h = e.source) {
                        is BufferHandle.Owned -> os.write(h.data, h.offset, h.sizeInBytes.toInt())
                        is BufferHandle.Borrowed -> os.write(h.data, h.offset, h.sizeInBytes.toInt())
                        else -> os.write(bytesOf(h))
                    }
                    written += e.source.sizeInBytes
                }
            }
        }

        // ------------------------------------------------------------------ tracing

        private fun trace(
            ctx: DefaultGraphExecutionContext,
            block: DefaultGraphExecutionContext.() -> Unit,
        ): DefaultExecutionTape {
            val tape = ctx.record {
                val ct = currentTape ?: error("no tape")
                Execution.tapeStack.pushTape(ct)
                try {
                    block()
                } finally {
                    Execution.tapeStack.popTape()
                }
            }.first
            return tape as DefaultExecutionTape
        }

        /**
         * Identity reshape. On an INPUT tensor at the top of a trace it pins the func-arg order (first-unresolved-use
         * order); on an output at the bottom it re-taps the value as a fresh graph sink, pinning the result order.
         */
        private fun DefaultGraphExecutionContext.touch(t: Tensor<FP32, Float>): Tensor<FP32, Float> =
            ops.reshape(t, t.shape)

        private fun idOf(ctx: DefaultGraphExecutionContext, t: Tensor<*, *>): String =
            ctx.session.refOf(t).id

        private fun voidData(dims: IntArray): TensorData<Nothing, Float> = object : TensorData<Nothing, Float> {
            override val shape: Shape = Shape(*dims)
            override fun get(vararg indices: Int): Float = 0.0f
            override fun set(vararg indices: Int, value: Float) {}
        }

        @Suppress("UNCHECKED_CAST")
        private fun voidF(vararg dims: Int): Tensor<FP32, Float> =
            VoidOpsTensor(voidData(dims) as TensorData<FP32, Float>, FP32::class)

        /** Int32-typed void tensor cast to the decoder's FP32 generic (the eager idiom). */
        @Suppress("UNCHECKED_CAST")
        private fun voidI(vararg dims: Int): Tensor<FP32, Float> =
            VoidOpsTensor(voidData(dims) as TensorData<Int32, Float>, Int32::class) as Tensor<FP32, Float>
    }
}
