package sk.ainet.models.whisper

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
 * Phase 2 export harness — the whisper "contract v2" StableHLO export shared by
 * [WhisperExportTest] (G2a: MLIR smoke asserts) and [WhisperVmfbParityTest]
 * (G2b: host-CPU vmfb greedy parity via the dockerized IREE toolchain).
 *
 * Three graphs, traced through ONE [DefaultGraphExecutionContext] so all three
 * share ONE TraceSession: tensor ref ids (`t<N>`) are assigned per tensor
 * OBJECT, so the same baked weight tensor gets the same external-parameter key
 * in the prefill and step modules — the merged `params.irpa` then dedups by
 * key with byte-identity asserted (a key collision with differing bytes would
 * mean the sessions diverged and is a hard error).
 *
 *  - `whisper_encoder`  mel `[1,80,400]` → features `[1,200,384]`; weights
 *    INLINE dense constants (default materialization) — self-contained vmfb.
 *  - `whisper_prefill`  promptIds `[4]` i32 + feat `[1,200,384]` → logits
 *    `[1,4,51865]` + per layer selfK/selfV `[1,48,384]` (zero-padded to maxP)
 *    and crossK/crossV `[1,200,384]`. causalMask/zeroPad are embedded
 *    constants. Weights external (`ExternalAlways(scope="model")`).
 *  - `whisper_step`     tok `[1]` i32, pos `[1]` i32, addMask `[1,1,1,48]`,
 *    wf `[1,48,1]` + per layer selfK/selfV/crossK/crossV → logits
 *    `[1,1,51865]` + per layer updated selfK/selfV. Weights external.
 *
 * ARGUMENT AND RESULT ORDER is pinned by "touch" identity reshapes: the graph
 * builder synthesizes input placeholders in first-unresolved-use order, so
 * touching every input tensor at the top of the trace (and re-tapping every
 * output at the bottom) makes the func signature exactly the order touched:
 *
 *   prefill args    = promptIds, feat
 *   prefill results = logits, then per layer l: selfK_l, selfV_l, crossK_l, crossV_l
 *   step args       = tok, pos, addMask, wf, then per layer l: selfK_l, selfV_l, crossK_l, crossV_l
 *   step results    = logits, then per layer l: selfK_l, selfV_l
 *
 * The identity reshapes fold away in iree-compile.
 */
object WhisperExportHarness {

    val cfg = WhisperConfig() // tiny multilingual @ audioCtx=200
    const val MAX_P = 48
    const val SCOPE = "model"
    const val IMAGE = "skainet/iree-compiler:3.11.0"

    /** Golden greedy sequence recorded from the validated ONNX pipeline (G1). */
    val GOLDEN_TOKENS = intArrayOf(37, 1248, 2804, 5553, 635, 20314, 13)

    fun snapshotDir(): File? =
        System.getenv("WHISPER_SAFETENSORS")?.let { File(it) }
            ?.takeIf { File(it, "model.safetensors").exists() }

    fun exportOutDir(): File =
        File(System.getenv("WHISPER_EXPORT_OUT") ?: "build/whisper-export").absoluteFile

    data class ExportResult(
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

    /** Bake real weights, trace + convert the three graphs, write MLIR + params + manifest. */
    fun export(outDir: File): ExportResult {
        val snap = requireNotNull(snapshotDir()) { "WHISPER_SAFETENSORS not set / invalid" }
        outDir.mkdirs()

        val cpuCtx = DirectCpuExecutionContext.create()
        val encoder = WhisperEncoderModel<FP32, Float>(cfg, FP32::class)
        val decoder = WhisperDecoderModel<FP32, Float>(cfg, FP32::class)
        SafeTensorsWeightSource({ JvmRandomAccessSource.open(File(snap, "model.safetensors").path) }).use { src ->
            bakeWhisperWeights(encoder, src, cfg, FP32::class, cpuCtx as ExecutionContext)
            bakeWhisperWeights(decoder, src, cfg, FP32::class, cpuCtx)
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
        val seq = 4
        val promptIds = voidI(seq)
        val feat = voidF(1, cfg.audioCtx, cfg.dim)
        val causal = WhisperMasks.causal<FP32, Float>(seq, cpuCtx, FP32::class)
        val zeroPad = WhisperMasks.zeroPad<FP32, Float>(seq, MAX_P, cfg.dim, cpuCtx, FP32::class)
        val prefillTape = trace(gctx) {
            val p = touch(promptIds)
            val f = touch(feat)
            val r = decoder.forwardPrefill(p, f, causal, zeroPad, seq, MAX_P, this)
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
            .createExtended(ConstantMaterializationPolicy.ExternalAlways(scope = SCOPE))
            .convert(prefillGraph, "whisper_prefill")

        // ---------- whisper_step ----------
        val tok = voidI(1)
        val pos = voidI(1)
        val addMask = voidF(1, 1, 1, MAX_P)
        val wf = voidF(1, MAX_P, 1)
        val selfKIn = List(cfg.decoderLayers) { voidF(1, MAX_P, cfg.dim) }
        val selfVIn = List(cfg.decoderLayers) { voidF(1, MAX_P, cfg.dim) }
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
            val r = decoder.forwardStep(t, p, m, w, sk, sv, ck, cv, MAX_P, this)
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
            .createExtended(ConstantMaterializationPolicy.ExternalAlways(scope = SCOPE))
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

        println(
            "EXPORTED encoder=${encFile.length() / (1 shl 20)}MiB prefill=${preFile.length() / (1 shl 20)}MiB " +
                "step=${stepFile.length() / (1 shl 20)}MiB params=${irpa.length() / (1 shl 20)}MiB " +
                "(prefillExt=${prefillModule.externalParameters.size} stepExt=${stepModule.externalParameters.size} " +
                "merged=${merged.size})"
        )
        return ExportResult(
            outDir, encFile, preFile, stepFile, irpa, manifest,
            prefillModule.externalParameters.size, stepModule.externalParameters.size, merged.size,
        )
    }

    /** Union of both modules' external parameters, deduped by key; byte-identity is asserted. */
    fun mergeExternalParameters(prefill: StableHloModule, step: StableHloModule): List<ExternalParameterRef> {
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
     * Path A (`sk.ainet.io.irpa.IrpaWriter` @ 0.38.0) was tried first and is
     * REJECTED by the IREE runtime: it writes `header_size=40` and pads DATA
     * entries to 80 bytes with a 4-byte gap after `type`, but IREE's
     * `parameter_archive.h` structs are PACKED — the runtime demands
     * `header_size=88` ("IRPA v0 header expected to be exactly 88 bytes but
     * was reported as 40") and reads `flags` at entry offset 12, `name` at 20.
     * So the harness emits the packed v0 layout itself (88-byte header,
     * 76-byte DATA entries advanced by align16=80, storage 64-aligned) —
     * validated end-to-end by [WhisperVmfbParityTest] via `iree-run-module
     * --parameters=model=params.irpa`.
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

    fun manifestJson(): String {
        val prompt = WhisperSpecialTokens.forVocab(cfg.vocabSize).transcribePrompt("de")
        fun perLayer(vararg names: String): String =
            (0 until cfg.decoderLayers).joinToString(",") { l -> names.joinToString(",") { "\"l$l.$it\"" } }
        return """
        |{
        |  "contractVersion": 2,
        |  "functions": { "encoder": "whisper_encoder", "prefill": "whisper_prefill", "step": "whisper_step" },
        |  "nLayers": ${cfg.decoderLayers},
        |  "dModel": ${cfg.dim},
        |  "audioCtx": ${cfg.audioCtx},
        |  "maxPositions": $MAX_P,
        |  "vocab": ${cfg.vocabSize},
        |  "eot": ${WhisperSpecialTokens.forVocab(cfg.vocabSize).eot},
        |  "promptIds": [${prompt.joinToString(",")}],
        |  "melFrames": ${cfg.melFrames},
        |  "parameters": "params.irpa",
        |  "parameterScope": "$SCOPE",
        |  "prefillArgs": ["promptIds","feat"],
        |  "prefillOutputs": ["logits",${perLayer("selfK", "selfV", "crossK", "crossV")}],
        |  "stepArgs": ["tok","pos","addMask","wf",${perLayer("selfK", "selfV", "crossK", "crossV")}],
        |  "stepOutputs": ["logits",${perLayer("selfK", "selfV")}]
        |}
        |""".trimMargin()
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
     * Identity reshape. On an INPUT tensor at the top of a trace it pins the
     * func-arg order (first-unresolved-use order); on an output at the bottom
     * it re-taps the value as a fresh graph sink, pinning the result order.
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

    /** Int32-typed void tensor cast to the decoder's FP32 generic (the G1 eager idiom). */
    @Suppress("UNCHECKED_CAST")
    private fun voidI(vararg dims: Int): Tensor<FP32, Float> =
        VoidOpsTensor(voidData(dims) as TensorData<Int32, Float>, Int32::class) as Tensor<FP32, Float>

    // ------------------------------------------------------------------ npy I/O

    /** Write a little-endian `.npy` (v1.0). [descr] = `<f4` or `<i4`. */
    private fun writeNpy(file: File, descr: String, shape: IntArray, bytes: ByteArray) {
        val shapeStr = shape.joinToString(", ") + (if (shape.size == 1) "," else "")
        var header = "{'descr': '$descr', 'fortran_order': False, 'shape': ($shapeStr), }"
        val pad = 64 - ((10 + header.length + 1) % 64)
        header += " ".repeat(if (pad == 64) 0 else pad) + "\n"
        val hlen = header.length
        file.outputStream().buffered().use { out ->
            out.write(
                byteArrayOf(
                    0x93.toByte(), 'N'.code.toByte(), 'U'.code.toByte(), 'M'.code.toByte(),
                    'P'.code.toByte(), 'Y'.code.toByte(), 1, 0,
                    (hlen and 0xff).toByte(), ((hlen shr 8) and 0xff).toByte(),
                )
            )
            out.write(header.toByteArray(Charsets.US_ASCII))
            out.write(bytes)
        }
    }

    fun writeNpyFloat(file: File, data: FloatArray, shape: IntArray) {
        val b = ByteArray(data.size * 4)
        for (i in data.indices) {
            val v = data[i].toRawBits()
            b[i * 4] = v.toByte(); b[i * 4 + 1] = (v shr 8).toByte()
            b[i * 4 + 2] = (v shr 16).toByte(); b[i * 4 + 3] = (v shr 24).toByte()
        }
        writeNpy(file, "<f4", shape, b)
    }

    fun writeNpyInt(file: File, data: IntArray, shape: IntArray) {
        val b = ByteArray(data.size * 4)
        for (i in data.indices) {
            val v = data[i]
            b[i * 4] = v.toByte(); b[i * 4 + 1] = (v shr 8).toByte()
            b[i * 4 + 2] = (v shr 16).toByte(); b[i * 4 + 3] = (v shr 24).toByte()
        }
        writeNpy(file, "<i4", shape, b)
    }

    /** Read a contiguous little-endian float32 `.npy` (what `--output=@f.npy` writes). */
    fun readNpyFloat(file: File): FloatArray {
        val b = file.readBytes()
        require(b.size > 10 && b[0] == 0x93.toByte()) { "not a .npy: ${file.name}" }
        val headerLen = (b[8].toInt() and 0xff) or ((b[9].toInt() and 0xff) shl 8)
        val dataStart = 10 + headerLen
        val n = (b.size - dataStart) / 4
        return FloatArray(n) { i ->
            val o = dataStart + i * 4
            val bits = (b[o].toInt() and 0xff) or ((b[o + 1].toInt() and 0xff) shl 8) or
                ((b[o + 2].toInt() and 0xff) shl 16) or ((b[o + 3].toInt() and 0xff) shl 24)
            Float.fromBits(bits)
        }
    }

    fun resourceFloats(name: String): FloatArray {
        val bytes = checkNotNull(javaClass.getResourceAsStream("/whisper-de/$name")) {
            "missing resource $name"
        }.readBytes()
        return FloatArray(bytes.size / 4) { i ->
            var bits = 0
            for (b in 0 until 4) bits = bits or ((bytes[i * 4 + b].toInt() and 0xFF) shl (8 * b))
            Float.fromBits(bits)
        }
    }

    fun argmax(a: FloatArray, from: Int = 0, len: Int = a.size - from): Int {
        var best = 0
        for (i in 0 until len) if (a[from + i] > a[from + best]) best = i
        return best
    }

    // ------------------------------------------------------------------ docker

    /** Runs the pinned IREE toolchain image with [workDir] mounted at /work. */
    class Toolchain(private val workDir: File, private val image: String = IMAGE) {

        fun available(): Boolean = try {
            ProcessBuilder("docker", "image", "inspect", image)
                .redirectErrorStream(true).start()
                .also { it.inputStream.readBytes() }
                .waitFor() == 0
        } catch (_: Exception) {
            false
        }

        /** Run one toolchain subcommand; returns exit code + combined output. */
        fun run(vararg args: String): Pair<Int, String> {
            val cmd = listOf(
                "docker", "run", "--rm",
                "-v", "${workDir.absolutePath}:/work", "-w", "/work", image,
            ) + args
            val p = ProcessBuilder(cmd).redirectErrorStream(true).start()
            val out = p.inputStream.bufferedReader().readText()
            return p.waitFor() to out
        }

        fun compileCpu(mlir: String, vmfb: String): Pair<Int, String> = run(
            "compile", "--",
            "--iree-input-type=stablehlo",
            "--iree-hal-target-backends=llvm-cpu",
            "--iree-llvmcpu-target-cpu=host",
            mlir, "-o", vmfb,
        )
    }
}
