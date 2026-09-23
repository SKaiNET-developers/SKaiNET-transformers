package sk.ainet.models.qwen

import kotlinx.coroutines.runBlocking
import sk.ainet.compile.hlo.ConstantMaterializationPolicy
import sk.ainet.compile.hlo.ExternalParameterRef
import sk.ainet.compile.hlo.StableHloConverterFactory
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.nn.dsl.decoder.DECODER_DEQUANTIZE_ALL
import sk.ainet.lang.nn.dsl.decoder.DecoderKvModel
import sk.ainet.lang.tensor.Dim
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
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
 * Qwen compiled-export harness (SKaiNET-transformers#411, closes the `QwenExportHarness` half of
 * issue #411) — the qwen-kv-v1 counterpart of `FunctionGemmaExportHarness`, over
 * [DecoderKvModel] instead of a hand-written per-architecture model class. Serves both
 * Qwen2.5-0.5B-Instruct and Qwen3-0.6B: [QwenKvArch] is DERIVED from the loaded GGUF, not
 * hardcoded (see [deriveArch]).
 *
 * Three graphs, following [QwenKvContract]'s arg/output orders exactly (verified against
 * `iree_kv_jni.c`'s native arg assembly — see that contract's class doc):
 *  - [exportPrefillAt] — `qwen_prefill_at`: the one-time catalog-prefix prefill, LM head applied
 *    to ONE position via a one-hot `select` row.
 *  - [exportPrefillWithPast] — `qwen_prefill_with_past`: a fixed-size chunk (an utterance,
 *    zero-padded) against the incoming cache — one call instead of C [exportWithPast] steps.
 *  - [exportWithPast] — `qwen_with_past`: one autoregressive decode step, true-dynamic past
 *    (`Dim.DYNAMIC`) so one vmfb serves every position.
 *
 * **Scope notes, both verified by tracing real checkpoints, not assumed (SKaiNET-transformers#411):**
 *
 * 1. **Per-graph archives, not merged.** Each graph writes its OWN `.safetensors`, exactly like
 *    FunctionGemma's three independent archives — the per-trace external-parameter numbering
 *    [writeSafetensors]'s doc explains means a shared archive needs the traces MERGED into one
 *    module with stable, shared parameter keys before this harness can honestly emit a single
 *    `qwen.irpa` (the `qwen-kv-v1` design's memory-saving goal, since three ~1-1.2 GB bf16
 *    archives would not fit a 32-bit process — see [QwenKvContract]'s class doc). That merge is
 *    real further work, tracked as a follow-up; until it lands, `IreeKvSession.nativeCreate`'s
 *    non-shared path (three independent sessions, Q1.1) is what a Qwen cartridge actually uses,
 *    at the three-archive memory cost FunctionGemma already pays.
 *
 * 2. **This harness's raw MLIR is pre-host-gather.** The traced graphs still contain the token
 *    embedding lookup as an in-graph `stablehlo.gather` against the embedding table (externalized
 *    as one `util.global`, exactly as [GemmaModel]'s harness produces) — NOT a separate `emb`
 *    function argument. [QwenKvContract]'s arg lists (which DO list `emb`, matching what
 *    `iree_kv_jni.c`'s native runtime actually consumes) describe the contract AFTER the
 *    blueprint plugin's host-gather MLIR rewrite (`HostGatherTask`, replacing the in-graph gather
 *    with an `%emb` argument because IREE 3.11's SPIR-V backend cannot lower it — see that
 *    plugin's rewrite and `iree_kv_jni.c`'s header comment) — a later pipeline stage, not this
 *    harness's job, same as FunctionGemma's own harness. Verified: `qwen_with_past` for
 *    Qwen3-0.6B traces to exactly `3 + 2*nLayers` args (token, cos, sin, then K/V per layer — no
 *    `emb`); `qwen_prefill_at` to `tokens, select` (2 args); `qwen_prefill_with_past` to
 *    `3 + 2*nLayers + mask + select`.
 *
 * **Known gap, found by tracing Qwen2.5-0.5B-Instruct (attnBias=true) for real, not guessed:**
 * the Q/K/V/O projection BIAS tensors do not externalize as `util.global` parameters the way
 * weight matrices do — they leak into the function signature as genuine runtime arguments
 * instead (one extra `tensor<{dim}xf32>` arg per layer, verified by inspecting the emitted MLIR:
 * `%argN` is never backed by a `util.global.load`, and its key is absent from the safetensors).
 * Qwen3-0.6B (attnBias=false) is unaffected — traced and verified clean, exact expected arg/output
 * counts, real weights in the archive. This is a SKaiNET core tracer/converter gap (bias params
 * use `ModuleParameter.BiasParameter`, weights use `ModuleParameter.WeightParameter`; only the
 * latter is recognized as an externalizable constant today), not a bug in this harness or in
 * [DecoderKvModel]'s bias-add code — no existing export harness in this codebase (FunctionGemma,
 * SmolLM2, Gemma3n) has ever traced a bias-bearing attention layer before this one. Tracked as a
 * follow-up in `NLU-QWEN-TRACKING.md`; Qwen2.5-0.5B export is blocked on it, Qwen3-0.6B is not.
 *
 * Downstream: `iree-convert-parameters` turns each safetensors into an `.irpa`, `iree-compile`
 * (llvm-cpu / vulkan-spirv) turns each MLIR into a vmfb — see `SKaiNET-iree-toolchain`'s
 * `skainet/iree-compiler:3.11.0` image.
 */
public object QwenExportHarness {

    /** SKEEP-005 phase 2: stamp the structural schedule (attention -> parallel_dims [batch, heads]); advisory header only. */
    private fun withStructuralSchedule(graph: sk.ainet.lang.graph.ComputeGraph): sk.ainet.lang.graph.ComputeGraph =
        sk.ainet.compile.opt.dagPipelineFor(
            "llvm-cpu", corePasses = listOf(sk.ainet.compile.opt.passes.ScheduleAnnotationPass("llvm-cpu")),
        ).optimize(graph).graph

    public data class ExportResult(
        val outDir: File,
        val prefillAtMlir: File,
        val prefillWithPastMlir: File,
        val withPastMlir: File,
        val manifest: File,
        val arch: QwenKvArch,
    )

    /** Load the GGUF, build the model + [DecoderKvModel], and derive [QwenKvArch] from the loaded
     *  metadata and tensor presence — the same `hasQkNorm`/`hasAttnBias` structural scan
     *  [QwenNetworkLoader.applyWeightsToNetwork] uses, so the graph this harness traces and the
     *  arch facts in the manifest always agree. */
    private fun load(gguf: String): Pair<DecoderKvModel<FP32, Float>, QwenKvArch> = runBlocking {
        val ctx = DirectCpuExecutionContext.create()
        val weights = QwenWeightLoader.loadToMapStreaming<FP32, Float>(
            ctx, { JvmRandomAccessSource.open(gguf) }, weightForm = DECODER_DEQUANTIZE_ALL,
        )
        val hasQkNorm = weights.tensors.keys.any { it.endsWith(".attn_q_norm.weight") }
        val hasAttnBias = weights.tensors.keys.any { it.endsWith(".attn_q.bias") }
        val module = QwenNetworkLoader.fromWeights(weights, kvCacheKind = sk.ainet.lang.nn.dsl.decoder.DecoderKVCacheKind.APPEND)
        val kv = DecoderKvModel<FP32, Float>(module, FP32::class)
        val md = weights.metadata
        val arch = deriveArch(
            architecture = md.architecture, nLayers = md.blockCount,
            headDim = md.ropeDimensionCount ?: (md.embeddingLength / md.headCount),
            nKvHeads = md.kvHeadCount, nHeads = md.headCount, hiddenSize = md.embeddingLength,
            vocabSize = md.vocabSize, ropeBase = md.ropeFreqBase, rmsNormEps = md.rmsNormEps,
            qkNorm = hasQkNorm, attnBias = hasAttnBias,
        )
        kv to arch
    }

    /** [QwenKvArch.eos]/[QwenKvArch.eot] are fixed vocabulary facts (`<|endoftext|>`/`<|im_end|>`,
     *  both models share them — see [QwenKvArch]'s companion), not derivable from metadata alone. */
    private fun deriveArch(
        architecture: String, nLayers: Int, headDim: Int, nKvHeads: Int, nHeads: Int,
        hiddenSize: Int, vocabSize: Int, ropeBase: Float, rmsNormEps: Float,
        qkNorm: Boolean, attnBias: Boolean,
    ): QwenKvArch = QwenKvArch(
        architecture = architecture, nLayers = nLayers, headDim = headDim, nKvHeads = nKvHeads,
        nHeads = nHeads, hiddenSize = hiddenSize, vocabSize = vocabSize, ropeBase = ropeBase,
        rmsNormEps = rmsNormEps, qkNorm = qkNorm, attnBias = attnBias,
        eos = QwenKvArch.ENDOFTEXT, eot = QwenKvArch.IM_END,
    )

    /** All three graphs + `manifest.json` — the one-call module export. */
    public fun exportAll(gguf: String, outDir: String, seq: Int, chunk: Int = QwenKvContract.DEFAULT_CHUNK, bf16: Boolean = true): ExportResult {
        val (kv, arch) = load(gguf)
        val prefillAt = exportPrefillAtTraced(kv, arch, outDir, seq, bf16)
        val prefillWithPast = exportPrefillWithPastTraced(kv, arch, outDir, chunk, bf16)
        val withPast = exportWithPastTraced(kv, arch, outDir, bf16)
        val manifest = File(outDir).apply { mkdirs() }.let { File(it, "manifest.json").apply { writeText(QwenKvContract.manifestJson(arch, chunk)) } }
        return ExportResult(File(outDir), prefillAt, prefillWithPast, withPast, manifest, arch)
    }

    /** The position-selected catalog-prefix prefill graph (`qwen_prefill_at`) from a GGUF path. */
    public fun exportPrefillAt(gguf: String, outDir: String, seq: Int, bf16: Boolean = true): File {
        val (kv, arch) = load(gguf)
        return exportPrefillAtTraced(kv, arch, outDir, seq, bf16)
    }

    /** The chunk prefill-with-past graph (`qwen_prefill_with_past`) from a GGUF path. */
    public fun exportPrefillWithPast(gguf: String, outDir: String, chunk: Int = QwenKvContract.DEFAULT_CHUNK, bf16: Boolean = true): File {
        val (kv, arch) = load(gguf)
        return exportPrefillWithPastTraced(kv, arch, outDir, chunk, bf16)
    }

    /** The true-dynamic decode-step graph (`qwen_with_past`) from a GGUF path. */
    public fun exportWithPast(gguf: String, outDir: String, bf16: Boolean = true): File {
        val (kv, arch) = load(gguf)
        return exportWithPastTraced(kv, arch, outDir, bf16)
    }

    // -------------------------------------------------------------- traced graphs

    /**
     * `qwen_prefill_at(tokens {seq}i32, emb {seq}x{hidden}f32, select 1x{seq}f32) -> per-layer
     * K,V (`1x{nKV}x{seq}x{headDim}`) THEN token 1xi32`. Fixed [seq] (zero-padded, causal-masked
     * by construction — a full uncached pass over real tokens then padding sees nothing beyond
     * itself); the caller slices real-length K/V off before use, matching FunctionGemma's
     * `exportPrefillAt`. `emb` is a graph INPUT here (the trace's own token-embedding gather),
     * turned into a real argument later by the blueprint's host-gather MLIR rewrite — see
     * [QwenKvContract]'s class doc on why the manifest lists `emb` explicitly for this contract.
     */
    private fun exportPrefillAtTraced(kv: DecoderKvModel<FP32, Float>, arch: QwenKvArch, outDir: String, seq: Int, bf16: Boolean): File {
        val tokens = voidF32(Shape(seq))
        val select = voidF32(Shape(1, seq))
        val tapeCtx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val tape = tapeCtx.record {
            val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
            Execution.tapeStack.pushTape(ct)
            try {
                val ectx = this as ExecutionContext
                val out = kv.forwardPrefillAt(tokens, select, ectx)
                ectx.ops.argMax(out.logits, dim = -1)   // [1] i32 -- the selected position's token
                // out.selfK / out.selfV are terminal (identity reshape in DecoderKvModel) -> graph outputs.
            } finally {
                Execution.tapeStack.popTape()
            }
        }.first
        val graph = (tape as DefaultExecutionTape).toComputeGraph(synthesizeExternalInputs = true, embedConstants = true)
        val module = StableHloConverterFactory
            .createBasic(ConstantMaterializationPolicy.ExternalAlways(scope = QwenKvContract.PARAMETER_SCOPE))
            .convert(withStructuralSchedule(graph), QwenKvContract.FN_PREFILL_AT)
        val mlir = if (bf16) rewriteGlobalsToBf16(module.content) else module.content
        File(outDir).mkdirs()
        val file = File(outDir, "qwen-prefill-at.mlir").apply { writeText(mlir) }
        writeSafetensors(module.externalParameters, File(outDir, "qwen-prefill-at.safetensors"), bf16)
        return file
    }

    /**
     * `qwen_prefill_with_past(tokens C i32, emb Cx{hidden}f32, cos/sin [C,headDim], per-layer
     * K/V (dynamic past), mask [1,nHeads,C,past+C] (additive, introduced once after the first
     * layer's K/V -- see [QwenKvContract.prefillWithPastArgs]), select 1xC) -> per-layer K,V
     * extended by C THEN token 1xi32`. One call per utterance instead of C [exportWithPastTraced]
     * steps. [QwenKvArch] has one RoPE base and no sliding window (every layer "global"), so
     * [DecoderKvModel.ChunkContext] carries exactly one cos/sin/mask triple -- half the arguments
     * FunctionGemma's per-layer-type version needs.
     */
    private fun exportPrefillWithPastTraced(kv: DecoderKvModel<FP32, Float>, arch: QwenKvArch, outDir: String, chunk: Int, bf16: Boolean): File {
        val nLayers = arch.nLayers
        val headDim = arch.headDim
        val nKv = arch.nKvHeads
        val nHeads = arch.nHeads

        val tokens = voidF32(Shape(chunk))
        val cos = voidF32(Shape(chunk, headDim)); val sin = voidF32(Shape(chunk, headDim))
        // Per-head [1, nHeads, C, past+C] (not per-KV-group): a broadcast to nHeads happens inside
        // the StableHLO attention converter under GQA (dims=[0,1,3,4]) -- see QwenKvContract / the
        // iree_kv_jni.c commit this mirrors. A dynamic past dim in the mask shape is fine: it is a
        // graph INPUT, not something the converter needs to broadcast a static shape onto.
        val mask = voidF32(Shape(1, nHeads, chunk, Dim.DYNAMIC))
        val select = voidF32(Shape(1, chunk))
        val selfKIn = List(nLayers) { voidF32(Shape(1, nKv, Dim.DYNAMIC, headDim)) }
        val selfVIn = List(nLayers) { voidF32(Shape(1, nKv, Dim.DYNAMIC, headDim)) }

        val tapeCtx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val tape = tapeCtx.record {
            val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
            Execution.tapeStack.pushTape(ct)
            try {
                val ectx = this as ExecutionContext
                val out = kv.forwardPrefillWithPast(
                    tokens, DecoderKvModel.ChunkContext(cos, sin, mask), select, selfKIn, selfVIn, ectx,
                )
                ectx.ops.argMax(out.logits, dim = -1)
            } finally {
                Execution.tapeStack.popTape()
            }
        }.first
        val graph = (tape as DefaultExecutionTape).toComputeGraph(synthesizeExternalInputs = true, embedConstants = true)
        val module = StableHloConverterFactory
            .createBasic(ConstantMaterializationPolicy.ExternalAlways(scope = QwenKvContract.PARAMETER_SCOPE))
            .convert(withStructuralSchedule(graph), QwenKvContract.FN_PREFILL_WITH_PAST)
        val mlir = if (bf16) rewriteGlobalsToBf16(module.content) else module.content
        File(outDir).mkdirs()
        val file = File(outDir, "qwen-prefill-with-past.mlir").apply { writeText(mlir) }
        writeSafetensors(module.externalParameters, File(outDir, "qwen-prefill-with-past.safetensors"), bf16)
        return file
    }

    /**
     * `qwen_with_past(token 1i32, emb {hidden}f32, cos/sin [1,headDim], per-layer K/V (dynamic
     * past)) -> per-layer K,V extended by 1 THEN token 1xi32`. True-dynamic past (`Dim.DYNAMIC`,
     * the engine's dynamic-safe tracer/emitter, no sentinel or post-emit text rewrite needed --
     * matches FunctionGemma's current default since #248) so one vmfb serves every decode
     * position.
     */
    private fun exportWithPastTraced(kv: DecoderKvModel<FP32, Float>, arch: QwenKvArch, outDir: String, bf16: Boolean): File {
        val nLayers = arch.nLayers
        val headDim = arch.headDim
        val nKv = arch.nKvHeads

        val tokenId = voidF32(Shape(1))
        val cos = voidF32(Shape(1, headDim)); val sin = voidF32(Shape(1, headDim))
        val selfKIn = List(nLayers) { voidF32(Shape(1, nKv, Dim.DYNAMIC, headDim)) }
        val selfVIn = List(nLayers) { voidF32(Shape(1, nKv, Dim.DYNAMIC, headDim)) }

        val tapeCtx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val tape = tapeCtx.record {
            val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
            Execution.tapeStack.pushTape(ct)
            try {
                val ectx = this as ExecutionContext
                val out = kv.forwardWithPast(tokenId, DecoderKvModel.RopeCosSin(cos, sin), selfKIn, selfVIn, ectx)
                ectx.ops.argMax(out.logits, dim = -1)
            } finally {
                Execution.tapeStack.popTape()
            }
        }.first
        val graph = (tape as DefaultExecutionTape).toComputeGraph(synthesizeExternalInputs = true, embedConstants = true)
        val module = StableHloConverterFactory
            .createBasic(ConstantMaterializationPolicy.ExternalAlways(scope = QwenKvContract.PARAMETER_SCOPE))
            .convert(withStructuralSchedule(graph), QwenKvContract.FN_WITH_PAST)
        val mlir = if (bf16) rewriteGlobalsToBf16(module.content) else module.content
        File(outDir).mkdirs()
        val file = File(outDir, "qwen-with-past.mlir").apply { writeText(mlir) }
        writeSafetensors(module.externalParameters, File(outDir, "qwen-with-past.safetensors"), bf16)
        return file
    }

    private fun voidF32(shape: Shape): Tensor<FP32, Float> =
        VoidOpsTensor(
            object : TensorData<FP32, Float> {
                override val shape = shape
                override fun get(vararg indices: Int): Float = 0.0f
                override fun set(vararg indices: Int, value: Float) {}
            },
            FP32::class,
        )

    // ------------------------------------------------------ safetensors + bf16 rewrite
    // Verbatim from FunctionGemmaExportHarness (also duplicated in SmolLm2ExportHarness): each
    // module owns its own copy of this generic, architecture-agnostic machinery rather than
    // sharing it, matching the established pattern in this codebase.

    /** Write [ext] as a safetensors archive ([bf16] truncation = core parity, else raw f32). */
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
            for (e in ext) writeHandle(os, e.source, bf16)
        }
    }

    /** Streams one external's f32 payload as raw f32 or truncating bf16, in 1 MiB chunks. */
    private fun writeHandle(os: java.io.OutputStream, src: BufferHandle, bf16: Boolean) {
        val n: Int
        val floatAt: (Int) -> Float
        when (src) {
            is BufferHandle.Owned -> {
                val data = src.data
                val base = src.offset
                n = (src.sizeInBytes / 4).toInt()
                if (!bf16) { os.write(data, base, n * 4); return }
                floatAt = { j ->
                    val o = base + j * 4
                    Float.fromBits(
                        (data[o].toInt() and 0xFF) or ((data[o + 1].toInt() and 0xFF) shl 8) or
                            ((data[o + 2].toInt() and 0xFF) shl 16) or ((data[o + 3].toInt() and 0xFF) shl 24),
                    )
                }
            }
            is BufferHandle.Floats -> {
                val f = src.data
                n = f.size
                floatAt = { j -> f[j] }
            }
            else -> error("unsupported BufferHandle ${src::class.simpleName} (${src.sizeInBytes} B)")
        }
        val bpe = if (bf16) 2 else 4
        val chunk = (1 shl 20) / bpe
        val buf = ByteArray(chunk * bpe)
        var j = 0
        while (j < n) {
            val m = minOf(chunk, n - j)
            if (bf16) {
                for (k in 0 until m) {
                    val bf = Bf16TensorData.floatToBf16Bits(floatAt(j + k))
                    buf[k * 2] = (bf and 0xFF).toByte()
                    buf[k * 2 + 1] = ((bf ushr 8) and 0xFF).toByte()
                }
            } else {
                val bb = ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN)
                for (k in 0 until m) bb.putFloat(floatAt(j + k))
            }
            os.write(buf, 0, m * bpe)
            j += m
        }
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
