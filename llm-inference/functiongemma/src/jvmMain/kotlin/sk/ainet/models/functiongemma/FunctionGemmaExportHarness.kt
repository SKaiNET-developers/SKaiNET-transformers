package sk.ainet.models.functiongemma

import kotlinx.coroutines.runBlocking
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
import sk.ainet.lang.tensor.data.Bf16TensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.tensor.storage.BufferHandle
import sk.ainet.lang.types.FP32
import sk.ainet.models.gemma.GEMMA_DEQUANTIZE_ALL
import sk.ainet.models.gemma.GemmaWeightLoader
import sk.ainet.models.gemma.GemmaModel
import sk.ainet.models.gemma.GemmaNetworkLoader
import sk.ainet.tape.Execution
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * FunctionGemma compiled-export harness — the module's ONE export surface
 * (whisper `WhisperExportHarness` pattern), spec-driven via [FunctionGemmaSpec].
 *
 * Three graphs (each traced in its OWN context, so each writes its OWN
 * safetensors — per-trace external-parameter numbering, the PR #291 lesson;
 * board-verified: binding another trace's archive fails with NOT_FOUND on the
 * first differently-numbered `tN` key):
 *
 *  - [exportRedecode] — `func @gemma`: ONE fixed-prefill pass over `[1,seq]`
 *    ending in the DSL argMax tail (`tensor<seqxi32>` out). The shipping
 *    re-decode path. Writes `gemma-gen.mlir` + `gemma.safetensors`.
 *  - [exportPrefill] — `func @gemma_prefill`: fixed `[seq]` prompt -> per-position
 *    argMax ids + per-layer initial self K/V that seed the decode loop. Writes
 *    `gemma-prefill.mlir` + `gemma-prefill.safetensors`.
 *  - [exportWithPast] — `func @gemma_with_past`: ONE new-token step over a
 *    true-dynamic (`1x{nKV}x?x{headDim}`) self-cache; RoPE cos/sin are runtime
 *    inputs so one vmfb serves every position. Writes `gemma-with-past.mlir` +
 *    `gemma-with-past.safetensors`.
 *
 * [exportAll] runs all three plus [writeManifest] (`manifest.json` — the
 * contract the board runtime consumes; see [FunctionGemmaContract]).
 *
 * Weights are emitted as **bf16** externals by default (halving the archive;
 * bit-exact drop-in for the f16 vmfb, board A/B verified), or per-row **int8**
 * for the 2-D matmuls with [FunctionGemmaQuant.INT8] (redecode only, Phase 5).
 * The bf16/int8 MLIR-text rewrites are moved VERBATIM from kgemma's
 * `FunctionGemmaExport` — converting them to graph transforms is #248's scope,
 * and this move is behavior-identical by design (golden MLIR/archive hashes).
 *
 * Downstream: `iree-convert-parameters` turns each safetensors into an `.irpa`,
 * `iree-compile` (llvm-cpu) turns each MLIR into a vmfb — see the demo's
 * `scripts/compile-gemma.sh` and the dockerized `skainet/iree-compiler:3.11.0`.
 */
public object FunctionGemmaExportHarness {

    public data class RedecodeResult(
        val mlirPath: String,
        val safetensorsPath: String,
        val externalParamCount: Int,
        val weightMiB: Long,
        val seq: Int,
    )

    public data class ExportResult(
        val outDir: File,
        val redecode: RedecodeResult,
        val prefillMlir: File,
        val withPastMlir: File,
        val manifest: File,
    )

    // ------------------------------------------------------------------ spec API

    /** All three graphs + `manifest.json` — the one-call module export. */
    public fun exportAll(spec: FunctionGemmaSpec, outDir: String): ExportResult {
        val redecode = exportRedecode(spec, outDir)
        exportPrefill(spec, outDir)
        exportWithPast(spec, outDir)
        val manifest = writeManifest(spec, outDir)
        return ExportResult(
            outDir = File(outDir),
            redecode = redecode,
            prefillMlir = File(outDir, "gemma-prefill.mlir"),
            withPastMlir = File(outDir, "gemma-with-past.mlir"),
            manifest = manifest,
        )
    }

    /** The fixed-seq re-decode graph (`func @gemma`) from [spec]. */
    public fun exportRedecode(spec: FunctionGemmaSpec, outDir: String): RedecodeResult = export(
        gguf = spec.gguf,
        outDir = outDir,
        seq = spec.seq,
        partialRotary = spec.partialRotary,
        bf16 = spec.quant != FunctionGemmaQuant.FP32,
        quantizeInt8 = spec.quant == FunctionGemmaQuant.INT8,
    )

    /** The KV-cache prefill graph (`func @gemma_prefill`) from [spec]. */
    public fun exportPrefill(spec: FunctionGemmaSpec, outDir: String): String = exportPrefill(
        gguf = spec.gguf,
        outDir = outDir,
        seq = spec.seq,
        partialRotary = spec.partialRotary,
        bf16 = spec.quant != FunctionGemmaQuant.FP32,
    )

    /** The true-dynamic KV-cache decode graph (`func @gemma_with_past`) from [spec]. */
    public fun exportWithPast(spec: FunctionGemmaSpec, outDir: String): String = exportWithPast(
        gguf = spec.gguf,
        outDir = outDir,
        dynamicPast = true,
        partialRotary = spec.partialRotary,
        bf16 = spec.quant != FunctionGemmaQuant.FP32,
    )

    /** Write `manifest.json` (see [FunctionGemmaContract.manifestJson]). */
    public fun writeManifest(spec: FunctionGemmaSpec, outDir: String): File {
        File(outDir).mkdirs()
        return File(outDir, "manifest.json").apply {
            writeText(FunctionGemmaContract.manifestJson(spec))
        }
    }

    // ------------------------------------------------------- legacy-parameter API
    // The parameter-level surface kgemma's deprecated FunctionGemmaExport shims
    // delegate to. Bodies moved VERBATIM from kgemma (behavior-identical; golden
    // MLIR/archive equivalence is gated in FunctionGemmaExportContractTest).

    /**
     * FunctionGemma compiled-export: author `gemmaNetwork()` from the real GGUF checkpoint,
     * trace ONE fixed-prefill pass that ends in the DSL argMax tail, and emit portable
     * StableHLO with EXTERNAL params (the 270M weights live in an `.irpa`, not baked).
     *
     * The per-position argmax tail is `ops.argMax(logits, -1)` (a real DSL op), so the
     * emitted `func @gemma` already returns `tensor<seqxi32>`; weights are emitted as
     * **bf16** externals (globals + a bf16 safetensors, f32 compute via a
     * `stablehlo.convert bf16->f32` on load), halving the archive. bf16 is a bit-exact
     * drop-in for the f16 vmfb (verified board A/B).
     *
     * Writes `<outDir>/gemma-gen.mlir` + `<outDir>/gemma.safetensors`. Entry function
     * stays `gemma` (no `@main` rename).
     */
    public fun export(
        gguf: String,
        outDir: String,
        seq: Int = 24,
        partialRotary: Float = 1.0f,
        bf16: Boolean = true,
        // Phase-5 perf: quantize the big 2D matmul weights to per-row (per-output-channel) symmetric
        // int8 in the compiled graph — `tensor<AxBxi8>` global + a `tensor<Axf32>` scale, dequant'd in
        // graph (`convert i8->f32` × broadcast(scale)) into the f32 matmul. Halves the irpa vs bf16
        // (831->~415 MiB — a real RAM win on the 1.9 GB board) + halves weight-read traffic (1 B/elem).
        // Norms/1-D globals stay bf16 (sensitive). Done in the safetensors writer + a text rewrite (NOT a
        // graph pass — no OOM). Off by default; numeric quality (per-row int8 from Q5_K) + speed are on-board.
        quantizeInt8: Boolean = false,
    ): RedecodeResult = runBlocking {
        val ctx = DirectCpuExecutionContext.create()
        val weights = GemmaWeightLoader(
            randomAccessProvider = { JvmRandomAccessSource.open(gguf) },
            weightForm = GEMMA_DEQUANTIZE_ALL,
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

        // ---- Phase-5 int8 weight-only quantized export ----
        if (quantizeInt8) {
            val quantShapes = parseQuantWeightShapes(module.content)   // key -> (rows, cols) for 2D weights
            val mlirQ = rewriteGlobalsToInt8(module.content, quantShapes)
            val mlirFileQ = File(out, "gemma-gen.mlir").apply { writeText(mlirQ) }
            val stFileQ = File(out, "gemma.safetensors")
            val bytes = writeQuantizedSafetensors(ext, quantShapes, stFileQ)
            return@runBlocking RedecodeResult(
                mlirPath = mlirFileQ.absolutePath,
                safetensorsPath = stFileQ.absolutePath,
                externalParamCount = ext.size + quantShapes.size,   // + one scale per quantized weight
                weightMiB = bytes / (1024 * 1024),
                seq = seq,
            )
        }

        val mlir = if (bf16) rewriteGlobalsToBf16(module.content) else module.content
        val mlirFile = File(out, "gemma-gen.mlir").apply { writeText(mlir) }

        val stFile = File(out, "gemma.safetensors")
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

    /**
     * KV-cache `decoder_with_past` graph: trace ONE new-token step of `GemmaModel.forwardWithPast`
     * (past K/V + per-base RoPE cos/sin as graph INPUTS) ending in the DSL argMax tail, and emit
     * StableHLO `func @gemma_with_past`. This is the second board graph — driven in a loop after the
     * prefill graph, it processes 1 token/step instead of re-running the whole seq (the KV-cache win).
     *
     * I/O (per-layer): inputs `token[1]i32`, per-RoPE-base `cos/sin[1,256]`, and the 18 past self-K/V
     * `[1,nKV,?,headDim]`; outputs the 18 extended self-K/V + `token'[1]i32`. BOARD-VERIFIED
     * (SL2610, 2026-08-11): outputs are K THEN V per block; the input arg order matches the trace
     * (see llm-runtime/gemma-iree/docs/GEMMA-KV-BOARD-LOOP.md and GemmaKvDecoder.kFirstInOutput).
     *
     * KV tensor shape: pass [dynamicPast]=true for one vmfb that serves EVERY decode position via a
     * dynamic (`1x{nKV}x?x{headDim}`) self-cache seq dim, traced with a real `Dim.DYNAMIC` extent
     * (engine 0.38+ dynamic-safe tracer/emitter — no post-emit text rewrite). Set
     * `GEMMA_SENTINEL_PAST=1` to roll back to the legacy sentinel-prime trace + regex relax
     * ([SENTINEL_PAST] / [relaxSeqDimToDynamic]); see #248. [dynamicPast]=false emits a fixed [past]
     * length (static probe / fixed-pad fallback). Writes its own `gemma-with-past.safetensors`
     * (per-trace key numbering — see [writeSafetensors]).
     *
     * Returns the emitted MLIR text (also written to `<outDir>/gemma-with-past.mlir`).
     */
    public fun exportWithPast(
        gguf: String,
        outDir: String,
        past: Int = 1,
        dynamicPast: Boolean = true,
        partialRotary: Float = 1.0f,
        bf16: Boolean = true,
    ): String = runBlocking {
        val ctx = DirectCpuExecutionContext.create()
        val weights = GemmaWeightLoader(
            randomAccessProvider = { JvmRandomAccessSource.open(gguf) },
            weightForm = GEMMA_DEQUANTIZE_ALL,
        ).loadToMapStreaming<FP32, Float>(ctx, FP32::class)
        val md = weights.metadata
        val patched = weights.copy(
            metadata = md.copy(ropeParametersFull = md.ropeParametersFull.copy(partialRotaryFactor = partialRotary)),
        )
        @Suppress("UNCHECKED_CAST")
        val model = GemmaNetworkLoader.fromWeights(ctx, patched, FP32::class) as GemmaModel<FP32, Float>

        val nLayers = md.blockCount
        val headDim = md.getHeadDim(0)
        val nKV = md.kvHeadCount
        // Dynamic graphs thread a real dynamic extent (Dim.DYNAMIC) straight through the trace —
        // the engine's dynamic-safe tracer (concat/reshape propagate a dynamic dim) and emitter
        // (dynamic_broadcast_in_dim) landed in core 0.38, so no post-emit text rewrite is needed
        // and the emitted `?` dims iree-compile cleanly. DEFAULT since #248 retired the
        // sentinel-prime hack. (A `-1` placeholder is NOT an option either way: concat
        // shape-inference computes `-1 + 1 = 0` and emits broken `1x1x0x256` output caches.)
        // GEMMA_SENTINEL_PAST=1: rollback to the legacy sentinel-prime trace (concrete prime dim,
        // regex-relaxed to `?` after emit via relaxSeqDimToDynamic) in case a downstream toolchain
        // chokes on the true-dynamic IR. Scheduled for removal with the remaining MLIR-text
        // rewrites tracked in #248.
        val sentinelRollback = System.getenv("GEMMA_SENTINEL_PAST") == "1"
        val pastDim = when {
            !dynamicPast -> past
            sentinelRollback -> SENTINEL_PAST
            else -> sk.ainet.lang.tensor.Dim.DYNAMIC
        }

        val tokenId = voidF32(Shape(1))
        val cosG = voidF32(Shape(1, headDim)); val sinG = voidF32(Shape(1, headDim))
        val cosS = voidF32(Shape(1, headDim)); val sinS = voidF32(Shape(1, headDim))
        val selfKIn = List(nLayers) { voidF32(Shape(1, nKV, pastDim, headDim)) }
        val selfVIn = List(nLayers) { voidF32(Shape(1, nKV, pastDim, headDim)) }

        val tapeCtx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val tape = tapeCtx.record {
            val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
            Execution.tapeStack.pushTape(ct)
            try {
                val ectx = this as ExecutionContext
                val out = model.forwardWithPast(
                    tokenId, GemmaModel.RopeCosSin(cosG, sinG, cosS, sinS), selfKIn, selfVIn, ectx,
                )
                // token output: argMax over [1, vocab] -> [1] i32 (small-int board contract).
                ectx.ops.argMax(out.logits, dim = -1)
                // out.selfK / out.selfV are terminal (identity reshape) -> graph K/V outputs.
            } finally {
                Execution.tapeStack.popTape()
            }
        }.first

        val graph = (tape as DefaultExecutionTape).toComputeGraph(
            synthesizeExternalInputs = true, embedConstants = true,
        )
        val module = StableHloConverterFactory
            .createBasic(ConstantMaterializationPolicy.ExternalAlways(scope = "model"))
            .convert(graph, "gemma_with_past")
        var mlir = if (bf16) rewriteGlobalsToBf16(module.content) else module.content
        // Only the sentinel rollback needs the post-emit text relax; the true-dynamic default
        // emits `?` dims directly from the trace.
        if (dynamicPast && sentinelRollback) mlir = relaxSeqDimToDynamic(mlir)
        File(outDir).apply { mkdirs() }
        File(outDir, "gemma-with-past.mlir").writeText(mlir)
        // This trace's own key numbering -> its own archive (see writeSafetensors).
        writeSafetensors(module.externalParameters, File(outDir, "gemma-with-past.safetensors"), bf16)
        mlir
    }

    /**
     * LEGACY (GEMMA_SENTINEL_PAST=1 rollback only, #248): sentinel prime the with_past self-cache
     * was traced at before true-dynamic tracing became the default. Chosen well above the graph's
     * SSA node count (so `%v7919` can't collide) and unlike any real model dim (256/640/vocab/…),
     * so relaxing its two derived seq dims (`sentinel` = past, `sentinel+1` = past+1) to `?` is
     * collision-free.
     */
    private const val SENTINEL_PAST = 7919

    /** LEGACY (GEMMA_SENTINEL_PAST=1 rollback only, #248): relax the sentinel-traced self-cache
     *  seq dims (`sentinel`, `sentinel+1`, and any derived attention-score seq dim) to a dynamic
     *  `?`, so the compiled vmfb serves every decode position. The sentinel only ever appears as an
     *  interior tensor dim (`x7919x` / `x7920x`), never as an SSA index, so the textual replace is
     *  safe. Remove together with the sentinel rollback. */
    private fun relaxSeqDimToDynamic(mlir: String): String =
        mlir.replace("x${SENTINEL_PAST}x", "x?x")
            .replace("x${SENTINEL_PAST + 1}x", "x?x")

    /**
     * KV-cache PREFILL graph: trace `GemmaModel.forwardPrefill` over a FIXED [seq]-length prompt and
     * emit StableHLO `func @gemma_prefill` returning the per-position argMax token ids PLUS the
     * per-layer initial self K/V (`1x{nKV}x{seq}x{headDim}`) that seed the with_past decode loop.
     *
     * Fixed seq (not dynamic) is deliberate: the seq dim pervades the prefill graph (leading dims the
     * with_past sentinel-relax can't reach), and a fixed prefill is simpler. The prompt is zero-padded
     * to [seq]; causal masking keeps the real positions `[0, P)` from attending the padding, so the
     * board runtime slices the emitted K/V to `[1, nKV, P, headDim]` before the decode loop (the
     * padding positions' K/V are discarded). Reads the first token at position `P-1`. Runs ONCE per
     * generation. Writes its own `gemma-prefill.safetensors` (per-trace key numbering — see
     * [writeSafetensors]).
     */
    public fun exportPrefill(
        gguf: String,
        outDir: String,
        seq: Int = 24,
        partialRotary: Float = 1.0f,
        bf16: Boolean = true,
    ): String = runBlocking {
        val ctx = DirectCpuExecutionContext.create()
        val weights = GemmaWeightLoader(
            randomAccessProvider = { JvmRandomAccessSource.open(gguf) },
            weightForm = GEMMA_DEQUANTIZE_ALL,
        ).loadToMapStreaming<FP32, Float>(ctx, FP32::class)
        val md = weights.metadata
        val patched = weights.copy(
            metadata = md.copy(ropeParametersFull = md.ropeParametersFull.copy(partialRotaryFactor = partialRotary)),
        )
        @Suppress("UNCHECKED_CAST")
        val model = GemmaNetworkLoader.fromWeights(ctx, patched, FP32::class) as GemmaModel<FP32, Float>

        val tokens = voidF32(Shape(seq))   // [seq] token ids -> `{seq}xi32` graph input

        val tapeCtx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val tape = tapeCtx.record {
            val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
            Execution.tapeStack.pushTape(ct)
            try {
                val ectx = this as ExecutionContext
                val out = model.forwardPrefill(tokens, ectx)
                ectx.ops.argMax(out.logits, dim = -1)   // [seq] i32 token ids (board reads position P-1)
                // out.selfK / out.selfV are terminal -> per-layer initial K/V outputs.
            } finally {
                Execution.tapeStack.popTape()
            }
        }.first

        val graph = (tape as DefaultExecutionTape).toComputeGraph(
            synthesizeExternalInputs = true, embedConstants = true,
        )
        val module = StableHloConverterFactory
            .createBasic(ConstantMaterializationPolicy.ExternalAlways(scope = "model"))
            .convert(graph, "gemma_prefill")
        val mlir = if (bf16) rewriteGlobalsToBf16(module.content) else module.content
        File(outDir).apply { mkdirs() }
        File(outDir, "gemma-prefill.mlir").writeText(mlir)
        // This trace's own key numbering -> its own archive (see writeSafetensors).
        writeSafetensors(module.externalParameters, File(outDir, "gemma-prefill.safetensors"), bf16)
        mlir
    }

    /**
     * Write [ext] as a safetensors archive ([bf16] truncation = core parity, else raw f32). Every traced
     * graph numbers its "model" externals independently (`t0`, `t10`, …), so an archive only serves the
     * graph whose trace produced its keys — the KV graphs each write their OWN safetensors/irpa
     * (board-verified on the SL2610: binding the redecode archive to the prefill vmfb fails with
     * NOT_FOUND on the first differently-numbered `tN` key).
     */
    private fun writeSafetensors(ext: List<sk.ainet.compile.hlo.ExternalParameterRef>, stFile: File, bf16: Boolean) {
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
                writeHandle(os, e.source, bf16)
            }
        }
    }

    /**
     * Streams one external's f32 payload as raw f32 or truncating bf16 (= core parity), in 1 MiB
     * chunks. Dispatches on the [BufferHandle] subtype: the 0.53.0 engine loader delivers large
     * constants (the 262144x640 tied embedding) as [BufferHandle.Floats], not [BufferHandle.Owned] —
     * the cast that #396 removed from the Gemma 3n harness was still here (#405).
     */
    private fun writeHandle(os: java.io.OutputStream, src: BufferHandle, bf16: Boolean) {
        val n: Int
        val floatAt: (Int) -> Float
        when (src) {
            is BufferHandle.Owned -> {
                val data = src.data
                val base = src.offset
                n = (src.sizeInBytes / 4).toInt()
                if (!bf16) {
                    os.write(data, base, n * 4)
                    return
                }
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
                    val bf = Bf16TensorData.floatToBf16Bits(floatAt(j + k)) // truncation = core parity
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

    /**
     * Little-endian f32 bytes + base offset for any [BufferHandle]; a [BufferHandle.Floats] is
     * materialised once (the quantizer reads rows by byte offset). See [writeHandle] / #405.
     */
    private fun ownedBytes(src: BufferHandle): Pair<ByteArray, Int> = when (src) {
        is BufferHandle.Owned -> src.data to src.offset
        is BufferHandle.Floats -> {
            val f = src.data
            val b = ByteArray(f.size * 4)
            val bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN)
            for (x in f) bb.putFloat(x)
            b to 0
        }
        else -> error("unsupported BufferHandle ${src::class.simpleName} (${src.sizeInBytes} B)")
    }

    private fun voidF32(shape: Shape): sk.ainet.lang.tensor.Tensor<FP32, Float> =
        VoidOpsTensor(
            object : TensorData<FP32, Float> {
                override val shape = shape
                override fun get(vararg indices: Int): Float = 0.0f
                override fun set(vararg indices: Int, value: Float) {}
            },
            FP32::class,
        )

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

    /** The 2-D matmul weight globals (`tensor<rows x cols x f32>`, both > 1) to quantize to int8.
     *  1-D / `1xN` globals (RMSNorm gains, scalars) are excluded — small and precision-sensitive. */
    private fun parseQuantWeightShapes(mlir: String): Map<String, Pair<Int, Int>> {
        val re = Regex("""util\.global private @(\w+) = #flow\.parameter\.named<"[^"]*"::"[^"]*"> : tensor<(\d+)x(\d+)xf32>""")
        val map = LinkedHashMap<String, Pair<Int, Int>>()
        for (m in re.findAll(mlir)) {
            val g = m.groupValues[1]
            val rows = m.groupValues[2].toInt()
            val cols = m.groupValues[3].toInt()
            if (rows > 1 && cols > 1) map[g] = rows to cols
        }
        return map
    }

    /**
     * Quantized weight globals -> `tensor<rows x cols x i8>` + a per-row `tensor<rows x f32>` scale
     * global, dequant'd in graph (`convert i8->f32` then multiply by broadcast(scale, dims=[0])); every
     * OTHER float global -> bf16 + convert (norms). Compute stays f32.
     */
    private fun rewriteGlobalsToInt8(mlir: String, quant: Map<String, Pair<Int, Int>>): String {
        var m = mlir
        // DECLS
        m = Regex("""util\.global private @(\w+) = (#flow\.parameter\.named<"([^"]*)"::"([^"]*)"> : tensor<[0-9x]*x)f32>""")
            .replace(m) { r ->
                val g = r.groupValues[1]
                val declPrefix = r.groupValues[2]   // '#flow...named<...> : tensor<SHAPEx' (trailing x kept)
                val scope = r.groupValues[3]
                val key = r.groupValues[4]
                val qs = quant[g]
                if (qs != null) {
                    "util.global private @$g = ${declPrefix}i8>\n" +
                        "  util.global private @${g}_scale = #flow.parameter.named<\"$scope\"::\"${key}_scale\"> : tensor<${qs.first}xf32>"
                } else {
                    "util.global private @$g = ${declPrefix}bf16>"
                }
            }
        // LOADS
        m = Regex("""(%\w+) = util\.global\.load @(\w+) : tensor<([0-9x]*)xf32>""")
            .replace(m) { r ->
                val ssa = r.groupValues[1]
                val g = r.groupValues[2]
                val shape = r.groupValues[3]
                val qs = quant[g]
                if (qs != null) {
                    "${ssa}_q = util.global.load @$g : tensor<${shape}xi8>\n" +
                        "    ${ssa}_s = util.global.load @${g}_scale : tensor<${qs.first}xf32>\n" +
                        "    ${ssa}_f = stablehlo.convert ${ssa}_q : (tensor<${shape}xi8>) -> tensor<${shape}xf32>\n" +
                        "    ${ssa}_sb = stablehlo.broadcast_in_dim ${ssa}_s, dims = [0] : (tensor<${qs.first}xf32>) -> tensor<${shape}xf32>\n" +
                        "    $ssa = stablehlo.multiply ${ssa}_f, ${ssa}_sb : tensor<${shape}xf32>"
                } else {
                    "${ssa}_h = util.global.load @$g : tensor<${shape}xbf16>\n" +
                        "    $ssa = stablehlo.convert ${ssa}_h : (tensor<${shape}xbf16>) -> tensor<${shape}xf32>"
                }
            }
        return m
    }

    private fun leF32(d: ByteArray, o: Int): Float {
        val b = (d[o].toInt() and 0xFF) or ((d[o + 1].toInt() and 0xFF) shl 8) or
            ((d[o + 2].toInt() and 0xFF) shl 16) or ((d[o + 3].toInt() and 0xFF) shl 24)
        return Float.fromBits(b)
    }

    private fun putF32Le(d: ByteArray, o: Int, v: Float) {
        val b = v.toRawBits()
        d[o] = (b and 0xFF).toByte(); d[o + 1] = ((b ushr 8) and 0xFF).toByte()
        d[o + 2] = ((b ushr 16) and 0xFF).toByte(); d[o + 3] = ((b ushr 24) and 0xFF).toByte()
    }

    /**
     * Write the int8 quantized safetensors: each quant weight as per-row symmetric int8
     * (`scale[r] = max|W[r,:]| / 127`) + its `[rows]` f32 scale; every other external as bf16 (norms).
     * Streams per-weight (no full-archive buffer). Returns the total file size in bytes.
     */
    private fun writeQuantizedSafetensors(
        ext: List<sk.ainet.compile.hlo.ExternalParameterRef>,
        quant: Map<String, Pair<Int, Int>>,
        stFile: File,
    ): Long {
        // Header pass — analytic sizes (weight i8 rows*cols, scale f32 rows*4, or bf16 n*2).
        var off = 0L
        val hdr = StringBuilder("{")
        var first = true
        fun add(key: String, dtype: String, shape: String, len: Long) {
            if (!first) hdr.append(","); first = false
            hdr.append("\"$key\":{\"dtype\":\"$dtype\",\"shape\":[$shape],\"data_offsets\":[$off,${off + len}]}")
            off += len
        }
        for (e in ext) {
            val n = e.source.sizeInBytes / 4
            val qs = quant[e.key]
            if (qs != null) {
                add(e.key, "I8", "${qs.first},${qs.second}", (qs.first.toLong() * qs.second))
                add("${e.key}_scale", "F32", "${qs.first}", qs.first.toLong() * 4)
            } else {
                add(e.key, "BF16", "$n", n.toLong() * 2)
            }
        }
        hdr.append("}")
        val headerBytes = hdr.toString().encodeToByteArray()

        BufferedOutputStream(FileOutputStream(stFile), 1 shl 20).use { os ->
            os.write(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(headerBytes.size.toLong()).array())
            os.write(headerBytes)
            for (e in ext) {
                val (data, base) = ownedBytes(e.source)
                val qs = quant[e.key]
                if (qs != null) {
                    val (rows, cols) = qs
                    val q = ByteArray(rows * cols)
                    val sb = ByteArray(rows * 4)
                    for (r in 0 until rows) {
                        val rowBase = base + r * cols * 4
                        var mx = 0f
                        for (c in 0 until cols) { val a = kotlin.math.abs(leF32(data, rowBase + c * 4)); if (a > mx) mx = a }
                        val s = if (mx > 0f) mx / 127f else 1f
                        putF32Le(sb, r * 4, s)
                        val inv = if (s != 0f) 1f / s else 0f
                        for (c in 0 until cols) {
                            var qi = kotlin.math.round(leF32(data, rowBase + c * 4) * inv).toInt()
                            if (qi > 127) qi = 127 else if (qi < -127) qi = -127
                            q[r * cols + c] = qi.toByte()
                        }
                    }
                    os.write(q)
                    os.write(sb)
                } else {
                    val n = e.source.sizeInBytes.toInt() / 4
                    val ob = ByteArray(n * 2)
                    for (j in 0 until n) {
                        val bf = Bf16TensorData.floatToBf16Bits(leF32(data, base + j * 4))
                        ob[j * 2] = (bf and 0xFF).toByte()
                        ob[j * 2 + 1] = ((bf ushr 8) and 0xFF).toByte()
                    }
                    os.write(ob)
                }
            }
        }
        return 8L + headerBytes.size + off
    }
}
