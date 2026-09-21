package sk.ainet.models.moonshine

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import sk.ainet.compile.hlo.StableHloConverterFactory
import sk.ainet.compile.hlo.toStableHlo
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tensor.Dim
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.types.FP32
import sk.ainet.tape.Execution
import java.io.DataOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * The shapes the five streaming graphs are traced with. Everything else comes from the checkpoint. The defaults
 * are a 1.28 s encoder window with 4 frames of frontend left context, and a 5.12 s utterance.
 */
public data class MoonshineV2ExportOptions(
    /** Frontend input length in samples (multiple of 80; 320 samples = one feature frame). 21760 = 68 frames. */
    val frontendSamples: Int = 21760,
    /** Encoder / adapter window in feature frames. */
    val encoderFrames: Int = 64,
    /**
     * Encoder memory the decoder attends to, in frames: the memory is padded to this length and an additive
     * cross-attention mask `[1,1,1,N]` hides the padding, so one prefill and one step graph serve every
     * utterance up to `N` frames (20 ms each).
     */
    val maxMemoryFrames: Int = 256,
    /**
     * `null` (default): the step graph's self-attention cache has a dynamic length, so one graph serves every
     * decode position. A number fixes it (one graph per position; for targets without dynamic shapes).
     */
    val staticPast: Int? = null,
) {
    init {
        require(frontendSamples > 0 && frontendSamples % MoonshineV2Frontend.FRAME == 0) { "frontendSamples must be a positive multiple of ${MoonshineV2Frontend.FRAME}" }
        require(encoderFrames > 0 && maxMemoryFrames > 0) { "encoderFrames and maxMemoryFrames must be positive" }
        require(staticPast == null || staticPast > 0) { "staticPast must be positive" }
    }
}

/**
 * Hugging Face checkpoint → the five StableHLO graphs of the Moonshine v2 **streaming** contract, plus the two
 * host-side tables a decode loop needs. Weights are folded into the graphs as `stablehlo.constant`; a compiler
 * can move them back out into a parameter archive (IREE: `--iree-opt-export-parameters`).
 *
 * | file | entry point | inputs → outputs |
 * |---|---|---|
 * | `frontend.mlir` | `@moonshine_v2_frontend` | audio `[1, samples]` → features `[1, frames, dim]` |
 * | `encoder.mlir` | `@moonshine_v2_encoder` | features `[1, frames, dim]` → memory |
 * | `adapter.mlir` | `@moonshine_v2_adapter` | memory, positions `[1, frames]` → position-aware memory (decoder width) |
 * | `prefill.mlir` | `@moonshine_v2_decoder_prefill` | token embedding `[1,1,D]`, memory `[1,N,D]`, mask `[1,1,1,N]` → logits + per-layer self/cross K/V |
 * | `with_past.mlir` | `@moonshine_v2_decoder_with_past` | token embedding, RoPE cos/sin, self K/V (dynamic), cross K/V, mask → logits + new self K/V |
 * | `dec_embed.bin` | — | token embedding table `[vocab, D]`, little-endian f32 (row lookup happens on the host) |
 * | `vocab.bin` | — | `u32` count, then per token id `u16` byte length + UTF-8 piece |
 */
public class MoonshineV2ExportHarness(
    private val checkpoint: MoonshineV2Checkpoint,
    private val options: MoonshineV2ExportOptions = MoonshineV2ExportOptions(),
    private val log: (String) -> Unit = {},
) {
    private val cfg = checkpoint.config

    public enum class Graph(public val file: String, public val function: String) {
        FRONTEND("frontend.mlir", "moonshine_v2_frontend"),
        ENCODER("encoder.mlir", "moonshine_v2_encoder"),
        ADAPTER("adapter.mlir", "moonshine_v2_adapter"),
        PREFILL("prefill.mlir", "moonshine_v2_decoder_prefill"),
        WITH_PAST("with_past.mlir", "moonshine_v2_decoder_with_past"),
    }

    /** Exports [graphs] into [outDir]. The decoder is baked once for both of its graphs. */
    public fun export(outDir: File, graphs: Set<Graph> = Graph.entries.toSet()): List<File> {
        outDir.mkdirs()
        val written = ArrayList<File>()
        if (Graph.FRONTEND in graphs) written += write(outDir, Graph.FRONTEND, frontend())
        if (Graph.ENCODER in graphs) written += write(outDir, Graph.ENCODER, encoder())
        if (Graph.ADAPTER in graphs) written += write(outDir, Graph.ADAPTER, adapter())
        if (Graph.PREFILL in graphs || Graph.WITH_PAST in graphs) {
            val dec = moonshineV2Decoder<FP32, Float>(cfg, FP32::class)
            val bakeCtx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
            val n = checkpoint.bake(dec, { MoonshineV2HfWeightMap.decoder(it, checkpoint.outputHead) }, FP32::class, bakeCtx as ExecutionContext)
            log("decoder: baked $n parameters (output head: ${checkpoint.outputHead})")
            if (Graph.PREFILL in graphs) written += write(outDir, Graph.PREFILL, prefill(dec))
            if (Graph.WITH_PAST in graphs) written += write(outDir, Graph.WITH_PAST, withPast(dec))
        }
        return written
    }

    public fun frontend(): String {
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val fe = moonshineV2Frontend<FP32, Float>(FP32::class, dim = cfg.dim)
        log("frontend: baked ${checkpoint.bake(fe, MoonshineV2HfWeightMap::frontend, FP32::class, ctx as ExecutionContext)} parameters")
        val audio = void(Shape(1, options.frontendSamples))
        // The conv1d converter lives in the extended registry only.
        return StableHloConverterFactory.createExtended()
            .convert(trace(ctx) { fe.forward(audio, it) }, Graph.FRONTEND.function).content
    }

    public fun encoder(): String {
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val enc = moonshineV2Encoder<FP32, Float>(cfg, FP32::class)
        log("encoder: baked ${checkpoint.bake(enc, MoonshineV2HfWeightMap::encoder, FP32::class, ctx as ExecutionContext)} parameters")
        val features = void(Shape(1, options.encoderFrames, cfg.dim))
        return toStableHlo(trace(ctx) { enc.forward(features, it) }, Graph.ENCODER.function).content
    }

    public fun adapter(): String {
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val adapter = MoonshineV2Adapter<FP32, Float>(cfg, maxFrames = checkpoint.maxPositions, dtype = FP32::class)
        log("adapter: baked ${checkpoint.bake(adapter, MoonshineV2HfWeightMap::encoder, FP32::class, ctx as ExecutionContext)} parameters (${checkpoint.maxPositions} positions)")
        val memory = void(Shape(1, options.encoderFrames, cfg.dim))
        val positions = void(Shape(1, options.encoderFrames))
        return toStableHlo(trace(ctx) { adapter.forward(memory, positions, it) }, Graph.ADAPTER.function).content
    }

    private fun prefill(dec: MoonshineDecoderModel<FP32, Float>): String {
        val n = options.maxMemoryFrames
        val graph = trace(DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())) {
            dec.forwardPrefill(void(Shape(1, 1, cfg.decoderDim)), void(Shape(1, n, cfg.decoderDim)), it, void(Shape(1, 1, 1, n)))
        }
        return toStableHlo(graph, Graph.PREFILL.function).content
    }

    private fun withPast(dec: MoonshineDecoderModel<FP32, Float>): String {
        val n = options.maxMemoryFrames
        val layers = cfg.decoderLayers; val heads = cfg.nHeads; val headDim = cfg.headDim
        val graph = trace(DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())) { ctx ->
            fun selfCache() = List(layers) {
                if (options.staticPast == null) void(Shape(1, heads, Dim.DYNAMIC, headDim)) else void(Shape(1, heads, options.staticPast, headDim))
            }
            fun crossCache() = List(layers) { void(Shape(1, heads, n, headDim)) }
            dec.forwardWithPast(
                void(Shape(1, 1, cfg.decoderDim)), void(Shape(1, headDim)), void(Shape(1, headDim)),
                selfCache(), selfCache(), crossCache(), crossCache(), ctx, void(Shape(1, 1, 1, n)),
            )
        }
        return toStableHlo(graph, Graph.WITH_PAST.function).content
    }

    /** `dec_embed.bin`: the token embedding table, little-endian f32, row-major `[vocab, decoderDim]`. */
    public fun writeEmbeddings(outDir: File): File {
        val table = checkpoint.floats(MoonshineV2HfWeightMap.EMBED_TOKENS)
        require(table.size == cfg.vocabSize * cfg.decoderDim) { "embedding table has ${table.size} values, expected ${cfg.vocabSize} x ${cfg.decoderDim}" }
        val bytes = ByteBuffer.allocate(table.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        bytes.asFloatBuffer().put(table)
        return File(outDir.apply { mkdirs() }, "dec_embed.bin").apply { writeBytes(bytes.array()) }
    }

    /**
     * `vocab.bin` from the snapshot's `tokenizer.json`: `u32` piece count, then per token id a `u16` byte length
     * and the UTF-8 piece (little-endian). Detokenizing is then a table lookup; the word-boundary marker U+2581
     * and the `<…>` specials are left for the reader to interpret. Only the tokenizer model's own pieces are
     * written — `added_tokens` beyond them are not text.
     */
    public fun writeVocab(outDir: File): File {
        val tokenizer = File(checkpoint.dir, "tokenizer.json").also { require(it.isFile) { "no tokenizer.json in ${checkpoint.dir}" } }
        val pieces = piecesOf(Json.parseToJsonElement(tokenizer.readText()).jsonObject)
        // The model's vocabulary may be larger than the tokenizer's (the English checkpoint: 32000 pieces, 32768
        // embedding rows — the rest are added special tokens that never detokenize to text). Never smaller.
        require(pieces.size <= cfg.vocabSize) { "tokenizer.json has ${pieces.size} pieces, more than the model's vocab_size ${cfg.vocabSize}" }
        val out = File(outDir.apply { mkdirs() }, "vocab.bin")
        DataOutputStream(out.outputStream().buffered()).use { s ->
            s.writeInt(Integer.reverseBytes(pieces.size))
            for (piece in pieces) {
                val utf8 = piece.encodeToByteArray()
                require(utf8.size < 65536) { "piece too long for a u16 length: ${utf8.size} bytes" }
                s.writeShort(java.lang.Short.reverseBytes(utf8.size.toShort()).toInt())
                s.write(utf8)
            }
        }
        return out
    }

    /** `manifest.json`: what was exported, from which geometry, with which shapes. */
    public fun writeManifest(outDir: File, files: List<File>): File {
        val manifest = buildJsonObject {
            put("model", "moonshine_streaming")
            put("config", buildJsonObject {
                put("encoder_dim", cfg.dim); put("encoder_layers", cfg.encoderLayers); put("encoder_ffn_dim", cfg.ffnDim)
                put("decoder_dim", cfg.decoderDim); put("decoder_layers", cfg.decoderLayers); put("decoder_ffn_dim", cfg.decoderFfnDim)
                put("heads", cfg.nHeads); put("head_dim", cfg.headDim); put("vocab_size", cfg.vocabSize)
                put("rope_base", cfg.ropeBase); put("partial_rotary_factor", cfg.partialRotaryFactor)
                put("max_positions", checkpoint.maxPositions)
                put("sliding_windows", JsonArray((0 until cfg.encoderLayers).map { l ->
                    buildJsonArray { add(kotlinx.serialization.json.JsonPrimitive(cfg.slidingWindowForLayer(l) - 1)); add(kotlinx.serialization.json.JsonPrimitive(cfg.rightContextForLayer(l))) }
                }))
            })
            put("shapes", buildJsonObject {
                put("frontend_samples", options.frontendSamples); put("encoder_frames", options.encoderFrames)
                put("max_memory_frames", options.maxMemoryFrames)
                options.staticPast?.let { put("static_past", it) } ?: put("past", "dynamic")
            })
            put("output_head", checkpoint.outputHead)
            put("graphs", JsonArray(Graph.entries.filter { g -> files.any { it.name == g.file } }.map { g ->
                buildJsonObject { put("file", g.file); put("function", g.function) }
            }))
            put("files", JsonArray(files.map { f -> buildJsonObject { put("file", f.name); put("bytes", f.length()) } }))
        }
        return File(outDir, "manifest.json").apply { writeText(Json { prettyPrint = true }.encodeToString(JsonObject.serializer(), manifest) + "\n") }
    }

    private fun piecesOf(tokenizer: JsonObject): List<String> {
        val vocab = tokenizer["model"]?.jsonObject?.get("vocab") ?: error("tokenizer.json has no model.vocab")
        return if (vocab is JsonArray) {
            vocab.map { it.jsonArray[0].jsonPrimitive.content }           // Unigram: [[piece, score], …] in id order
        } else {
            val byId = vocab.jsonObject.entries.associate { (piece, id) -> id.jsonPrimitive.int to piece }   // BPE: {piece: id}
            List(byId.size) { byId[it] ?: error("tokenizer.json: no piece for id $it") }
        }
    }

    private fun write(outDir: File, graph: Graph, mlir: String): File {
        check("stablehlo.constant" in mlir) { "${graph.file}: the baked weights did not fold into constants" }
        return File(outDir, graph.file).apply { writeText(mlir) }.also { log("${graph.file}: @${graph.function}, ${it.length() / 1024} KiB") }
    }

    private fun trace(ctx: DefaultGraphExecutionContext, body: (ExecutionContext) -> Unit): sk.ainet.lang.graph.ComputeGraph {
        val tape = ctx.record {
            val current = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
            Execution.tapeStack.pushTape(current)
            try { body(this as ExecutionContext) } finally { Execution.tapeStack.popTape() }
        }.first
        return (tape as DefaultExecutionTape).toComputeGraph(synthesizeExternalInputs = true, embedConstants = true)
    }

    private fun void(shape: Shape): VoidOpsTensor<FP32, Float> = VoidOpsTensor(
        object : TensorData<FP32, Float> {
            override val shape: Shape = shape
            override fun get(vararg indices: Int): Float = 0.0f
            override fun set(vararg indices: Int, value: Float) {}
        },
        FP32::class,
    )
}
