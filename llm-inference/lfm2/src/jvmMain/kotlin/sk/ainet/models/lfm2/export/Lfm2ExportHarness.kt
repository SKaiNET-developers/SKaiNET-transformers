package sk.ainet.models.lfm2.export

import kotlinx.serialization.json.*
import sk.ainet.models.lfm2.*
import java.io.File

/** Fixed-shape host export of a verified, already loaded checkpoint. No reference fixtures are needed. */
class Lfm2ExportHarness(private val loaded: LoadedLfm2Model) {
    private val dimensions get() = loaded.model.dimensions

    fun prefill(length: Int): Lfm2TracedFunction = traceLfm2Prefill(Lfm2PrefillNetwork(dimensions, loaded.weights), length)

    fun shortConvolution(layer: Int, length: Int): Lfm2TracedFunction {
        requireLayer(layer, "conv", length)
        return traceLfm2ShortConv(convolution(layer), length)
    }

    fun convolutionBlock(layer: Int, length: Int): Lfm2TracedFunction {
        requireLayer(layer, "conv", length)
        val d = dimensions
        fun w(name: String) = loaded.weights.getValue("model.layers.$layer.$name.weight")
        val op = BlockTensorOp(d.width, d.kernel, d.intermediate, d.epsilon, convolution(layer),
            w("operator_norm"), w("ffn_norm"), w("feed_forward.w1"), w("feed_forward.w2"), w("feed_forward.w3"), RmsReductionPolicy.BALANCED_FP32)
        return traceLfm2ConvBlock(op, length)
    }

    fun attentionBlock(layer: Int, past: Int, length: Int,
        layoutPolicy: AttentionLayoutPolicy = AttentionLayoutPolicy.CURRENT_BATCHED,
    ): Lfm2TracedFunction {
        requireLayer(layer, "full_attention", length)
        require(past >= 0 && past.toLong() + length <= dimensions.contextLimit)
        val d = dimensions
        fun w(name: String) = loaded.weights.getValue("model.layers.$layer.$name.weight")
        val op = AttentionTensorOp(d.width, d.queryHeads, d.kvHeads, d.headDim, d.intermediate, d.epsilon,
            w("operator_norm"), w("ffn_norm"), w("self_attn.q_proj"), w("self_attn.k_proj"), w("self_attn.v_proj"), w("self_attn.out_proj"),
            w("self_attn.q_layernorm"), w("self_attn.k_layernorm"), w("feed_forward.w1"), w("feed_forward.w2"), w("feed_forward.w3"),
            RmsReductionPolicy.BALANCED_FP32, layoutPolicy)
        return traceLfm2Attention(op, past, length)
    }

    /** Exact source payload hashes for one of this harness's export modes. */
    fun weightHashes(kind: Lfm2ExportKind, layer: Int? = null): List<String> {
        if (kind == Lfm2ExportKind.PREFILL) return loaded.records.map { it.fp32Sha256 }
        require(layer != null && layer in dimensions.layerTypes.indices)
        val prefix = "model.layers.$layer."
        val expectedKind = if (kind == Lfm2ExportKind.ATTENTION) "full_attention" else "conv"
        require(dimensions.layerTypes[layer] == expectedKind)
        return loaded.records.filter {
            it.tensorName.startsWith(prefix) && (kind != Lfm2ExportKind.SHORT_CONV || it.tensorName.startsWith(prefix + "conv."))
        }.map { it.fp32Sha256 }
    }

    private fun convolution(layer: Int): ShortConvTensorOp {
        fun w(name: String) = loaded.weights.getValue("model.layers.$layer.conv.$name.weight")
        return ShortConvTensorOp(dimensions.width, dimensions.kernel, w("in_proj"), w("conv"), w("out_proj"))
    }

    private fun requireLayer(layer: Int, kind: String, length: Int) {
        require(layer in dimensions.layerTypes.indices && dimensions.layerTypes[layer] == kind) { "Expected a $kind layer" }
        require(length in 1..dimensions.contextLimit)
    }
}

enum class Lfm2ExportKind { PREFILL, SHORT_CONV, CONV_BLOCK, ATTENTION }

/** CLI entry point. All inputs are local; checkpoint identity is mandatory and nothing is downloaded. */
fun main(args: Array<String>) {
    if (args.contentEquals(arrayOf("--help"))) {
        println("LFM2 export: --kind prefill|short-conv|conv-block|attention --config FILE --checkpoint FILE --checkpoint-sha256 SHA256 --length N --output NEW_DIRECTORY [--layer N] [--past N] [--context-limit N] [--attention-layout CURRENT_BATCHED|DIRECT_KV_PER_HEAD]")
        return
    }
    require(args.size % 2 == 0) { "Expected --name value pairs; use --help" }
    val options = linkedMapOf<String, String>()
    val allowed = setOf("kind", "config", "checkpoint", "checkpoint-sha256", "length", "output", "layer", "past", "context-limit", "attention-layout")
    for (pair in args.toList().chunked(2)) {
        require(pair[0].startsWith("--"))
        val key = pair[0].removePrefix("--")
        require(key in allowed && options.put(key, pair[1]) == null) { "Unknown or repeated option: $key" }
    }
    fun required(name: String) = options[name] ?: error("Missing --$name")
    val kind = Lfm2ExportKind.valueOf(required("kind").replace('-', '_').uppercase())
    val length = required("length").toInt()
    val layer = options["layer"]?.toInt()
    val past = options["past"]?.toInt() ?: 0
    require(kind != Lfm2ExportKind.PREFILL || layer == null) { "Prefill does not select a layer" }
    require(kind == Lfm2ExportKind.PREFILL || layer != null) { "A component export requires --layer" }
    require(kind == Lfm2ExportKind.ATTENTION || ("past" !in options && "attention-layout" !in options))
    val layout = options["attention-layout"]?.let(AttentionLayoutPolicy::valueOf) ?: AttentionLayoutPolicy.CURRENT_BATCHED
    val configBytes = File(required("config")).readBytes()
    val config = Lfm2Config.parse(configBytes.toString(Charsets.UTF_8))
    val contextLimit = options["context-limit"]?.toInt() ?: minOf(2048, config.maximumPositions)
    require(length > 0 && past >= 0 && past.toLong() + length <= contextLimit)
    val output = File(required("output"))
    require(!output.exists()) { "Use a new export directory" }
    val loaded = loadLfm2Model(config, File(required("checkpoint")), contextLimit, required("checkpoint-sha256"))
    val harness = Lfm2ExportHarness(loaded)
    val trace = when (kind) {
        Lfm2ExportKind.PREFILL -> harness.prefill(length)
        Lfm2ExportKind.SHORT_CONV -> harness.shortConvolution(layer!!, length)
        Lfm2ExportKind.CONV_BLOCK -> harness.convolutionBlock(layer!!, length)
        Lfm2ExportKind.ATTENTION -> harness.attentionBlock(layer!!, past, length, layout)
    }
    val provenance = buildJsonObject {
        put("kind", kind.name); put("checkpoint_sha256", loaded.checkpointSha256); put("config_sha256", sha256(configBytes))
        put("chunk_length", length); put("past_length", past); put("context_limit", contextLimit)
        layer?.let { put("layer", it) }
        put("rms_reduction_policy", RmsReductionPolicy.BALANCED_FP32.name)
        put("attention_layout_policy", layout.name)
        put("config", Json.parseToJsonElement(configBytes.toString(Charsets.UTF_8)))
    }
    val manifest = writeLfm2ExportBundle(trace, output, harness.weightHashes(kind, layer), provenance)
    println(Json { prettyPrint = true }.encodeToString(JsonObject.serializer(), manifest))
}
