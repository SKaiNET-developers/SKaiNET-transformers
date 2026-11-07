package sk.ainet.models.lfm2.export

import kotlinx.serialization.json.*
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.dsl.decoder.DecodeStepExport
import sk.ainet.lang.nn.dsl.decoder.ExportedDecodeStep
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Q4_0Quantizer
import sk.ainet.lang.tensor.data.Q8_0Quantizer
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.trace.PackedConstantHandling
import sk.ainet.lang.types.FP32
import sk.ainet.models.lfm2.AttentionLayoutPolicy
import sk.ainet.models.lfm2.Lfm2Config
import sk.ainet.models.lfm2.Lfm2DecodeStep
import sk.ainet.models.lfm2.ShortConvLowering
import sk.ainet.models.lfm2.lfm2ProjectionsAs
import sk.ainet.models.lfm2.loadLfm2Model
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

/** The step's argument order before the state: `embedded`, `cos`, `sin`, `mask`, `write_mask`. */
fun lfm2DecodeStepInputs(step: Lfm2DecodeStep): List<Pair<String, Shape>> {
    val d = step.dimensions
    return listOf(
        "embedded" to Shape(1, d.width),
        "cos" to Shape(1, d.headDim),
        "sin" to Shape(1, d.headDim),
        "mask" to Shape(1, step.capacity + 1),
        "write_mask" to Shape(1, step.capacity, 1),
    ) + step.stateShapes().map { (name, shape) -> name to shape }
}

/**
 * Trace [step] through [DecodeStepExport.exportPorts]: one graph for every position up to the
 * capacity, results `logits` then every state. With [PackedConstantHandling.KEEP_PACKED] the
 * block-quantized projections and head stay packed and are folded.
 */
fun exportLfm2DecodeStep(
    step: Lfm2DecodeStep,
    packedConstants: PackedConstantHandling = PackedConstantHandling.KEEP_PACKED,
): ExportedDecodeStep = DecodeStepExport.exportPorts<FP32, Float>(lfm2DecodeStepInputs(step), FP32::class, "decode_step", packedConstants) { p, ctx ->
    val states = step.stateShapes().keys.associateWith { p.getValue(it) }
    val r = step.forward(p.getValue("embedded"), p.getValue("cos"), p.getValue("sin"), p.getValue("mask"), p.getValue("write_mask"), states, ctx)
    linkedMapOf("logits" to r.logits).apply { putAll(r.states) }
}

/** [embedding] re-encoded as [encoding] (Q4_0 or Q8_0) for a separate output head, or the tensor itself for null. */
@Suppress("UNCHECKED_CAST")
fun lfm2Head(embedding: Tensor<FP32, Float>, encoding: TensorEncoding?, ctx: ExecutionContext): Tensor<FP32, Float> {
    if (encoding == null) return embedding
    val dense = (embedding.data as? PackedBlockStorage)?.toFloatArray() ?: embedding.data.copyToFloatArray()
    val data = when (encoding) {
        TensorEncoding.Q4_0 -> Q4_0Quantizer.quantize(dense, embedding.shape)
        TensorEncoding.Q8_0 -> Q8_0Quantizer.quantize(dense, embedding.shape)
        else -> throw IllegalArgumentException("Unsupported head encoding ${encoding.name}")
    }
    return ctx.fromData(data as TensorData<FP32, Float>, FP32::class)
}

/** Projections the decode step folds: five per convolution layer, seven per attention layer, plus the head. */
fun lfm2FoldableMatmuls(layerTypes: List<String>): Int = layerTypes.sumOf { if (it == "conv") 5 else 7 } + 1

/**
 * Local checkpoint export of the NPU decode step: no download, compilation or device execution.
 * Writes `decode_step.mlir`, `io.txt` (the port table), `embed.bf16.bin` (the host's embedding
 * rows, `[vocabulary, width]` bf16 little-endian) and `export.json`.
 */
object Lfm2DecodeStepExportCli {
    @JvmStatic
    fun main(args: Array<String>) {
        if (args.isEmpty() || args.contentEquals(arrayOf("--help"))) {
            println("LFM2 decode-step export: --config FILE --config-sha256 SHA256 --checkpoint FILE --checkpoint-sha256 SHA256 " +
                "--capacity N --output NEW_DIRECTORY [--weights q8_0|q4_0|f32] [--head q4_0|q8_0|tied] [--conv multiply-sum|depthwise] [--attention batched|per-head] [--parameters inline|external-bf16]")
            return
        }
        val allowed = setOf("--config", "--config-sha256", "--checkpoint", "--checkpoint-sha256", "--capacity", "--output",
            "--weights", "--head", "--conv", "--attention", "--parameters")
        require(args.size % 2 == 0) { "Every option requires one value; use --help" }
        val options = linkedMapOf<String, String>()
        for (pair in args.toList().chunked(2)) {
            require(pair[0] in allowed && options.put(pair[0], pair[1]) == null) { "Unknown or repeated option ${pair[0]}" }
        }
        fun required(name: String): String = requireNotNull(options[name]) { "Missing $name" }
        fun pin(name: String): String = required(name).also {
            require(it.matches(Regex("[0-9a-f]{64}"))) { "$name must be lowercase SHA-256" }
        }
        fun encoding(value: String, name: String): TensorEncoding? = when (value) {
            "q8_0" -> TensorEncoding.Q8_0
            "q4_0" -> TensorEncoding.Q4_0
            "f32", "tied" -> null
            else -> throw IllegalArgumentException("Invalid $name $value")
        }
        val configFile = File(required("--config"))
        val checkpoint = File(required("--checkpoint"))
        val configPin = pin("--config-sha256")
        val checkpointPin = pin("--checkpoint-sha256")
        val capacity = requireNotNull(required("--capacity").toIntOrNull()) { "Invalid --capacity" }
        val weightsName = options["--weights"] ?: "q8_0"
        val headName = options["--head"] ?: "q4_0"
        val weights = encoding(weightsName, "--weights")
        val head = encoding(headName, "--head")
        val conv = when (options["--conv"] ?: "multiply-sum") {
            "multiply-sum" -> ShortConvLowering.MULTIPLY_SUM
            "depthwise" -> ShortConvLowering.DEPTHWISE
            else -> throw IllegalArgumentException("Invalid --conv")
        }
        val attention = when (options["--attention"] ?: "batched") {
            "per-head" -> AttentionLayoutPolicy.DIRECT_KV_PER_HEAD
            "batched" -> AttentionLayoutPolicy.CURRENT_BATCHED
            else -> throw IllegalArgumentException("Invalid --attention")
        }
        val output = File(required("--output"))
        require(!output.exists() && output.absoluteFile.parentFile.isDirectory) { "Use a new output directory beneath an existing parent" }
        require(configFile.isFile && configFile.canRead() && sha256(configFile) == configPin) { "Configuration SHA-256 mismatch or unreadable configuration" }
        val config = Lfm2Config.parse(configFile.readText())
        require(capacity in 1..config.maximumPositions) { "Capacity exceeds the checkpoint position limit" }

        val loaded = loadLfm2Model(config, checkpoint, contextLimit = capacity, expectedCheckpointSha256 = checkpointPin,
            weightFormFor = weights?.let(::lfm2ProjectionsAs))
        val embedding = loaded.weights.getValue("model.embed_tokens.weight")
        val step = Lfm2DecodeStep(loaded.model.dimensions, loaded.weights, capacity, lfm2Head(embedding, head, loaded.model.ctx), conv, attention)
        if ((options["--parameters"] ?: "inline") == "external-bf16") {
            require(weights == null && head == null) { "--parameters external-bf16 takes dense weights: --weights f32 --head tied" }
            output.mkdirs()
            val trace = traceLfm2DecodeStepExternal(step)
            File(output, "decode_step.mlir").writeText(writeLfm2Bf16Parameters(trace, File(output, "model.bf16.safetensors")))
            File(output, "io.txt").writeText(trace.ioTable())
            val embedFile = File(output, "embed.bf16.bin")
            writeBf16(embedding, embedFile)
            val archive = File(output, "model.bf16.safetensors")
            val d = loaded.model.dimensions
            val manifest = buildJsonObject {
                put("schema_version", 1); put("kind", "lfm2_decode_step_export"); put("function_name", "decode_step")
                put("config_sha256", configPin); put("checkpoint_sha256", loaded.checkpointSha256)
                put("capacity", capacity); put("weights", "external-bf16"); put("head", "tied")
                put("conv_lowering", conv.name.lowercase()); put("attention_layout", attention.name.lowercase())
                put("parameters", buildJsonObject {
                    put("file", archive.name); put("scope", "model"); put("dtype", "bf16"); put("count", trace.module.externalParameters.size)
                    put("sha256", sha256(archive))
                })
                put("dimensions", buildJsonObject {
                    put("vocabulary", d.vocabulary); put("width", d.width); put("head_dimension", d.headDim)
                    put("kv_heads", d.kvHeads); put("conv_kernel", d.kernel); put("rope_theta", d.theta)
                    put("layer_types", JsonArray(d.layerTypes.map(::JsonPrimitive)))
                })
                put("embedding_rows", buildJsonObject {
                    put("file", embedFile.name); put("dtype", "bf16"); put("shape", JsonArray(listOf(d.vocabulary, d.width).map(::JsonPrimitive)))
                    put("sha256", sha256(embedFile))
                })
                put("host_control", "embedding_lookup_rope_masks_position_and_stop_policy")
            }
            File(output, "export.json").writeText(Json { prettyPrint = true }.encodeToString(JsonObject.serializer(), manifest) + "\n")
            println("Exported decode_step with ${trace.module.externalParameters.size} external bf16 parameters: capacity=$capacity, " +
                "conv=${conv.name.lowercase()}, attention=${attention.name.lowercase()}; convert with iree-convert-parameters")
            return
        }
        val exported = exportLfm2DecodeStep(step)
        val skipped = exported.diagnostics.filter { it.startsWith("BLOCK_QUANT_SKIPPED") }
        require(skipped.isEmpty()) { "Packed weights the fusion could not fold:\n" + skipped.joinToString("\n") }
        val expectedFolds = when {
            weights == null && head == null -> 0
            weights == null -> 1
            head == null -> lfm2FoldableMatmuls(loaded.model.dimensions.layerTypes) - 1
            else -> lfm2FoldableMatmuls(loaded.model.dimensions.layerTypes)
        }
        check(exported.foldedBlockQuantMatmuls == expectedFolds) {
            "Folded ${exported.foldedBlockQuantMatmuls} block-quant matmuls, expected $expectedFolds"
        }

        output.mkdirs()
        File(output, "decode_step.mlir").writeText(exported.mlir)
        File(output, "io.txt").writeText(exported.ioTable())
        val embedFile = File(output, "embed.bf16.bin")
        writeBf16(embedding, embedFile)
        val d = loaded.model.dimensions
        val manifest = buildJsonObject {
            put("schema_version", 1); put("kind", "lfm2_decode_step_export"); put("function_name", "decode_step")
            put("config_sha256", configPin); put("checkpoint_sha256", loaded.checkpointSha256)
            put("capacity", capacity); put("weights", weightsName); put("head", headName)
            put("conv_lowering", conv.name.lowercase()); put("attention_layout", attention.name.lowercase()); put("folded_block_quant_matmuls", exported.foldedBlockQuantMatmuls)
            put("dimensions", buildJsonObject {
                put("vocabulary", d.vocabulary); put("width", d.width); put("head_dimension", d.headDim)
                put("kv_heads", d.kvHeads); put("conv_kernel", d.kernel); put("rope_theta", d.theta)
                put("layer_types", JsonArray(d.layerTypes.map(::JsonPrimitive)))
            })
            put("embedding_rows", buildJsonObject {
                put("file", embedFile.name); put("dtype", "bf16"); put("shape", JsonArray(listOf(d.vocabulary, d.width).map(::JsonPrimitive)))
                put("sha256", sha256(embedFile))
            })
            put("host_control", "embedding_lookup_rope_masks_position_and_stop_policy")
        }
        File(output, "export.json").writeText(Json { prettyPrint = true }.encodeToString(JsonObject.serializer(), manifest) + "\n")
        println("Exported decode_step: capacity=$capacity, weights=$weightsName, head=$headName, conv=${conv.name.lowercase()}, " +
            "folded=${exported.foldedBlockQuantMatmuls}; compilation and device execution remain unqualified")
    }

    /** Round-to-nearest-even bf16 of [tensor]'s values, little-endian. */
    private fun writeBf16(tensor: Tensor<FP32, Float>, file: File) {
        val values = (tensor.data as? PackedBlockStorage)?.toFloatArray() ?: tensor.data.copyToFloatArray()
        FileOutputStream(file).buffered(1 shl 20).use { out ->
            val buffer = ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN)
            for (v in values) {
                val bits = java.lang.Float.floatToRawIntBits(v)
                val rounded = (bits + 0x7FFF + ((bits ushr 16) and 1)) ushr 16
                buffer.clear(); buffer.putShort(rounded.toShort()); out.write(buffer.array())
            }
        }
    }

    private fun sha256(file: File): String = file.inputStream().buffered().use { input ->
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(1 shl 20)
        while (true) {
            val n = input.read(buffer)
            if (n < 0) break
            digest.update(buffer, 0, n)
        }
        digest.digest().joinToString("") { "%02x".format(it) }
    }
}

/**
 * [step] traced with every weight as an external `model`-scope parameter, for a compiler that serves the weights from
 * an archive (IREE `--parameters`, e.g. a GPU): `module` keeps f32 types and `inputNames`/`outputNames` the step's
 * ports in argument order. Use [writeLfm2Bf16Parameters] to store the weights as bf16.
 */
fun traceLfm2DecodeStepExternal(step: Lfm2DecodeStep): Lfm2TracedFunction =
    traceLfm2Function("decode_step", LinkedHashMap(lfm2DecodeStepInputs(step).toMap())) { p, ctx ->
        val states = step.stateShapes().keys.associateWith { p.getValue(it) }
        val r = step.forward(p.getValue("embedded"), p.getValue("cos"), p.getValue("sin"), p.getValue("mask"), p.getValue("write_mask"), states, ctx)
        linkedMapOf("logits" to r.logits).apply { putAll(r.states) }
    }

/**
 * Writes [trace]'s external parameters to [archive] as flat BF16 safetensors (round to nearest even; exact for a
 * bf16 checkpoint) and returns the module text with every parameter global typed bf16 and converted to f32 where it
 * is loaded, so the graph computes as before and the weights take half the memory.
 */
fun writeLfm2Bf16Parameters(trace: Lfm2TracedFunction, archive: File): String {
    require(!archive.exists()) { "Parameter archive already exists: ${archive.name}" }
    val resolver = sk.ainet.lang.tensor.storage.DefaultBufferResolver()
    val params = trace.module.externalParameters
    require(params.isNotEmpty() && params.all { it.scope == "model" })
    val sizes = params.map { p -> resolver.resolve(p.source).use { it.sizeInBytes } }
    require(sizes.all { it > 0 && it % 4 == 0L })
    var offset = 0L
    val header = buildJsonObject {
        for ((i, p) in params.withIndex()) {
            val n = sizes[i] / 4
            put(p.key, buildJsonObject {
                put("dtype", "BF16"); put("shape", buildJsonArray { add(n) })
                put("data_offsets", buildJsonArray { add(offset); add(offset + 2 * n) })
            })
            offset += 2 * n
        }
    }.toString().toByteArray(Charsets.UTF_8)
    val padded = header.size + (8 - header.size % 8) % 8
    FileOutputStream(archive).buffered(1 shl 20).use { out ->
        out.write(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(padded.toLong()).array())
        out.write(header)
        repeat(padded - header.size) { out.write(32) }
        for ((i, p) in params.withIndex()) resolver.resolve(p.source).use { source ->
            var position = 0L
            while (position < sizes[i]) {
                val count = minOf(1L shl 20, sizes[i] - position).toInt()
                val f32 = ByteBuffer.wrap(source.readBytes(position, count)).order(ByteOrder.LITTLE_ENDIAN)
                val bf16 = ByteBuffer.allocate(count / 2).order(ByteOrder.LITTLE_ENDIAN)
                while (f32.hasRemaining()) {
                    val bits = f32.int
                    bf16.putShort(((bits + 0x7FFF + ((bits ushr 16) and 1)) ushr 16).toShort())
                }
                out.write(bf16.array())
                position += count
            }
        }
    }
    var mlir = Regex("""(util\.global private @[\w$.-]+ = #flow\.parameter\.named<"[^"]*"::"[^"]*"> : tensor<[0-9x]*x)f32>""")
        .replace(trace.module.content) { it.groupValues[1] + "bf16>" }
    mlir = Regex("""(%[\w$.-]+) = util\.global\.load @([\w$.-]+) : tensor<([0-9x]*)xf32>""").replace(mlir) { r ->
        val (ssa, g, shape) = r.destructured
        "${ssa}_bf16 = util.global.load @$g : tensor<${shape}xbf16>\n    $ssa = stablehlo.convert ${ssa}_bf16 : (tensor<${shape}xbf16>) -> tensor<${shape}xf32>"
    }
    val converted = Regex("""util\.global\.load @[\w$.-]+ : tensor<[0-9x]*xbf16>""").findAll(mlir).count()
    check(converted == params.size) { "Retyped $converted of ${params.size} parameter loads; the exporter's global form changed" }
    return mlir
}

/** `io.txt` for a traced step: `in <i> <name> [dims]` in argument order, then `out <i> <name> [dims]`. */
fun Lfm2TracedFunction.ioTable(): String =
    (inputNames.zip(inputShapes).mapIndexed { i, (n, s) -> "in $i $n ${s.dimensions.toList()}" } +
        outputNames.zip(outputShapes).mapIndexed { i, (n, s) -> "out $i $n ${s.dimensions.toList()}" }).joinToString("\n", postfix = "\n")
