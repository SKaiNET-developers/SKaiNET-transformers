package sk.ainet.models.lfm2.export

import kotlinx.serialization.json.*
import sk.ainet.lang.tensor.Shape
import sk.ainet.models.lfm2.Lfm2CachedTensorNetwork
import sk.ainet.models.lfm2.expectedModelWeights
import sk.ainet.models.lfm2.Lfm2Config
import sk.ainet.models.lfm2.loadLfm2Model
import sk.ainet.models.lfm2.splitLfm2Embedding
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.StandardOpenOption.CREATE_NEW
import java.nio.file.StandardOpenOption.WRITE
import java.security.MessageDigest

/** One graph serves every prompt token and every cached decode step up to capacity. */
fun traceLfm2CachedStep(network: Lfm2CachedTensorNetwork): Lfm2TracedFunction {
    val trace = traceLfm2Function("lfm2_cached_step", cachedInputShapes(network)) { inputs, ctx ->
        val states = network.stateShapes().keys.associateWith { inputs.getValue(it) }
        val result = network.forward(inputs.getValue("token_one_hot"), inputs.getValue("cos"), inputs.getValue("sin"),
            inputs.getValue("mask"), inputs.getValue("write_mask"), states, ctx)
        linkedMapOf("logits" to result.logits).apply { putAll(result.states) }
    }
    val expectedParameters = expectedModelWeights(network.dimensions).size - 1 + network.embeddingShardRows.size
    check(trace.module.externalParameters.size == expectedParameters) {
        "Cached graph must use each decoder weight once and reuse the embedding row shards for its tied head"
    }
    check(!trace.module.content.contains("custom_call") && !trace.module.content.contains("stablehlo.gather")) {
        "Cached FP32 graph must use ordinary tensor operations without gather or custom calls"
    }
    return trace
}

/**
 * Tensor/control ABI for a persistent runtime. Artifact file names and hashes are bound by
 * the export bundle and the subsequent compilation manifest, not inferred here.
 */
fun lfm2CachedStepContract(network: Lfm2CachedTensorNetwork, trace: Lfm2TracedFunction): JsonObject {
    val inputs = cachedInputShapes(network)
    val outputs = linkedMapOf("logits" to Shape(1, network.dimensions.vocabulary)).apply { putAll(network.stateShapes()) }
    require(trace.module.functionName == "lfm2_cached_step" && trace.inputNames == inputs.keys.toList() &&
        trace.inputShapes == inputs.values.toList() && trace.outputNames == outputs.keys.toList() &&
        trace.outputShapes == outputs.values.toList()) { "Trace differs from the cached network ABI" }
    val d = network.dimensions
    fun ports(shapes: Map<String, Shape>): JsonArray = buildJsonArray {
        for ((name, shape) in shapes) add(buildJsonObject {
            put("name", name); put("shape", JsonArray(shape.dimensions.map(::JsonPrimitive))); put("dtype", "f32")
        })
    }
    return buildJsonObject {
        put("schema_version", 1); put("function_name", "lfm2_cached_step"); put("capacity", network.capacity)
        put("dimensions", buildJsonObject {
            put("vocabulary", d.vocabulary); put("width", d.width); put("head_dimension", d.headDim)
            put("kv_heads", d.kvHeads); put("conv_kernel", d.kernel); put("rope_theta", d.theta)
            put("context_limit", d.contextLimit); put("layer_types", JsonArray(d.layerTypes.map(::JsonPrimitive)))
        })
        put("inputs", ports(inputs)); put("outputs", ports(outputs))
        put("rms_reduction_policy", "BALANCED_FP32"); put("attention_layout_policy", "DIRECT_KV_PER_HEAD")
        put("embedding_strategy", "one_hot_matmul_contiguous_row_shards_tied_head")
        put("embedding_shard_rows", JsonArray(network.embeddingShardRows.map(::JsonPrimitive)))
        put("cache_update", "prefix_cache_append_at_capacity_then_one_hot_write_no_wrap")
        put("host_control", "token_selection_rope_masks_position_and_stop_policy")
        put("compiled_execution", false); put("device_execution", false)
    }
}

private fun cachedInputShapes(network: Lfm2CachedTensorNetwork): LinkedHashMap<String, Shape> = linkedMapOf(
    "token_one_hot" to Shape(1, network.dimensions.vocabulary),
    "cos" to Shape(1, network.dimensions.headDim),
    "sin" to Shape(1, network.dimensions.headDim),
    "mask" to Shape(1, network.capacity + 1),
    "write_mask" to Shape(1, network.capacity, 1),
).apply { putAll(network.stateShapes()) }

/** Local checkpoint export only: no model download, IREE compilation or device execution. */
object Lfm2GenerationExportCli {
    @JvmStatic
    fun main(args: Array<String>) {
        if (args.isEmpty() || args.contentEquals(arrayOf("--help"))) {
            println("LFM2 cached generation export: --config FILE --config-sha256 SHA256 --checkpoint FILE --checkpoint-sha256 SHA256 --capacity N --output NEW_DIRECTORY [--rows-per-shard N]")
            return
        }
        val allowed = setOf("--config", "--config-sha256", "--checkpoint", "--checkpoint-sha256", "--capacity", "--output", "--rows-per-shard")
        require(args.size % 2 == 0) { "Every option requires one value; use --help" }
        val options = linkedMapOf<String, String>()
        for (pair in args.toList().chunked(2)) {
            require(pair[0] in allowed && options.put(pair[0], pair[1]) == null) { "Unknown or repeated option ${pair[0]}" }
        }
        fun required(name: String): String = requireNotNull(options[name]) { "Missing $name" }
        fun pin(name: String): String = required(name).also {
            require(it.matches(Regex("[0-9a-f]{64}"))) { "$name must be lowercase SHA-256" }
        }
        val configFile = File(required("--config"))
        val checkpoint = File(required("--checkpoint"))
        val configPin = pin("--config-sha256")
        val checkpointPin = pin("--checkpoint-sha256")
        val capacity = requireNotNull(required("--capacity").toIntOrNull()) { "Invalid --capacity" }
        val shardRows = options["--rows-per-shard"]?.let {
            requireNotNull(it.toIntOrNull()) { "Invalid --rows-per-shard" }
        } ?: 16384
        val output = File(required("--output"))
        require(!output.exists() && output.absoluteFile.parentFile.isDirectory) { "Use a new output directory beneath an existing parent" }
        require(configFile.isFile && configFile.canRead() && sha256(configFile) == configPin) { "Configuration SHA-256 mismatch or unreadable configuration" }
        val config = Lfm2Config.parse(configFile.readText())
        require(capacity in 1..config.maximumPositions) { "Capacity exceeds the checkpoint position limit" }
        require(shardRows > 0 && shardRows.toLong() * config.hiddenSize * 4L <= 64L * 1024L * 1024L) {
            "Embedding row shard exceeds the 64 MiB evaluation bound"
        }
        val loaded = loadLfm2Model(config, checkpoint, contextLimit = capacity, expectedCheckpointSha256 = checkpointPin)
        val embeddingName = "model.embed_tokens.weight"
        val shards = splitLfm2Embedding(loaded.weights.getValue(embeddingName), loaded.model.ctx, shardRows)
        val (shardHashes, shardRecords) = hashShards(shards, loaded.records.single { it.tensorName == embeddingName }.fp32Sha256)
        val expectedHashes = loaded.records.filterNot { it.tensorName == embeddingName }.map { it.fp32Sha256 } + shardHashes
        val network = Lfm2CachedTensorNetwork(loaded.model.dimensions, loaded.weights, capacity, shards)
        val trace = traceLfm2CachedStep(network)
        val contract = lfm2CachedStepContract(network, trace)
        val contractBytes = (Json { prettyPrint = true }.encodeToString(JsonObject.serializer(), contract) + "\n").toByteArray(Charsets.UTF_8)
        val provenance = buildJsonObject {
            put("kind", "lfm2_cached_generation_export")
            put("config_sha256", configPin); put("checkpoint_sha256", loaded.checkpointSha256)
            put("loaded_tensors", buildJsonArray {
                for (record in loaded.records) add(buildJsonObject {
                    put("name", record.tensorName); put("shape", JsonArray(record.shape.map(::JsonPrimitive)))
                    put("source_dtype", record.sourceDtype); put("fp32_sha256", record.fp32Sha256); put("fp32_bytes", record.fp32Bytes)
                })
            })
            put("embedding_shards", shardRecords)
            put("generation_contract", buildJsonObject {
                put("file", "generation-contract.json")
                put("sha256", MessageDigest.getInstance("SHA-256").digest(contractBytes).generationHex())
                put("size_bytes", contractBytes.size)
            })
        }
        writeLfm2ExportBundle(trace, output, expectedHashes, provenance)
        Files.write(File(output, "generation-contract.json").toPath(), contractBytes, CREATE_NEW, WRITE)
        println("Exported lfm2_cached_step: capacity=$capacity, source_tensors=${loaded.records.size}, external_parameters=${trace.module.externalParameters.size}; compilation and execution remain unqualified")
    }

    /** Independently hash contiguous source slices and verify their concatenation against the loaded source record. */
    private fun hashShards(shards: List<Tensor<FP32, Float>>, expectedEmbeddingHash: String): Pair<List<String>, JsonArray> {
        val complete = MessageDigest.getInstance("SHA-256")
        val hashes = mutableListOf<String>()
        val records = mutableListOf<JsonObject>()
        var firstRow = 0
        for (shard in shards) {
            val digest = MessageDigest.getInstance("SHA-256")
            val buffer = ByteBuffer.allocate(1024 * 1024).order(ByteOrder.LITTLE_ENDIAN)
            fun flush() {
                digest.update(buffer.array(), 0, buffer.position())
                complete.update(buffer.array(), 0, buffer.position())
                buffer.clear()
            }
            val values = shard.data.copyToFloatArray()
            for (value in values) {
                require(value.isFinite()) { "Non-finite embedding shard" }
                if (buffer.remaining() < 4) flush()
                buffer.putFloat(value)
            }
            flush()
            val hash = digest.digest().generationHex()
            hashes.add(hash)
            records.add(buildJsonObject {
                put("first_row", firstRow); put("rows", shard.shape[0]); put("width", shard.shape[1])
                put("fp32_sha256", hash); put("fp32_bytes", values.size * 4L)
            })
            firstRow += shard.shape[0]
        }
        require(complete.digest().generationHex() == expectedEmbeddingHash) { "Embedding shards do not reconstruct the verified source tensor" }
        return hashes to JsonArray(records)
    }
}

private fun ByteArray.generationHex(): String = joinToString("") { "%02x".format(it) }
