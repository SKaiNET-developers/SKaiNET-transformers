package sk.ainet.models.lfm2.export

import kotlinx.serialization.json.*
import sk.ainet.compile.hlo.StableHloModule
import sk.ainet.lang.tensor.storage.DefaultBufferResolver
import java.io.BufferedOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.StandardOpenOption.CREATE_NEW
import java.nio.file.StandardOpenOption.WRITE
import java.security.MessageDigest

/** Streams FP32 payloads with a 1 MiB buffer and publishes only after source-hash verification.
 * Existing files are never replaced. The archive uses flat shapes; the MLIR retains tensor layouts.
 */
fun writeLfm2ParameterArchive(module: StableHloModule, file: File, expectedWeightHashes: List<String>): JsonObject {
    require(!Files.exists(file.toPath())) { "Parameter archive already exists: ${file.name}" }
    require(expectedWeightHashes.all { it.matches(Regex("[0-9a-f]{64}")) }) { "Expected FP32 payload hashes must be lowercase SHA-256" }
    val resolver = DefaultBufferResolver()
    val sizes = module.externalParameters.map { p -> resolver.resolve(p.source).use { it.sizeInBytes } }
    require(sizes.isNotEmpty() && sizes.all { it > 0 && it % 4 == 0L } && module.externalParameters.all { it.scope == "model" })
    require(module.externalParameters.map { it.key }.distinct().size == sizes.size)
    var offset = 0L
    val header = buildJsonObject {
        for ((i, p) in module.externalParameters.withIndex()) {
            val end = Math.addExact(offset, sizes[i])
            put(p.key, buildJsonObject {
                put("dtype", "F32"); put("shape", buildJsonArray { add(sizes[i] / 4) })
                put("data_offsets", buildJsonArray { add(offset); add(end) })
            })
            offset = end
        }
    }.toString().toByteArray(Charsets.UTF_8)
    val paddedSize = header.size + (8 - header.size % 8) % 8
    val temporary = Files.createTempFile(file.absoluteFile.parentFile.toPath(), ".${file.name}.", ".partial")
    val records = mutableListOf<JsonObject>()
    try {
        BufferedOutputStream(Files.newOutputStream(temporary)).use { output ->
            output.write(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(paddedSize.toLong()).array())
            output.write(header)
            repeat(paddedSize - header.size) { output.write(32) }
            for ((i, parameter) in module.externalParameters.withIndex()) {
                val digest = MessageDigest.getInstance("SHA-256")
                resolver.resolve(parameter.source).use { source ->
                    require(source.sizeInBytes == sizes[i]) { "Parameter source size changed while exporting" }
                    var position = 0L
                    while (position < source.sizeInBytes) {
                        val count = minOf(1024 * 1024L, source.sizeInBytes - position).toInt()
                        val bytes = source.readBytes(position, count)
                        check(bytes.size == count) { "Incomplete parameter read" }
                        output.write(bytes); digest.update(bytes); position += count
                    }
                }
                records.add(buildJsonObject {
                    put("key", parameter.key); put("shape", buildJsonArray { add(sizes[i] / 4) })
                    put("size_bytes", sizes[i]); put("sha256", digest.digest().hex())
                })
            }
        }
        require(records.map { it.getValue("sha256").jsonPrimitive.content }.sorted() == expectedWeightHashes.sorted()) {
            "External weight payloads differ from the verified source tensors"
        }
        // A same-directory hard link publishes atomically and fails if any destination exists.
        // No ATOMIC_MOVE is used: that option may replace an existing file on some providers.
        Files.createLink(file.toPath(), temporary)
    } finally { Files.deleteIfExists(temporary) }
    return buildJsonObject {
        put("file", file.name); put("sha256", sha256(file)); put("size_bytes", file.length()); put("scope", "model")
        put("entries", JsonArray(records)); put("payload_bytes", offset); put("stream_chunk_bytes", 1024 * 1024)
    }
}

/** Create an immutable export bundle in a new directory. A manifest exists only after success.
 * The caller supplies checkpoint/config provenance; this function adds build-time source pins.
 * Export success establishes neither IREE compilation nor device execution.
 */
fun writeLfm2ExportBundle(
    trace: Lfm2TracedFunction,
    outputDirectory: File,
    expectedWeightHashes: List<String>,
    provenance: JsonObject,
): JsonObject {
    Files.createDirectory(outputDirectory.toPath())
    val mlir = File(outputDirectory, "${trace.module.functionName}.mlir")
    Files.writeString(mlir.toPath(), trace.module.content, CREATE_NEW, WRITE)
    val archive = writeLfm2ParameterArchive(trace.module, File(outputDirectory, "model.safetensors"), expectedWeightHashes)
    val pins = lfm2ExportSourcePins()
    val manifest = buildJsonObject {
        put("schema_version", 1); put("status", "exported_not_yet_compiled")
        put("engine_version", "0.62.0"); put("function", trace.module.functionName)
        put("module", buildJsonObject { put("file", mlir.name); put("sha256", sha256(mlir)); put("size_bytes", mlir.length()) })
        put("parameters", archive); put("provenance", provenance); put("implementation_sources", pins)
        put("input_order", JsonArray(trace.inputNames.map(::JsonPrimitive)))
        put("output_order", JsonArray(trace.outputNames.map(::JsonPrimitive)))
        put("input_shapes", JsonArray(trace.inputShapes.map { JsonArray(it.dimensions.map(::JsonPrimitive)) }))
        put("output_shapes", JsonArray(trace.outputShapes.map { JsonArray(it.dimensions.map(::JsonPrimitive)) }))
        put("recorded_operations", trace.operationCount)
        put("compiled_execution", false)
    }
    val text = Json { prettyPrint = true }.encodeToString(JsonObject.serializer(), manifest) + "\n"
    Files.writeString(File(outputDirectory, "export-manifest.json").toPath(), text, CREATE_NEW, WRITE)
    return manifest
}

/** Relative paths and content hashes generated into the published JAR; no checkout is required. */
fun lfm2ExportSourcePins(): JsonObject {
    val stream = Lfm2TracedFunction::class.java.getResourceAsStream("/lfm2-source-pins.json")
        ?: error("This LFM2 artifact lacks its build-generated source provenance")
    return stream.bufferedReader(Charsets.UTF_8).use { Json.parseToJsonElement(it.readText()).jsonObject }
}

fun sha256(file: File): String = file.inputStream().buffered().use { input ->
    val digest = MessageDigest.getInstance("SHA-256")
    val bytes = ByteArray(1024 * 1024)
    while (true) { val count = input.read(bytes); if (count < 0) break; digest.update(bytes, 0, count) }
    digest.digest().hex()
}
internal fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(bytes).hex()
private fun ByteArray.hex(): String = joinToString("") { "%02x".format(it) }
