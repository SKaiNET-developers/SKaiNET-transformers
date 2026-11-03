package sk.ainet.apps.klfm2

import kotlinx.serialization.json.*
import java.io.File
import java.nio.file.Files
import java.nio.file.LinkOption.NOFOLLOW_LINKS
import java.nio.file.StandardOpenOption.CREATE_NEW
import java.nio.file.StandardOpenOption.WRITE
import java.security.MessageDigest
import kotlin.system.exitProcess

internal data class IreeChatRequest(val id: String, val messages: List<Lfm2ChatMessage>, val maxNewTokens: Int, val decoderPolicy: Lfm2DecodingPolicy)

internal fun parseIreeChatRequests(text: String): List<IreeChatRequest> {
    val root = parseIreeObject(text.toByteArray(Charsets.UTF_8))
    require(root.keys == setOf("schema_version", "requests") && root.getValue("schema_version").jsonPrimitive.int == 1)
    val requests = root.getValue("requests").jsonArray
    require(requests.size in 1..128) { "Supply between one and 128 requests" }
    val parsed = requests.map { value ->
        val request = value.jsonObject
        require(request.keys == setOf("id", "messages", "maxNewTokens", "decoderPolicy")) { "Unexpected or missing request fields" }
        require(request.getValue("id").jsonPrimitive.isString && request.getValue("decoderPolicy").jsonPrimitive.isString)
        require(!request.getValue("maxNewTokens").jsonPrimitive.isString)
        val id = request.getValue("id").jsonPrimitive.content
        require(id.isNotBlank() && id.length <= 256)
        val messages = request.getValue("messages").jsonArray.map { message ->
            val o = message.jsonObject
            require(o.keys.containsAll(listOf("role", "content")) && o.keys.all { it in setOf("role", "content", "thinking") })
            require(o.getValue("role").jsonPrimitive.isString && o.getValue("content").jsonPrimitive.isString)
            val thinking = o["thinking"]?.let { require(it.jsonPrimitive.isString); it.jsonPrimitive.content }
            Lfm2ChatMessage(o.getValue("role").jsonPrimitive.content, o.getValue("content").jsonPrimitive.content, thinking)
        }
        require(messages.isNotEmpty())
        val maximum = request.getValue("maxNewTokens").jsonPrimitive.int
        require(maximum > 0)
        val policy = Lfm2DecodingPolicy.valueOf(request.getValue("decoderPolicy").jsonPrimitive.content)
        // Validate supported roles/thinking independently of tokenizer/model startup.
        renderLfm2TextChat(messages, "<|startoftext|>")
        IreeChatRequest(id, messages, maximum, policy)
    }
    require(parsed.map { it.id }.distinct().size == parsed.size) { "Request IDs must be unique" }
    return parsed
}

/** Portable command-line example. Answers are computed by the loaded compiled model, never read from fixtures. */
object Lfm2IreeChatCli {
    @JvmStatic fun main(args: Array<String>) {
        if (args.contentEquals(arrayOf("--help"))) {
            println("LFM2 IREE chat: --bundle DIR --python EXECUTABLE --device vulkan://UUID --model-metadata DIR --requests FILE --output NEW.json [--packages DIR] [--timeout-ms N]")
            return
        }
        val options = parseOptions(args)
        fun required(name: String) = options[name] ?: error("Missing --$name")
        val output = File(required("output"))
        require(!output.exists()) { "Output must be a new file" }
        val bundle = File(required("bundle"))
        val metadata = File(required("model-metadata"))
        val requestFile = File(required("requests"))
        val python = File(required("python"))
        val packages = options["packages"]?.let(::File)
        val configuration = Lfm2IreeOptions(bundle, python, required("device"), packages,
            timeoutMillis = options["timeout-ms"]?.toLong() ?: 600_000)
        val requestBytes = requestFile.readBytes()
        val requests = parseIreeChatRequests(parseIreeObject(requestBytes).toString())
        val before = inventory(bundle, metadata, requestFile, python, packages)
        val requestHash = MessageDigest.getInstance("SHA-256").digest(requestBytes).joinToString("") { "%02x".format(it) }
        require(before.getValue("request_file").jsonObject.getValue("sha256").jsonPrimitive.content == requestHash) { "Request file changed while being read" }
        val samples = mutableListOf<JsonObject>()
        var worker: JsonObject? = null
        var sourcePins: JsonObject? = null
        var failure: Throwable? = null
        val started = System.nanoTime()
        var openNanos: Long? = null
        try {
            val frontend = loadLfm2TextFrontend(metadata)
            Lfm2IreeGenerationRuntime(configuration).use { runtime ->
                openNanos = System.nanoTime() - started
                worker = runtime.workerInfo
                sourcePins = runtime.implementationSourcePins
                for (request in requests) {
                    val events = mutableListOf<JsonObject>()
                    val start = System.nanoTime()
                    try {
                        val result = runtime.textGenerator(frontend, request.decoderPolicy).generate(request.messages,
                            Lfm2GenerationOptions(request.maxNewTokens), onToken = { event ->
                                events.add(buildJsonObject {
                                    put("index", event.index); put("token_id", event.tokenId); put("text", event.text)
                                    put("is_stop_token", event.isStopToken); put("elapsed_ns", System.nanoTime() - start)
                                })
                            })
                        samples.add(buildJsonObject {
                            put("id", request.id); put("status", "complete"); put("decoder_policy", request.decoderPolicy.name)
                            put("prompt_ids", JsonArray(result.promptTokenIds.map(::JsonPrimitive)))
                            put("generated_ids", JsonArray(result.generatedTokenIds.map(::JsonPrimitive)))
                            put("text", result.text); put("stop_reason", result.stopReason); put("state_input_tokens", result.stateInputTokens)
                            put("events", JsonArray(events)); put("elapsed_ns", System.nanoTime() - start)
                            put("completed_device_steps", runtime.completedDeviceSteps)
                        })
                    } catch (error: Throwable) {
                        samples.add(buildJsonObject { put("id", request.id); put("status", "failed"); put("error", error.message ?: error.javaClass.name); put("events", JsonArray(events)) })
                        throw error
                    }
                }
            }
        } catch (error: Throwable) { failure = error }
        var after: JsonObject? = null
        try {
            after = inventory(bundle, metadata, requestFile, python, packages)
            check(before == after) { "Execution artifacts changed during generation" }
        } catch (error: Throwable) { if (failure == null) failure = error else failure.addSuppressed(error) }
        val report = buildJsonObject {
            put("schema_version", 1); put("status", if (failure == null) "complete" else "failed")
            put("backend", "skainet_iree_vulkan"); put("cpu_model_fallback", false)
            put("requests", JsonArray(samples)); put("artifacts_before", before)
            after?.let { put("artifacts_after", it) }; worker?.let { put("worker", it) }; sourcePins?.let { put("implementation_sources", it) }
            openNanos?.let { put("open_and_frontend_ns", it) }; put("total_ns", System.nanoTime() - started)
            put("host", buildJsonObject { put("os", System.getProperty("os.name")); put("arch", System.getProperty("os.arch")); put("java_version", System.getProperty("java.version")) })
            failure?.let { put("error", it.message ?: it.javaClass.name) }
        }
        Files.writeString(output.toPath(), Json { prettyPrint = true }.encodeToString(JsonObject.serializer(), report) + "\n", CREATE_NEW, WRITE)
        println(buildJsonObject { put("status", report.getValue("status")); put("requests", samples.size) })
        if (failure != null) exitProcess(1)
    }

    private fun parseOptions(args: Array<String>): Map<String, String> {
        require(args.size % 2 == 0) { "Expected --name value pairs" }
        val allowed = setOf("bundle", "python", "device", "model-metadata", "requests", "output", "packages", "timeout-ms")
        val result = linkedMapOf<String, String>()
        for ((key, value) in args.toList().chunked(2)) {
            require(key.startsWith("--") && key.removePrefix("--") in allowed && result.put(key.removePrefix("--"), value) == null) { "Unknown or repeated option: $key" }
        }
        return result
    }

    private fun inventory(bundle: File, metadata: File, requests: File, python: File, packages: File?): JsonObject = buildJsonObject {
        put("bundle", inventoryIreeDirectory(bundle))
        put("model_metadata", JsonArray(listOf("config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja", "LICENSE")
            .map { record(File(metadata, it), it) }))
        put("request_file", record(requests, requests.name))
        put("python_executable", record(python.canonicalFile, python.canonicalFile.name))
        packages?.let { put("python_packages", inventoryIreeDirectory(it)) }
        val jars = System.getProperty("java.class.path").split(File.pathSeparator).map(::File)
        require(jars.isNotEmpty() && jars.all { it.isFile && it.extension == "jar" }) { "Run the installed distribution so the complete runtime JAR closure can be pinned" }
        require(jars.map { it.name }.distinct().size == jars.size) { "Ambiguous runtime JAR names" }
        put("runtime_jars", JsonArray(jars.sortedBy { it.name }.map { record(it, "lib/" + it.name) }))
        val javaHome = File(System.getProperty("java.home"))
        val jvm = listOf("lib/server/libjvm.so", "lib/server/libjvm.dylib", "bin/server/jvm.dll").single { File(javaHome, it).isFile }
        put("java_runtime", JsonArray(listOf("bin/java", "release", "lib/modules", jvm).map { record(File(javaHome, it).canonicalFile, it) }))
        put("java_runtime_coverage", "executable_release_modules_libjvm")
    }

    internal fun record(file: File, name: String): JsonObject {
        require(file.isFile) { "Missing artifact: $name" }
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { input ->
            val buffer = ByteArray(1024 * 1024)
            while (true) { val count = input.read(buffer); if (count < 0) break; digest.update(buffer, 0, count) }
        }
        return buildJsonObject { put("file", name); put("size_bytes", file.length()); put("sha256", digest.digest().joinToString("") { "%02x".format(it) }) }
    }
}

internal fun inventoryIreeDirectory(directory: File): JsonArray {
    require(Files.isDirectory(directory.toPath(), NOFOLLOW_LINKS)) { "Inventory root must be a real directory" }
    val root = directory.toPath()
    val files = Files.walk(root).use { paths -> paths.filter { it != root }.toList() }
    require(files.none(Files::isSymbolicLink)) { "Artifact inventories reject symlinks" }
    return JsonArray(files.filter { Files.isRegularFile(it, NOFOLLOW_LINKS) }.sortedBy { root.relativize(it).toString() }.map {
        Lfm2IreeChatCli.record(it.toFile(), root.relativize(it).toString().replace(File.separatorChar, '/'))
    })
}
