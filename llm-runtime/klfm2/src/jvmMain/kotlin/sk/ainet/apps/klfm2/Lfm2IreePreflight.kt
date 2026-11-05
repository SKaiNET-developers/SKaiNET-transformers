package sk.ainet.apps.klfm2

import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonPrimitive
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.file.Files
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/** Device initialization only. No model/archive is loaded and no kernel-execution claim is made. */
internal fun preflightLfm2Iree(
    options: Lfm2IreeOptions,
    launch: (List<String>) -> Process = { ProcessBuilder(it).start() },
    workerSource: File? = null,
): JsonObject {
    val executor = Executors.newFixedThreadPool(2) { task -> Thread(task, "lfm2-iree-preflight").apply { isDaemon = true } }
    var temporary: File? = null
    var process: Process? = null
    val errorLock = Any()
    var stderr = ""
    try {
        val source = workerSource ?: run {
            val resource = Lfm2IreeGenerationRuntime::class.java.getResourceAsStream("/sk/ainet/apps/klfm2/iree_worker.py")
                ?: error("Runtime artifact lacks its IREE worker")
            val directory = Files.createTempDirectory("lfm2-iree-preflight-").toFile().also { temporary = it }
            File(directory, "iree_worker.py").also { file -> resource.use { input -> file.outputStream().use { input.copyTo(it) } } }
        }
        val command = mutableListOf(options.pythonExecutable.absolutePath, "-I", "-B", source.absolutePath, "--probe", "--device", options.deviceUri)
        options.packageDirectory?.let { command.addAll(listOf("--python-package-directory", it.absolutePath)) }
        val child = launch(command).also { process = it }
        executor.submit {
            try {
                child.errorStream.bufferedReader(Charsets.UTF_8).use { reader ->
                    val buffer = CharArray(2048)
                    while (true) {
                        val count = reader.read(buffer); if (count < 0) break
                        synchronized(errorLock) { stderr = (stderr + String(buffer, 0, count)).takeLast(16 * 1024) }
                    }
                }
            } catch (_: Exception) { }
        }
        val result = executor.submit<JsonObject> {
            val bytes = ByteArrayOutputStream()
            child.inputStream.buffered().use { input ->
                val buffer = ByteArray(4096)
                while (true) {
                    val count = input.read(buffer); if (count < 0) break
                    require(bytes.size() + count <= 64 * 1024) { "Oversized IREE preflight response" }
                    bytes.write(buffer, 0, count)
                }
            }
            check(child.waitFor() == 0) { "IREE preflight process failed" }
            parseIreeObject(bytes.toByteArray())
        }.get(options.timeoutMillis, TimeUnit.MILLISECONDS)
        require(result["protocol"]?.jsonPrimitive?.contentOrNull == "skainet.lfm2.iree.v1" &&
            result["event"]?.jsonPrimitive?.contentOrNull == "preflight" &&
            result["runtime_version"]?.jsonPrimitive?.contentOrNull == "3.11.0" &&
            result["device_uri"]?.jsonPrimitive?.contentOrNull == options.deviceUri &&
            result["scope"]?.jsonPrimitive?.contentOrNull == "device_initialization_only" &&
            result["ready"]?.jsonPrimitive?.boolean == true &&
            result["cpu_model_fallback"]?.jsonPrimitive?.boolean == false) { "IREE preflight returned a different device/runtime contract" }
        require(!result["device_name"]?.jsonPrimitive?.contentOrNull.isNullOrBlank())
        return result
    } catch (error: Throwable) {
        if (error is InterruptedException) Thread.currentThread().interrupt()
        val detail = synchronized(errorLock) { stderr.trim() }
        throw IllegalStateException(if (detail.isEmpty()) "IREE device preflight failed" else "IREE device preflight failed: $detail", error)
    } finally {
        try {
            process?.let { child ->
                if (child.isAlive) {
                    try { child.toHandle().descendants().forEach { it.destroyForcibly() } } catch (_: Exception) { }
                    child.destroyForcibly()
                    check(child.waitFor(options.closeTimeoutMillis, TimeUnit.MILLISECONDS)) { "Owned IREE probe did not exit within cleanup deadline" }
                }
            }
        } finally { executor.shutdownNow(); temporary?.deleteRecursively() }
    }
}
