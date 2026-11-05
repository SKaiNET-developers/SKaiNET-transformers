package sk.ainet.apps.klfm2

import kotlinx.serialization.json.*
import sk.ainet.apps.llm.InferenceRuntime
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import java.io.BufferedInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.util.Base64
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/** Explicit optional IREE installation and bundle; there is no implicit CPU backend or package install. */
data class Lfm2IreeOptions(
    val bundleDirectory: File,
    val pythonExecutable: File,
    val deviceUri: String,
    val packageDirectory: File? = null,
    val timeoutMillis: Long = 600_000,
    val closeTimeoutMillis: Long = 5_000,
) {
    init {
        require(bundleDirectory.isDirectory) { "IREE bundle directory does not exist" }
        require(pythonExecutable.isFile && pythonExecutable.canExecute()) { "Supply an executable Python path" }
        require(packageDirectory == null || packageDirectory.isDirectory) { "Private Python package directory does not exist" }
        require(deviceUri.startsWith("vulkan://") && deviceUri.removePrefix("vulkan://").isNotBlank()) { "Select an explicit Vulkan device URI" }
        require(timeoutMillis in 1..86_400_000 && closeTimeoutMillis in 1..60_000)
    }
}

/**
 * Persistent optional subprocess executing the compiled model and hybrid state on one Vulkan device.
 * Only token controls and final logits cross the process/device boundary. The CPU context below
 * merely wraps received logits for the existing InferenceRuntime interface; it executes no model.
 * Calls serialize on this owner. Cancellation from the text loop is observed at forward boundaries.
 */
class Lfm2IreeGenerationRuntime internal constructor(
    private val options: Lfm2IreeOptions,
    private val launch: (List<String>) -> Process,
    private val workerSource: File?,
) : InferenceRuntime<FP32>, AutoCloseable {
    constructor(options: Lfm2IreeOptions) : this(options, { ProcessBuilder(it).start() }, null)

    private val monitor = ReentrantLock()
    private val tensorContext = DirectCpuExecutionContext.create()
    private val io = Executors.newSingleThreadExecutor { task -> Thread(task, "lfm2-iree-io").apply { isDaemon = true } }
    private val stderrLock = Any()
    private var stderrTail = ""
    private var temporaryDirectory: File? = null
    private var child: Process? = null
    private lateinit var stdout: BufferedInputStream
    private var requestId = 0L
    private var leased = false
    private val terminated = AtomicBoolean(false)
    @Volatile private var closed = false
    @Volatile var position: Int = 0
        private set
    @Volatile var completedDeviceSteps: Long = 0
        private set
    val vocabularySize: Int
    val capacity: Int
    val workerInfo: JsonObject
    /** Build-time portable source hashes, including this adapter and its exact Python worker. */
    val implementationSourcePins: JsonObject by lazy {
        val stream = javaClass.getResourceAsStream("/klfm2-source-pins.json")
            ?: error("Runtime artifact lacks its generated source provenance")
        stream.bufferedReader(Charsets.UTF_8).use { Json.parseToJsonElement(it.readText()).jsonObject }
    }

    init {
        try {
            val manifest = parseIreeObject(File(options.bundleDirectory, "runtime-manifest.json").readBytes())
            require(manifest.getValue("protocol").jsonPrimitive.content == PROTOCOL && manifest.getValue("schema_version").jsonPrimitive.int == 1)
            require(manifest.getValue("backend").jsonPrimitive.content == "vulkan-spirv" && manifest.getValue("dtype").jsonPrimitive.content == "float32")
            vocabularySize = manifest.getValue("dimensions").jsonObject.getValue("vocabulary").jsonPrimitive.int
            capacity = manifest.getValue("capacity").jsonPrimitive.int
            require(vocabularySize in 1..MAX_VOCABULARY && capacity > 0)
            val source = workerSource ?: extractWorker()
            val command = mutableListOf(options.pythonExecutable.absolutePath, "-I", "-B", source.absolutePath,
                "--bundle", options.bundleDirectory.absolutePath, "--device", options.deviceUri)
            options.packageDirectory?.let { command.addAll(listOf("--python-package-directory", it.absolutePath)) }
            val process = launch(command)
            child = process
            stdout = BufferedInputStream(process.inputStream)
            Thread({
                try {
                    process.errorStream.bufferedReader(Charsets.UTF_8).use { reader ->
                        val buffer = CharArray(2048)
                        while (true) {
                            val count = reader.read(buffer); if (count < 0) break
                            synchronized(stderrLock) { stderrTail = (stderrTail + String(buffer, 0, count)).takeLast(16 * 1024) }
                        }
                    }
                } catch (_: Exception) { /* Process termination can close the pipe. */ }
            }, "lfm2-iree-stderr").apply { isDaemon = true; start() }
            workerInfo = exchange(null)
            require(workerInfo.getValue("protocol").jsonPrimitive.content == PROTOCOL && workerInfo.getValue("event").jsonPrimitive.content == "ready")
            require(workerInfo.getValue("runtime_version").jsonPrimitive.content == "3.11.0")
            require(workerInfo.getValue("state_location").jsonPrimitive.content == "device" && !workerInfo.getValue("cpu_model_fallback").jsonPrimitive.boolean)
            require(workerInfo.getValue("device_uri").jsonPrimitive.content == options.deviceUri)
            require(workerInfo.getValue("position").jsonPrimitive.int == 0 && workerInfo.getValue("capacity").jsonPrimitive.int == capacity &&
                workerInfo.getValue("vocabulary").jsonPrimitive.int == vocabularySize)
            for (kind in listOf("module", "parameters")) {
                val expected = manifest.getValue(kind).jsonObject.getValue("sha256").jsonPrimitive.content
                require(expected.matches(Regex("[0-9a-f]{64}")) && workerInfo.getValue("${kind}_sha256").jsonPrimitive.content == expected) {
                    "IREE worker loaded a different $kind artifact"
                }
            }
        } catch (failure: Throwable) {
            terminate()
            throw failure
        }
    }

    override fun forward(tokenId: Int): Tensor<FP32, Float> = forwardBatched(intArrayOf(tokenId))

    override fun forwardBatched(tokenIds: IntArray): Tensor<FP32, Float> = monitor.withLock {
        ensureOpen(); check(!leased) { "The worker is leased by text generation" }
        tensor(forwardValues(tokenIds))
    }

    override fun reset() = monitor.withLock {
        ensureOpen(); check(!leased) { "The worker is leased by text generation" }
        resetWorker()
    }

    /** Each generation gets an exclusive reset state; closing its session releases the lease, not this owner. */
    fun textGenerator(frontend: Lfm2TextFrontend,
        decodingPolicy: Lfm2DecodingPolicy = Lfm2DecodingPolicy.CHECKPOINT,
    ): Lfm2TextGenerator {
        require(frontend.vocabularySize == vocabularySize) { "Tokenizer and GPU vocabulary sizes differ" }
        return Lfm2TextGenerator(vocabularySize, capacity, object : GenerationTextCodec {
            override val stopTokenIds get() = frontend.stopTokenIds
            override fun encodeMessages(messages: List<Lfm2ChatMessage>, preserveThinking: Boolean) =
                frontend.encodeMessages(messages, addGenerationPrompt = true, preserveThinking = preserveThinking)
            override fun decode(tokenIds: IntArray) = frontend.decode(tokenIds, skipSpecialTokens = true, policy = decodingPolicy)
        }, ::openTextSession)
    }

    internal fun openTextSession(): GenerationTokenSession = monitor.withLock {
        ensureOpen(); check(!leased) { "Another text generation owns the worker" }
        resetWorker(); leased = true
        object : GenerationTokenSession {
            private var sessionClosed = false
            override val position: Int get() = monitor.withLock { this@Lfm2IreeGenerationRuntime.position }
            override fun forwardBatched(tokenIds: IntArray): FloatArray = monitor.withLock {
                check(!sessionClosed) { "Text session is closed" }; ensureOpen(); forwardValues(tokenIds)
            }
            override fun forward(tokenId: Int): FloatArray = forwardBatched(intArrayOf(tokenId))
            override fun close() = monitor.withLock {
                if (!sessionClosed) {
                    sessionClosed = true
                    try { if (!closed) resetWorker() } finally { leased = false }
                }
            }
        }
    }

    private fun forwardValues(tokenIds: IntArray): FloatArray {
        val tokens = tokenIds.copyOf()
        require(tokens.isNotEmpty() && tokens.all { it in 0 until vocabularySize }) { "Token IDs must be nonempty and inside the vocabulary" }
        require(position.toLong() + tokens.size <= capacity) { "The fixed-capacity GPU context is exhausted" }
        val expectedPosition = position + tokens.size
        val expectedSteps = Math.addExact(completedDeviceSteps, tokens.size.toLong())
        val response = command("forward", buildJsonObject { put("tokens", JsonArray(tokens.map(::JsonPrimitive))) })
        return validateResponse {
            check(response.getValue("position").jsonPrimitive.int == expectedPosition) { "Worker position disagrees with accepted token count" }
            check(response.getValue("completed_device_steps").jsonPrimitive.long == expectedSteps) { "Worker device-step count disagrees with accepted token count" }
            val encoded = response.getValue("logits_f32le").jsonPrimitive.content
            val bytes = Base64.getDecoder().decode(encoded)
            check(bytes.size == vocabularySize * 4) { "Worker returned an incorrect logit byte count" }
            val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val values = FloatArray(vocabularySize) { buffer.float }
            check(values.all { it.isFinite() }) { "Worker returned non-finite logits" }
            position = expectedPosition; completedDeviceSteps = expectedSteps
            values
        }
    }

    private fun resetWorker() {
        val response = command("reset")
        validateResponse { check(response.getValue("position").jsonPrimitive.int == 0); position = 0 }
    }

    private fun command(operation: String, fields: JsonObject = buildJsonObject {}, timeoutMillis: Long = options.timeoutMillis): JsonObject {
        val id = Math.addExact(requestId, 1).also { requestId = it }
        val request = buildJsonObject { put("id", id); put("op", operation); fields.forEach { (key, value) -> put(key, value) } }
        val response = exchange(request, timeoutMillis)
        return validateResponse {
            check(response.getValue("id").jsonPrimitive.long == id) { "IREE worker response ID mismatch" }
            if (!response.getValue("ok").jsonPrimitive.boolean) {
                val detail = response["error"]?.jsonPrimitive?.content ?: "IREE worker failed"
                // Even a recoverable protocol error indicates disagreement with local validation.
                throw IllegalStateException(detail)
            }
            response
        }
    }

    private fun exchange(request: JsonObject?, timeoutMillis: Long = options.timeoutMillis): JsonObject {
        val future = io.submit<JsonObject> {
            request?.let {
                val bytes = (it.toString() + "\n").toByteArray(Charsets.UTF_8)
                require(bytes.size <= 1024 * 1024) { "IREE request exceeds the protocol limit" }
                child!!.outputStream.write(bytes); child!!.outputStream.flush()
            }
            val bytes = ByteArrayOutputStream()
            while (true) {
                val next = stdout.read()
                check(next >= 0) { "IREE worker exited before a complete response" }
                if (next == 10) break
                check(bytes.size() < MAX_RESPONSE_BYTES) { "IREE worker response exceeds the protocol limit" }
                bytes.write(next)
            }
            parseIreeObject(bytes.toByteArray())
        }
        try {
            return future.get(timeoutMillis, TimeUnit.MILLISECONDS)
        } catch (failure: Throwable) {
            future.cancel(true)
            if (failure is InterruptedException) Thread.currentThread().interrupt()
            terminate()
            val detail = synchronized(stderrLock) { stderrTail.trim() }
            val reason = if (failure is TimeoutException) "IREE worker response timed out" else "IREE worker exchange failed"
            throw IllegalStateException(if (detail.isEmpty()) reason else "$reason: $detail", failure)
        }
    }

    private inline fun <T> validateResponse(block: () -> T): T = try { block() } catch (failure: Throwable) {
        terminate(); throw failure
    }

    private fun tensor(values: FloatArray): Tensor<FP32, Float> = tensorContext.fromFloatArray(Shape(vocabularySize), FP32::class, values)
    private fun ensureOpen() = check(!closed) { "IREE runtime is closed" }

    override fun close() {
        if (closed) return
        // Closing from another thread must not wait for a long GPU forward to release the owner lock.
        if (!monitor.tryLock()) {
            terminate()
            return
        }
        try {
            if (!closed) {
                var acknowledged = false
                try { command("close", timeoutMillis = options.closeTimeoutMillis); acknowledged = true }
                finally { terminate(graceful = acknowledged) }
            }
        } finally { monitor.unlock() }
    }

    private fun terminate(graceful: Boolean = false) {
        closed = true
        if (!terminated.compareAndSet(false, true)) return
        val process = child
        var stillAlive = false
        if (process != null) {
            val deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(options.closeTimeoutMillis)
            // Child ownership is explicit. No shell, process-name search, or global signal is used.
            val descendants = try { process.toHandle().descendants().toList() } catch (_: Exception) { emptyList() }
            try {
                if (graceful && process.isAlive) process.waitFor(maxOf(1, options.closeTimeoutMillis / 2), TimeUnit.MILLISECONDS)
                if (process.isAlive) {
                    descendants.forEach { try { it.destroyForcibly() } catch (_: Exception) {} }
                    process.destroyForcibly()
                    val remaining = maxOf(1, deadline - System.nanoTime())
                    process.waitFor(remaining, TimeUnit.NANOSECONDS)
                }
                stillAlive = process.isAlive
            } catch (_: InterruptedException) {
                process.destroyForcibly()
                Thread.currentThread().interrupt()
            }
        }
        io.shutdownNow()
        temporaryDirectory?.deleteRecursively()
        check(!stillAlive) { "Owned IREE worker did not exit within its cleanup deadline" }
    }

    private fun extractWorker(): File {
        val resource = javaClass.getResourceAsStream("/sk/ainet/apps/klfm2/iree_worker.py")
            ?: error("This runtime artifact does not contain its IREE worker resource")
        val directory = Files.createTempDirectory("lfm2-iree-worker-").toFile().also { temporaryDirectory = it }
        return File(directory, "iree_worker.py").also { file -> resource.use { input -> file.outputStream().use { output -> input.copyTo(output) } } }
    }

    companion object {
        /** Validates the explicit Python/IREE/Vulkan device without loading parameters or executing a model. */
        @JvmStatic fun preflight(options: Lfm2IreeOptions): JsonObject = preflightLfm2Iree(options)

        private const val PROTOCOL = "skainet.lfm2.iree.v1"
        private const val MAX_RESPONSE_BYTES = 4 * 1024 * 1024
        private const val MAX_VOCABULARY = 512 * 1024
    }
}
