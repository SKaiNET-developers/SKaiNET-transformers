package sk.ainet.apps.klfm2

import kotlinx.serialization.json.*
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.InputStream
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Base64
import java.util.concurrent.TimeUnit
import java.util.concurrent.Executors
import javax.tools.ToolProvider
import kotlin.test.*

class Lfm2IreeGenerationRuntimeTest {
    private val moduleHash = "a".repeat(64)
    private val parameterHash = "b".repeat(64)
    private val device = "vulkan://test-device"
    private fun ready(overrides: Map<String, JsonElement> = emptyMap()) = JsonObject(buildJsonObject {
        put("protocol", "skainet.lfm2.iree.v1"); put("event", "ready"); put("vocabulary", 4); put("capacity", 6); put("position", 0)
        put("device_uri", device); put("runtime_version", "3.11.0"); put("state_location", "device"); put("cpu_model_fallback", false)
        put("module_sha256", moduleHash); put("parameters_sha256", parameterHash)
    } + overrides)
    private fun response(id: Int, position: Int, steps: Int? = null, values: FloatArray = floatArrayOf(1f, 2f, 4f, 3f)) = buildJsonObject {
        put("id", id); put("ok", true); put("position", position)
        steps?.let {
            put("completed_device_steps", it)
            val bytes = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN).apply { values.forEach { putFloat(it) } }.array()
            put("logits_f32le", Base64.getEncoder().encodeToString(bytes))
        }
    }

    private fun <T> withBundle(block: (File, File) -> T): T {
        val directory = kotlin.io.path.createTempDirectory("lfm2-iree-test").toFile()
        val executable = File(System.getProperty("java.home"), "bin/java")
        try {
            File(directory, "runtime-manifest.json").writeText(buildJsonObject {
                put("schema_version", 1); put("protocol", "skainet.lfm2.iree.v1"); put("backend", "vulkan-spirv"); put("dtype", "float32")
                put("dimensions", buildJsonObject { put("vocabulary", 4) }); put("capacity", 6)
                put("module", buildJsonObject { put("sha256", moduleHash) })
                put("parameters", buildJsonObject { put("sha256", parameterHash) })
            }.toString())
            return block(directory, executable)
        } finally { directory.deleteRecursively() }
    }

    @Test fun publishedRuntimeChecksInputsPositionsStepsAndReturnsLastLogits() = withBundle { directory, executable ->
        val child = RecordedProcess(listOf(ready(), response(1, 2, 2), response(2, 0), response(3, 1, 3), response(4, 1)))
        val owner = Lfm2IreeGenerationRuntime(Lfm2IreeOptions(directory, executable, device), { child }, File(directory, "fake.py"))
        assertEquals(4, owner.vocabularySize); assertEquals(6, owner.capacity)
        for (bad in listOf(intArrayOf(), intArrayOf(-1), intArrayOf(4), IntArray(7))) {
            assertFailsWith<IllegalArgumentException> { owner.forwardBatched(bad) }
        }
        assertEquals("", child.requests.toString(Charsets.UTF_8))
        assertContentEquals(floatArrayOf(1f, 2f, 4f, 3f), owner.forwardBatched(intArrayOf(0, 1)).data.copyToFloatArray())
        assertEquals(2, owner.position); assertEquals(2L, owner.completedDeviceSteps)
        owner.reset(); assertEquals(0, owner.position)
        owner.forward(3); assertEquals(1, owner.position); assertEquals(3L, owner.completedDeviceSteps)
        owner.close(); owner.close()
        assertFalse(child.isAlive)
        assertFailsWith<IllegalStateException> { owner.forward(0) }
        assertFailsWith<IllegalStateException> { owner.reset() }
        val requests = child.requests.toString(Charsets.UTF_8).trim().lines().map { Json.parseToJsonElement(it).jsonObject }
        assertEquals(listOf("forward", "reset", "forward", "close"), requests.map { it.getValue("op").jsonPrimitive.content })
        assertEquals(listOf(0, 1), requests.first().getValue("tokens").jsonArray.map { it.jsonPrimitive.int })
    }

    @Test fun unexpectedReadyMetadataAndMalformedForwardResponsesTerminateOwnedWorker() = withBundle { directory, executable ->
        for (changed in listOf(mapOf("module_sha256" to JsonPrimitive("c".repeat(64))), mapOf("cpu_model_fallback" to JsonPrimitive(true)),
            mapOf("device_uri" to JsonPrimitive("local-task")), mapOf("runtime_version" to JsonPrimitive("other")))) {
            val child = RecordedProcess(listOf(ready(changed)))
            assertFails { Lfm2IreeGenerationRuntime(Lfm2IreeOptions(directory, executable, device), { child }, File(directory, "fake.py")) }
            assertFalse(child.isAlive)
        }
        for (bad in listOf(response(99, 1, 1), response(1, 2, 1), response(1, 1, 2), response(1, 1, 1, floatArrayOf(1f)),
            response(1, 1, 1, floatArrayOf(1f, Float.NaN, 0f, 0f)),
            buildJsonObject { put("id", 1); put("ok", false); put("fatal", true); put("error", "dispatch failed") })) {
            val child = RecordedProcess(listOf(ready(), bad))
            val owner = Lfm2IreeGenerationRuntime(Lfm2IreeOptions(directory, executable, device), { child }, File(directory, "fake.py"))
            assertFails { owner.forward(0) }
            assertFalse(child.isAlive)
            assertFailsWith<IllegalStateException> { owner.forward(0) }
            owner.close()
        }
    }

    @Test fun textSessionsOwnExclusiveFreshStateAndReleaseItAfterCallbackFailure() = withBundle { directory, executable ->
        val child = RecordedProcess(listOf(ready(), response(1, 0), response(2, 1, 1), response(3, 0), response(4, 0), response(5, 0), response(6, 0)))
        val owner = Lfm2IreeGenerationRuntime(Lfm2IreeOptions(directory, executable, device), { child }, File(directory, "fake.py"))
        val codec = object : GenerationTextCodec {
            override val stopTokenIds = setOf(3)
            override fun encodeMessages(messages: List<Lfm2ChatMessage>, preserveThinking: Boolean) = intArrayOf(0)
            override fun decode(tokenIds: IntArray) = "token"
        }
        val generator = Lfm2TextGenerator(4, 6, codec) { owner.openTextSession() }
        val failure = assertFailsWith<IllegalArgumentException> {
            generator.generate(listOf(Lfm2ChatMessage("user", "x")), Lfm2GenerationOptions(1)) {
                assertFailsWith<IllegalStateException> { owner.openTextSession() }
                assertFailsWith<IllegalStateException> { owner.forward(0) }
                throw IllegalArgumentException("callback failed")
            }
        }
        assertEquals("callback failed", failure.message)
        assertEquals(0, owner.position)
        val fresh = owner.openTextSession(); assertEquals(0, fresh.position)
        fresh.close(); fresh.close()
        assertFailsWith<IllegalStateException> { fresh.forward(0) }
        owner.close()
        val operations = child.requests.toString(Charsets.UTF_8).trim().lines().map { Json.parseToJsonElement(it).jsonObject.getValue("op").jsonPrimitive.content }
        assertEquals(listOf("reset", "forward", "reset", "reset", "reset", "close"), operations)
    }

    @Test fun forwardTimeoutKillsAndReapsTheExactOwnedChildWithinBoundedTime() = withBundle { directory, executable ->
        val source = File(directory, "SleepingWorker.java")
        source.writeText("""
            public class SleepingWorker {
                public static void main(String[] args) throws Exception {
                    System.out.println(${JsonPrimitive(ready().toString())});
                    System.out.flush();
                    new java.io.BufferedReader(new java.io.InputStreamReader(System.in)).readLine();
                    Thread.sleep(30000);
                }
            }
        """.trimIndent())
        val compiler = assertNotNull(ToolProvider.getSystemJavaCompiler(), "The JVM build requires a JDK")
        val diagnostics = ByteArrayOutputStream()
        assertEquals(0, compiler.run(null, diagnostics, diagnostics, source.absolutePath), diagnostics.toString())
        lateinit var process: Process
        val owner = Lfm2IreeGenerationRuntime(Lfm2IreeOptions(directory, executable, device, timeoutMillis = 1000, closeTimeoutMillis = 1000), {
            ProcessBuilder(executable.absolutePath, "-cp", directory.absolutePath, "SleepingWorker").start().also { process = it }
        }, source)
        val start = System.nanoTime()
        val failure = assertFailsWith<IllegalStateException> { owner.forward(1) }
        assertTrue(failure.message!!.contains("timed out"))
        assertTrue(TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start) < 5000)
        assertFalse(process.isAlive)
        assertTrue(process.waitFor(1, TimeUnit.SECONDS))
        owner.close()
    }

    @Test fun crossThreadCloseInterruptsPendingForwardWithoutWaitingForItsLongTimeout() = withBundle { directory, executable ->
        val source = File(directory, "CloseWorker.java")
        source.writeText("""
            public class CloseWorker {
                public static void main(String[] args) throws Exception {
                    System.out.println(${JsonPrimitive(ready().toString())});
                    System.out.flush();
                    new java.io.BufferedReader(new java.io.InputStreamReader(System.in)).readLine();
                    java.nio.file.Files.writeString(java.nio.file.Path.of(args[0]), "received");
                    Thread.sleep(30000);
                }
            }
        """.trimIndent())
        val compiler = assertNotNull(ToolProvider.getSystemJavaCompiler())
        val diagnostics = ByteArrayOutputStream()
        assertEquals(0, compiler.run(null, diagnostics, diagnostics, source.absolutePath), diagnostics.toString())
        val received = File(directory, "received")
        lateinit var process: Process
        val owner = Lfm2IreeGenerationRuntime(Lfm2IreeOptions(directory, executable, device, timeoutMillis = 30_000, closeTimeoutMillis = 1000), {
            ProcessBuilder(executable.absolutePath, "-cp", directory.absolutePath, "CloseWorker", received.absolutePath).start().also { process = it }
        }, source)
        val executor = Executors.newSingleThreadExecutor { Thread(it, "iree-close-test").apply { isDaemon = true } }
        try {
            val request = executor.submit { assertFailsWith<IllegalStateException> { owner.forward(1) } }
            val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5)
            while (!received.exists() && System.nanoTime() < deadline) Thread.sleep(5)
            assertTrue(received.exists(), "Child must be inside its forward before close")
            val start = System.nanoTime()
            owner.close()
            assertTrue(TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start) < 3000)
            request.get(3, TimeUnit.SECONDS)
            assertFalse(process.isAlive)
        } finally { owner.close(); process.destroyForcibly(); executor.shutdownNow() }
    }

    @Test fun invalidConfigurationNeverStartsAWorker() = withBundle { directory, executable ->
        assertFailsWith<IllegalArgumentException> { Lfm2IreeOptions(directory, executable, "local-task") }
        assertFailsWith<IllegalArgumentException> { Lfm2IreeOptions(directory, File(directory, "missing-python"), device) }
        assertFailsWith<IllegalArgumentException> { Lfm2IreeOptions(directory, executable, device, timeoutMillis = 0) }
        var launched = false
        File(directory, "runtime-manifest.json").writeText("{}")
        assertFails { Lfm2IreeGenerationRuntime(Lfm2IreeOptions(directory, executable, device), { launched = true; error("must not launch") }, File(directory, "fake.py")) }
        assertFalse(launched)
    }

    private class RecordedProcess(responses: List<JsonObject>) : Process() {
        val requests = ByteArrayOutputStream()
        private val output = ByteArrayInputStream((responses.joinToString("\n") + "\n").toByteArray(Charsets.UTF_8))
        private var alive = true
        override fun getOutputStream(): OutputStream = requests
        override fun getInputStream(): InputStream = output
        override fun getErrorStream(): InputStream = ByteArrayInputStream(byteArrayOf())
        override fun waitFor(): Int { alive = false; return 0 }
        override fun waitFor(timeout: Long, unit: TimeUnit): Boolean = !alive
        override fun exitValue(): Int { check(!alive); return 0 }
        override fun destroy() { alive = false }
        override fun destroyForcibly(): Process { alive = false; return this }
        override fun isAlive(): Boolean = alive
    }
}
