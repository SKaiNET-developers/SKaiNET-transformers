package sk.ainet.apps.klfm2

import kotlinx.serialization.json.*
import java.io.*
import java.util.concurrent.TimeUnit
import kotlin.test.*

class Lfm2IreePreflightTest {
    @Test fun probeBindsExplicitDeviceAndNeverPassesTheModelBundleToPython() {
        val directory = kotlin.io.path.createTempDirectory("iree-probe-test").toFile()
        try {
            val options = Lfm2IreeOptions(directory, File(System.getProperty("java.home"), "bin/java"), "vulkan://selected-device")
            val response = buildJsonObject {
                put("protocol", "skainet.lfm2.iree.v1"); put("event", "preflight"); put("runtime_version", "3.11.0")
                put("device_uri", options.deviceUri); put("device_name", "Synthetic device")
                put("ready", true); put("cpu_model_fallback", false); put("scope", "device_initialization_only")
            }
            var command = listOf<String>()
            val result = preflightLfm2Iree(options, { args -> command = args; ProbeProcess(response) }, File(directory, "fake.py"))
            assertEquals(response, result)
            assertTrue("--probe" in command && "-I" in command && "-B" in command)
            assertFalse("--bundle" in command)
            assertEquals(options.deviceUri, command[command.indexOf("--device") + 1])
            assertFailsWith<IllegalStateException> {
                preflightLfm2Iree(options, { ProbeProcess(JsonObject(response + ("cpu_model_fallback" to JsonPrimitive(true)))) }, File(directory, "fake.py"))
            }
        } finally { directory.deleteRecursively() }
    }

    private class ProbeProcess(response: JsonObject) : Process() {
        private var alive = true
        private val output = ByteArrayInputStream((response.toString() + "\n").toByteArray())
        override fun getInputStream(): InputStream = output
        override fun getOutputStream(): OutputStream = ByteArrayOutputStream()
        override fun getErrorStream(): InputStream = ByteArrayInputStream(byteArrayOf())
        override fun waitFor(): Int { alive = false; return 0 }
        override fun waitFor(timeout: Long, unit: TimeUnit): Boolean = !alive
        override fun exitValue(): Int = if (alive) throw IllegalThreadStateException() else 0
        override fun destroy() { alive = false }
        override fun destroyForcibly(): Process { alive = false; return this }
        override fun isAlive() = alive
    }
}
