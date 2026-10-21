package sk.ainet.transformers.iree.android

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * On-device check of [IreeModuleRunner]: runs every case of a `cases.json` served from the host and compares the
 * output with the case's reference. Nothing model-specific lives here — the cases (vmfb, optional irpa, function,
 * device, raw little-endian float32 input and expected output) come from whoever serves them:
 *
 * ```
 * adb reverse tcp:8766 tcp:8766
 * (cd /path/to/cases && python3 -m http.server 8766) &
 * ./gradlew :llm-runtime:iree-android:connectedAndroidTest
 * ```
 *
 * `cases.json`: `[{"name", "vmfb", "irpa"?, "function", "device", "input", "input_shape", "expected", "rtol", "atol",
 * "runs"?}]`, file names relative to the served directory. Skips (does not fail) when no server is reachable.
 * Instrumentation arg `moduleCasesUrl` overrides the base URL. Per case it logs max |Δ| and p50/p95 latency under
 * tag `IreeModuleDevice`.
 */
@RunWith(AndroidJUnit4::class)
class IreeModuleRunnerDeviceTest {

    private companion object {
        const val TAG = "IreeModuleDevice"
        const val DEFAULT_URL = "http://localhost:8766"
    }

    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val baseUrl = InstrumentationRegistry.getArguments().getString("moduleCasesUrl") ?: DEFAULT_URL
    private val cacheDir = File(instrumentation.targetContext.cacheDir, "iree-module-cases").apply { mkdirs() }

    private fun log(msg: String) {
        Log.i(TAG, msg)
        println(msg)
    }

    private fun fetch(name: String): File? {
        val dest = File(cacheDir, name)
        return try {
            val conn = URL("$baseUrl/$name").openConnection() as HttpURLConnection
            conn.connectTimeout = 3_000
            conn.readTimeout = 60_000
            conn.inputStream.use { input -> dest.outputStream().use { input.copyTo(it) } }
            dest
        } catch (e: Exception) {
            log("cannot fetch $baseUrl/$name: ${e.message}")
            null
        }
    }

    private fun floats(file: File): FloatArray {
        val bytes = file.readBytes()
        return FloatArray(bytes.size / 4).also { ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(it) }
    }

    private fun JSONArray.ints() = IntArray(length()) { getInt(it) }

    @Test
    fun everyServedCaseMatchesItsReference() {
        val index = fetch("cases.json")
        assumeTrue("no case server at $baseUrl (see the class doc)", index != null)
        val cases = JSONArray(index!!.readText())
        assertTrue("cases.json is empty", cases.length() > 0)
        val failures = mutableListOf<String>()
        for (i in 0 until cases.length()) runCase(cases.getJSONObject(i))?.let { failures += it }
        assertTrue("failed cases:\n" + failures.joinToString("\n"), failures.isEmpty())
    }

    /** Returns a failure line, or null when the case passes. */
    private fun runCase(c: JSONObject): String? {
        val name = c.getString("name")
        val vmfb = fetch(c.getString("vmfb")) ?: return "$name: vmfb not served"
        val irpa = if (c.has("irpa")) fetch(c.getString("irpa")) ?: return "$name: irpa not served" else null
        val input = FloatTensor(floats(fetch(c.getString("input")) ?: return "$name: input not served"), c.getJSONArray("input_shape").ints())
        val expected = floats(fetch(c.getString("expected")) ?: return "$name: expected not served")
        val rtol = c.getDouble("rtol").toFloat()
        val atol = c.getDouble("atol").toFloat()
        val runs = c.optInt("runs", 1).coerceAtLeast(1)

        val createStart = System.nanoTime()
        val runner = try {
            IreeModuleRunner(vmfb.path, irpa?.path, c.getString("function"), c.getString("device"))
        } catch (e: IllegalStateException) {
            return "$name: ${e.message}"
        }
        val createMs = (System.nanoTime() - createStart) / 1e6
        runner.use { r ->
            val times = DoubleArray(runs)
            var out: FloatTensor? = null
            for (k in 0 until runs) {
                val t0 = System.nanoTime()
                out = r.run(input)
                times[k] = (System.nanoTime() - t0) / 1e6
            }
            val actual = out!!.data
            if (actual.size != expected.size) return "$name: output $out has ${actual.size} values, reference ${expected.size}"
            var maxDiff = 0f
            var bad = 0
            for (j in actual.indices) {
                val d = kotlin.math.abs(actual[j] - expected[j])
                if (d > maxDiff) maxDiff = d
                if (!(d <= atol + rtol * kotlin.math.abs(expected[j]))) bad++
            }
            times.sort()
            fun pct(p: Double) = times[((times.size - 1) * p).toInt()]
            log(
                "$name: device=${c.getString("device")} output $out max|Δ|=$maxDiff outside tolerance=$bad " +
                    "create=%.0f ms first=%.1f ms p50=%.1f ms p95=%.1f ms (runs=$runs)".format(createMs, times.first(), pct(0.5), pct(0.95)),
            )
            return if (bad == 0) null else "$name: $bad of ${actual.size} values outside rtol $rtol / atol $atol (max |Δ| $maxDiff)"
        }
    }
}
