package sk.ainet.apps.kllama

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.generate
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.exec.tensor.ops.KernelProfile
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.gguf.createRandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.types.FP32
import sk.ainet.models.llama.LlamaNetworkLoader
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import org.junit.Assert.assertTrue
import kotlin.time.measureTime
import kotlin.time.measureTimedValue

/**
 * On-device end-to-end generation spike for SmolLM2-135M-Instruct (Q8_0) —
 * the Android half of the cross-target numbers in transformers#272. Same
 * load/generate path as the commonTest `SmolLm2InferenceSpike`, but running
 * on a real device/emulator with the NEON JNI backend
 * (`skainet-backend-jni-cpu`) that the kllama Android artifact ships since
 * 0.39.0 — this turns the engine-side decode-kernel projection (~24 tok/s,
 * Pixel 8a) into a measured end-to-end generation number.
 *
 * The model is fetched from the host over `adb reverse` so the APK stays
 * small and nothing touches shared storage:
 *
 * ```
 * adb reverse tcp:8765 tcp:8765
 * (cd /path/to/models && python3 -m http.server 8765) &
 * ./gradlew :llm-runtime:kllama:connectedCheck
 * ```
 *
 * Skips (does not fail) when no model server is reachable, so plain CI
 * emulator lanes without the server are unaffected. Knobs via
 * instrumentation args: `smollm2Url`, `smollm2Steps`.
 */
@RunWith(AndroidJUnit4::class)
class AndroidSmolLm2E2eTest {

    private companion object {
        const val TAG = "SmolLm2E2e"
        const val DEFAULT_URL = "http://localhost:8765/SmolLM2-135M-Instruct-Q8_0.gguf"
    }

    private fun log(msg: String) {
        Log.i(TAG, msg)
        println(msg) // also lands in the instrumentation stream
    }

    /** Downloads the GGUF into cacheDir once; returns null if unreachable. */
    private fun fetchModel(url: String): File? {
        val cacheDir = InstrumentationRegistry.getInstrumentation().targetContext.cacheDir
        val dest = File(cacheDir, url.substringAfterLast('/'))
        if (dest.exists() && dest.length() > 0) return dest
        return try {
            val conn = URL(url).openConnection() as HttpURLConnection
            conn.connectTimeout = 3_000
            conn.readTimeout = 60_000
            conn.inputStream.use { input ->
                dest.outputStream().use { output -> input.copyTo(output, 1 shl 20) }
            }
            dest
        } catch (e: Exception) {
            Log.w(TAG, "model download failed: $e")
            dest.delete()
            null
        }
    }

    @Test
    fun generate_e2e_and_measure_tokens_per_second() {
        val args = InstrumentationRegistry.getArguments()
        val url = args.getString("smollm2Url") ?: DEFAULT_URL
        val steps = args.getString("smollm2Steps")?.toIntOrNull() ?: 44

        val modelFile = fetchModel(url)
        assumeTrue(
            "no model server reachable at $url — start one and `adb reverse tcp:8765 tcp:8765`",
            modelFile != null,
        )
        val modelPath = modelFile!!.absolutePath
        log("model: $modelPath (${modelFile.length() / (1 shl 20)} MiB)")

        runBlocking {
            val ctx = DirectCpuExecutionContext()

            // fromRandomAccessSource, NOT fromSource: the Source overload goes
            // through the legacy GGUFReader, which materializes the whole file
            // on the ART heap — instant OOM on a standard app heap, and the
            // exact failure mode the mobile field report described. The
            // streaming path reads metadata only.
            val tokenizer = GGUFTokenizer.fromRandomAccessSource(
                createRandomAccessSource(modelPath)
                    ?: error("createRandomAccessSource returned null on Android")
            )

            // RandomAccessSource overload — streaming load, packed weights
            // stay packed under NATIVE_OPTIMIZED. On Android this exercises
            // the engine's real createRandomAccessSource (engine #922): the
            // full-file-on-heap OOM path from the field report is gone.
            val racProvider: () -> RandomAccessSource = {
                createRandomAccessSource(modelPath)
                    ?: error("createRandomAccessSource returned null on Android")
            }
            val (model, loadElapsed) = measureTimedValue {
                LlamaNetworkLoader
                    .fromGguf(racProvider, QuantPolicy.NATIVE_OPTIMIZED)
                    .load<FP32, Float>(ctx)
            }
            val runtime = OptimizedLLMRuntime(
                model = model,
                ctx = ctx,
                mode = OptimizedLLMMode.DIRECT,
                dtype = FP32::class,
                bos = 1,
            )

            val prompt = tokenizer.encode("The capital of France is")
            val out = StringBuilder()
            var generated = 0
            KernelProfile.reset()
            val decodeElapsed = measureTime {
                runtime.generate(prompt = prompt, steps = steps, temperature = 0f) { tokenId ->
                    generated++
                    out.append(tokenizer.decode(tokenId))
                }
            }

            val tokPerSec = generated.toDouble() /
                decodeElapsed.inWholeMilliseconds.coerceAtLeast(1) * 1000.0
            log("=== SmolLM2-135M Android e2e spike (NATIVE_OPTIMIZED) ===")
            log("load: ${loadElapsed.inWholeMilliseconds} ms")
            log("decode: $generated tokens in ${decodeElapsed.inWholeMilliseconds} ms")
            log("tok/s: ${(tokPerSec * 100).toLong() / 100.0}")
            log("output: ${out.toString().take(120)}")
            // Where decode time actually goes: packed-quant (JNI NEON when
            // registered) vs dense FP32 scalar vs generic matmuls.
            KernelProfile.report().lines().forEach { log(it) }

            assertTrue("no tokens generated", generated > 0)
            assertTrue("empty decode output", out.isNotBlank())
        }
    }
}
