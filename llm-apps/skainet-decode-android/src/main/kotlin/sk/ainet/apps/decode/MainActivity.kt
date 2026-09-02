package sk.ainet.apps.decode

import android.app.Activity
import android.graphics.Typeface
import android.os.Bundle
import android.text.method.ScrollingMovementMethod
import android.util.Log
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import sk.ainet.apps.decode.core.DecodeSession
import sk.ainet.io.MappedRandomAccessSource
import sk.ainet.io.gguf.AndroidGguf
import sk.ainet.lang.memory.MemoryProbe
import sk.ainet.lang.memory.sample
import sk.ainet.lang.memory.trace.AndroidTraceSink
import java.io.File
import java.util.concurrent.Executors

/**
 * The on-device skainet-decode leg (SKaiNET#1244): pick a pushed GGUF, decode, and render the
 * same `GenerationMetrics.render()` block as the JVM CLI — plus the rows only a device can
 * answer: RSS and major-fault rate for mapped weights.
 *
 * Everything measurement-relevant runs on ONE background thread ([worker]):
 * `RecordingTraceSink` is not thread-safe and `KernelDispatch.defaultSink` is assigned in the
 * `DecodeSession` constructor, so session construction, load, and the whole traced loop stay on
 * that thread. The UI only ever receives strings via [Activity.runOnUiThread].
 *
 * The full report is mirrored to logcat (tag `SkDecode`) and written to
 * `getExternalFilesDir(null)/decode-report.md` for `adb pull` — the M2-A5 harness pattern.
 */
class MainActivity : Activity() {

    private companion object {
        const val TAG = "SkDecode"

        /** KV-sizing context for the header-only pre-flight plan; prompt+steps stay well under it. */
        const val PREFLIGHT_CTX = 512
    }

    private val workerExecutor = Executors.newSingleThreadExecutor { r -> Thread(r, "skainet-decode") }
    private val worker = workerExecutor.asCoroutineDispatcher()
    private val scope = CoroutineScope(worker)

    private lateinit var pathField: EditText
    private lateinit var promptField: EditText
    private lateinit var stepsField: EditText
    private lateinit var runButton: Button
    private lateinit var output: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val modelsDir = getExternalFilesDir(null)
        val defaultModel = File(modelsDir, "model.gguf")
        val ggufs = modelsDir?.listFiles { f -> f.name.endsWith(".gguf") }?.map { it.name }.orEmpty()

        val pad = (8 * resources.displayMetrics.density).toInt()
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad)
        }

        root.addView(TextView(this).apply {
            text = if (ggufs.isEmpty()) {
                "No .gguf in ${modelsDir?.absolutePath} — push one:\nadb push model.gguf ${modelsDir?.absolutePath}/model.gguf"
            } else {
                "Models in ${modelsDir?.absolutePath}:\n" + ggufs.joinToString("\n") { "  $it" }
            }
            typeface = Typeface.MONOSPACE
            textSize = 11f
        })

        pathField = EditText(this).apply {
            hint = "model path"
            setText(defaultModel.absolutePath)
            maxLines = 2
        }
        promptField = EditText(this).apply {
            hint = "prompt"
            setText("Once upon a time")
        }
        stepsField = EditText(this).apply {
            hint = "steps"
            setText("32")
        }
        runButton = Button(this).apply {
            text = "Run"
            setOnClickListener { runDecode() }
        }
        output = TextView(this).apply {
            typeface = Typeface.MONOSPACE
            textSize = 11f
            movementMethod = ScrollingMovementMethod()
        }

        root.addView(pathField)
        root.addView(promptField)
        root.addView(stepsField)
        root.addView(runButton)
        root.addView(output, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))

        setContentView(ScrollView(this).apply { addView(root) })
    }

    override fun onDestroy() {
        super.onDestroy()
        scope.cancel()
        workerExecutor.shutdownNow()
    }

    private fun runDecode() {
        val path = pathField.text.toString().trim()
        val prompt = promptField.text.toString()
        val steps = stepsField.text.toString().trim().toIntOrNull() ?: 32
        runButton.isEnabled = false
        output.text = ""

        val report = StringBuilder()
        fun line(s: String) {
            Log.i(TAG, s)
            report.appendLine(s)
            runOnUiThread { output.append(s + "\n") }
        }

        scope.launch {
            try {
                if (!File(path).canRead()) {
                    line("cannot read $path — push a model first (see the hint above)")
                    return@launch
                }

                // Pre-flight: refuse before a byte of payload is read (M2-F6).
                val device = AndroidGguf.deviceMemory(this@MainActivity)
                line("device: ram ${device.totalRamBytes.mib()} MiB (avail ${device.availableRamBytes.mib()} MiB), " +
                        "heap cap ${device.heapMaxBytes.mib()} MiB" + if (device.lowMemory) " [LOW MEMORY]" else "")
                val fit = AndroidGguf.fits(this@MainActivity, path, ctx = PREFLIGHT_CTX, weightsMapped = true)
                line("fit @ctx=$PREFLIGHT_CTX: " + if (fit.fits) "OK" else "REFUSED (pool: ${fit.blockingPool})")
                if (!fit.fits) {
                    fit.suggestions.forEach { line("  suggestion: $it") }
                    return@launch
                }

                val before = MemoryProbe.sample()
                line("rss before: $before")

                // Session construction assigns KernelDispatch.defaultSink — must happen here,
                // on the same single thread that runs the traced loop.
                val session = DecodeSession(extraSink = AndroidTraceSink())
                val result = session.run(
                    sourceProvider = { MappedRandomAccessSource.open(path) },
                    prompt = prompt,
                    steps = steps,
                    onModelInfo = { info ->
                        line("model: ${info.architecture} (${info.family}), ${info.blockCount} blocks, " +
                                "vocab ${info.vocabSize}, embd ${info.embeddingLength}")
                    },
                    onGenerationStart = { line("generating…") },
                    onToken = { piece -> runOnUiThread { output.append(piece) } },
                )
                runOnUiThread { output.append("\n") }
                Log.i(TAG, "text: ${result.text}")
                report.appendLine().appendLine("```").appendLine(result.text).appendLine("```")

                line("")
                line(result.metrics.render())
                if (result.droppedTraceEvents > 0) {
                    line("WARNING: trace ring dropped ${result.droppedTraceEvents} events — early prefill spans may be undercounted")
                }

                val after = MemoryProbe.sample()
                line("")
                line("--- device footer ---")
                line("rss after:  $after (peak not tracked; last decode-span RSS is in the metrics above)")
                line("major faults over run: ${after.majorFaultsSince(before) ?: "—"}")
                line("minor faults over run: ${minorDelta(before.minorFaults, after.minorFaults)}")
                line("heap: max ${Runtime.getRuntime().maxMemory().mib()} MiB, " +
                        "used ${(Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()).mib()} MiB")

                val reportFile = File(getExternalFilesDir(null), "decode-report.md")
                reportFile.writeText(report.toString())
                line("report written: ${reportFile.absolutePath}")
            } catch (t: Throwable) {
                Log.e(TAG, "decode failed", t)
                line("FAILED: $t")
            } finally {
                runOnUiThread { runButton.isEnabled = true }
            }
        }
    }

    private fun Long.mib(): Long = this / (1024 * 1024)

    private fun minorDelta(before: Long?, after: Long?): String =
        if (before != null && after != null) (after - before).toString() else "—"
}
