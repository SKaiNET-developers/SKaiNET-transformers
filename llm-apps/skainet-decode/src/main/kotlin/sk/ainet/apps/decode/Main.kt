package sk.ainet.apps.decode

import kotlinx.coroutines.runBlocking
import sk.ainet.apps.decode.core.DecodeSession
import sk.ainet.io.JvmRandomAccessSource
import kotlin.system.exitProcess

/**
 * `skainet-decode` (SKaiNET#1129): load a GGUF, decode, and report `GenerationMetrics` — TTFT,
 * prefill and decode tok/s, **effective memory bandwidth**, kernel/adapter shares, page-fault
 * rate. These are the numbers the SKEEP-003 memory work exists to move, measured on a real
 * model instead of a microbench.
 *
 * The flow itself lives in [DecodeSession] (`llm-apps:skainet-decode-core`, SKaiNET#1244) so the
 * Android activity leg reports the same numbers from the same loop; this CLI is a thin caller
 * that owns only argument parsing, the JVM source, and stdout.
 */
fun main(args: Array<String>) {
    var model: String? = null
    var prompt = "The capital of France is"
    var steps = 32
    var temperature = 0f
    var i = 0
    while (i < args.size) {
        when (val a = args[i]) {
            "-m" -> model = args.getOrNull(++i)
            "-p" -> prompt = args.getOrNull(++i) ?: prompt
            "-s" -> steps = args.getOrNull(++i)?.toIntOrNull() ?: steps
            "-k" -> temperature = args.getOrNull(++i)?.toFloatOrNull() ?: temperature
            else -> { System.err.println("unknown arg '$a'"); exitProcess(1) }
        }
        i++
    }
    val modelPath = model ?: run {
        System.err.println("usage: skainet-decode -m <model.gguf> [-p prompt] [-s steps] [-k temperature]")
        exitProcess(1)
    }

    val session = DecodeSession()

    runBlocking {
        val report = session.run(
            sourceProvider = { JvmRandomAccessSource.open(modelPath) },
            prompt = prompt,
            steps = steps,
            temperature = temperature,
            onModelInfo = { info ->
                println("Model: $modelPath (${info.family.displayName}, ${info.blockCount} layers, vocab=${info.vocabSize})")
            },
            onGenerationStart = {
                println("Generating $steps tokens (temperature=$temperature)...")
                print(prompt)
            },
            onToken = ::print,
        )
        println()
        println()

        println(report.metrics.render())
        if (report.droppedTraceEvents > 0) {
            println("(trace ring dropped ${report.droppedTraceEvents} events — early prefill spans may be undercounted)")
        }
    }
}
