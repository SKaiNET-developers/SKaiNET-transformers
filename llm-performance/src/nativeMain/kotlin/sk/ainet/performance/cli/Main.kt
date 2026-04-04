package sk.ainet.performance.cli

import kotlinx.coroutines.runBlocking
import sk.ainet.performance.BenchmarkOutputFormat
import sk.ainet.performance.BenchmarkRunRequest
import sk.ainet.performance.native.NativeBenchmarkOrchestrator
import sk.ainet.performance.native.NativeConsoleReporter
import sk.ainet.performance.native.availableNativeBackends

fun main(args: Array<String>) {
    val orchestrator = NativeBenchmarkOrchestrator()

    if (args.isEmpty()) {
        printUsage()
        return
    }

    when (args[0]) {
        "list-scenarios" -> {
            NativeConsoleReporter.renderScenarioList(orchestrator.listScenarios())
        }
        "run" -> {
            val parsed = parseRunArgs(args.drop(1))
            val result = runBlocking {
                orchestrator.run(
                    BenchmarkRunRequest(
                        scenarioId = parsed.scenario,
                        target = "macos",
                        modelReference = parsed.modelPath,
                        outputFormat = parsed.format,
                        warmupRuns = parsed.warmupRuns,
                        measuredRuns = parsed.measuredRuns,
                        steps = parsed.steps,
                    )
                )
            }
            when (parsed.format) {
                BenchmarkOutputFormat.CONSOLE -> NativeConsoleReporter.render(result)
                BenchmarkOutputFormat.JSON -> println(result)
            }
        }
        else -> {
            println("Unknown command: ${args[0]}")
            printUsage()
        }
    }
}

private fun printUsage() {
    println("Usage: llm-performance <command> [options]")
    println()
    println("Commands:")
    println("  list-scenarios                  List available benchmark scenarios")
    println("  run --scenario <id> [options]   Run a benchmark scenario")
    println()
    println("Run options:")
    println("  --scenario <id>          Benchmark scenario id (required)")
    println("  --model-path <path>      Path to GGUF model file (required)")
    println("  --warmup-runs <n>        Warmup iterations per case (default: 3)")
    println("  --measured-runs <n>      Measured iterations per case (default: 3)")
    println("  --steps <n,n,...>        Comma-separated generation step counts (default: 16,64)")
    println("  --format <fmt>           Output format: console or json (default: console)")
    println()
    println("Available backends: ${availableNativeBackends().joinToString(", ")}")
}

private data class RunArgs(
    val scenario: String,
    val modelPath: String?,
    val format: BenchmarkOutputFormat,
    val warmupRuns: Int,
    val measuredRuns: Int,
    val steps: List<Int>,
)

private fun parseRunArgs(args: List<String>): RunArgs {
    var scenario: String? = null
    var modelPath: String? = null
    var format = BenchmarkOutputFormat.CONSOLE
    var warmupRuns = 3
    var measuredRuns = 3
    var steps = listOf(16, 64)

    val iter = args.iterator()
    while (iter.hasNext()) {
        when (val arg = iter.next()) {
            "--scenario" -> scenario = iter.next()
            "--model-path", "--model" -> modelPath = iter.next()
            "--format" -> {
                format = when (iter.next().lowercase()) {
                    "json" -> BenchmarkOutputFormat.JSON
                    else -> BenchmarkOutputFormat.CONSOLE
                }
            }
            "--warmup-runs" -> warmupRuns = iter.next().toInt()
            "--measured-runs" -> measuredRuns = iter.next().toInt()
            "--steps" -> {
                steps = iter.next().split(',').map { it.trim().toInt() }
            }
            else -> println("Warning: unknown option '$arg'")
        }
    }

    requireNotNull(scenario) { "--scenario is required" }
    return RunArgs(scenario, modelPath, format, warmupRuns, measuredRuns, steps)
}
