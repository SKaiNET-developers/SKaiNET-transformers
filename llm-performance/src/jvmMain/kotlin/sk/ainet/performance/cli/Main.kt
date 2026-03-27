package sk.ainet.performance.cli

import kotlinx.cli.ArgParser
import kotlinx.cli.ArgType
import kotlinx.cli.Subcommand
import kotlinx.cli.default
import kotlinx.cli.required
import kotlinx.coroutines.runBlocking
import sk.ainet.performance.BenchmarkOutputFormat
import sk.ainet.performance.BenchmarkRunRequest
import sk.ainet.performance.jvm.JvmBenchmarkOrchestrator
import sk.ainet.performance.jvm.JvmConsoleReporter

fun main(args: Array<String>) {
    val parser = ArgParser("llm-performance")
    val orchestrator = JvmBenchmarkOrchestrator()

    parser.subcommands(
        RunCommand(orchestrator),
        ListScenariosCommand(orchestrator),
        ResolveModelCommand(orchestrator),
    )

    parser.parse(args)
}

private class RunCommand(
    private val orchestrator: JvmBenchmarkOrchestrator,
) : Subcommand(name = "run", actionDescription = "Run a benchmark scenario") {
    private val scenario by option(
        type = ArgType.String,
        fullName = "scenario",
        description = "Benchmark scenario id",
    ).required()

    private val target by option(
        type = ArgType.String,
        fullName = "target",
        description = "Execution target",
    ).default("jvm")

    private val model by option(
        type = ArgType.String,
        fullName = "model",
        description = "Model reference",
    )

    private val format by option(
        type = ArgType.String,
        fullName = "format",
        description = "Output format: console or json",
    ).default("console")

    override fun execute() {
        val outputFormat = when (format.lowercase()) {
            "console" -> BenchmarkOutputFormat.CONSOLE
            "json" -> BenchmarkOutputFormat.JSON
            else -> error("Unsupported format '$format'.")
        }

        val result = runBlocking {
            orchestrator.run(
                BenchmarkRunRequest(
                    scenarioId = scenario,
                    target = target,
                    model = model,
                    outputFormat = outputFormat,
                )
            )
        }

        when (outputFormat) {
            BenchmarkOutputFormat.CONSOLE -> JvmConsoleReporter.render(result)
            BenchmarkOutputFormat.JSON -> println(result)
        }
    }
}

private class ListScenariosCommand(
    private val orchestrator: JvmBenchmarkOrchestrator,
) : Subcommand(name = "list-scenarios", actionDescription = "List available scenarios") {
    override fun execute() {
        JvmConsoleReporter.renderScenarioList(orchestrator.listScenarios())
    }
}

private class ResolveModelCommand(
    private val orchestrator: JvmBenchmarkOrchestrator,
) : Subcommand(name = "resolve-model", actionDescription = "Resolve a model reference") {
    private val model by option(
        type = ArgType.String,
        fullName = "model",
        description = "Model reference to resolve",
    ).required()

    override fun execute() {
        JvmConsoleReporter.renderResolvedModel(orchestrator.resolveModelReference(model))
    }
}
