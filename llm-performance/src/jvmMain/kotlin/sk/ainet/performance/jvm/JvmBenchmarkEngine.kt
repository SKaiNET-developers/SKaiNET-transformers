package sk.ainet.performance.jvm

import kotlin.system.measureNanoTime
import kotlinx.coroutines.delay
import sk.ainet.performance.BenchmarkCaseResult
import sk.ainet.performance.BenchmarkCaseStatus
import sk.ainet.performance.BenchmarkMetric
import sk.ainet.performance.BenchmarkOutputFormat
import sk.ainet.performance.BenchmarkRunRequest
import sk.ainet.performance.BenchmarkRunResult
import sk.ainet.performance.BenchmarkRunner
import sk.ainet.performance.BenchmarkScenario
import sk.ainet.performance.ResolvedBenchmarkScenario

internal interface JvmBenchmarkScenario : BenchmarkScenario {
    suspend fun execute(request: BenchmarkRunRequest): BenchmarkRunResult
}

internal class JvmBenchmarkRegistry(
    private val scenarios: List<JvmBenchmarkScenario> = listOf(JvmSmokeBenchmarkScenario()),
) {
    fun list(): List<ResolvedBenchmarkScenario> = scenarios.map { scenario ->
        ResolvedBenchmarkScenario(
            scenario = scenario,
            supportedTargets = setOf("jvm"),
        )
    }

    fun get(id: String): JvmBenchmarkScenario = scenarios.firstOrNull { it.id == id }
        ?: error("Unknown benchmark scenario '$id'. Use list-scenarios to inspect available scenarios.")
}

class JvmBenchmarkOrchestrator : BenchmarkRunner<BenchmarkRunRequest, BenchmarkRunResult> {
    private val registry: JvmBenchmarkRegistry = JvmBenchmarkRegistry()
    override suspend fun run(config: BenchmarkRunRequest): BenchmarkRunResult {
        require(config.target == "jvm") {
            "Unsupported target '${config.target}'. Only 'jvm' is available in the current implementation."
        }
        return registry.get(config.scenarioId).execute(config)
    }

    fun listScenarios(): List<ResolvedBenchmarkScenario> = registry.list()

    fun resolveModelReference(raw: String): String = raw
}

internal class JvmSmokeBenchmarkScenario : JvmBenchmarkScenario {
    override val id: String = "jvm-smoke"
    override val description: String = "Verifies llm-performance orchestration, timing, and result reporting on JVM."

    override suspend fun execute(request: BenchmarkRunRequest): BenchmarkRunResult {
        val startedAt = System.currentTimeMillis()
        val durationNanos = measureNanoTime {
            delay(5)
        }
        val finishedAt = System.currentTimeMillis()

        val case = BenchmarkCaseResult(
            caseId = "smoke-delay",
            status = BenchmarkCaseStatus.SUCCESS,
            metrics = listOf(
                BenchmarkMetric(
                    name = "duration",
                    value = durationNanos / 1_000_000.0,
                    unit = "ms",
                )
            ),
            notes = buildList {
                add("Scenario completed successfully on JVM.")
                if (request.model != null) {
                    add("Model reference received but not yet resolved in PERF-1 through PERF-5: ${request.model}")
                }
                add("Output format requested: ${request.outputFormat.name.lowercase()}")
            },
        )

        return BenchmarkRunResult(
            scenarioId = id,
            target = request.target,
            startedAtEpochMillis = startedAt,
            finishedAtEpochMillis = finishedAt,
            cases = listOf(case),
        )
    }
}

object JvmConsoleReporter {
    fun render(result: BenchmarkRunResult) {
        println("Scenario: ${result.scenarioId}")
        println("Target: ${result.target}")
        println("Started: ${result.startedAtEpochMillis}")
        println("Finished: ${result.finishedAtEpochMillis}")
        println()
        result.cases.forEach { case ->
            println("Case: ${case.caseId}")
            println("Status: ${case.status}")
            case.metrics.forEach { metric ->
                println("Metric: ${metric.name}=${"%.3f".format(metric.value)} ${metric.unit}")
            }
            case.notes.forEach { note ->
                println("Note: $note")
            }
            println()
        }
    }

    fun renderScenarioList(scenarios: List<ResolvedBenchmarkScenario>) {
        scenarios.forEach { resolved ->
            println("${resolved.scenario.id}\t${resolved.supportedTargets.joinToString()}\t${resolved.scenario.description}")
        }
    }

    fun renderResolvedModel(reference: String) {
        println(reference)
    }
}
