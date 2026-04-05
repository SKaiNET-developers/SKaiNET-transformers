package sk.ainet.performance

public interface BenchmarkScenario {
    public val id: String
    public val description: String
}

public interface BenchmarkRunner<C, R> {
    public suspend fun run(config: C): R
}

public enum class BenchmarkCaseStatus {
    SUCCESS,
    SKIPPED,
    FAILED,
}

public enum class BenchmarkOutputFormat {
    CONSOLE,
    JSON,
}

public data class BenchmarkMetric(
    val name: String,
    val value: Double,
    val unit: String,
)

public data class BenchmarkCaseResult(
    val caseId: String,
    val status: BenchmarkCaseStatus,
    val metrics: List<BenchmarkMetric>,
    val runtime: String? = null,
    val promptLabel: String? = null,
    val promptTokenCount: Int? = null,
    val steps: Int? = null,
    val notes: List<String> = emptyList(),
)

public data class BenchmarkRunResult(
    val scenarioId: String,
    val target: String,
    val startedAtEpochMillis: Long,
    val finishedAtEpochMillis: Long,
    val modelReference: String? = null,
    val resolvedModelPath: String? = null,
    val modelResolutionSource: String? = null,
    val cases: List<BenchmarkCaseResult>,
)

public data class BenchmarkRunRequest(
    val scenarioId: String,
    val target: String = "jvm",
    val modelReference: String? = null,
    val outputFormat: BenchmarkOutputFormat = BenchmarkOutputFormat.CONSOLE,
    val warmupRuns: Int = 3,
    val measuredRuns: Int = 3,
    val steps: List<Int> = listOf(16, 64),
)

public data class ResolvedBenchmarkScenario(
    val scenario: BenchmarkScenario,
    val supportedTargets: Set<String>,
)
