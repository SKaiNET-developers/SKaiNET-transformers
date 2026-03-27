package sk.ainet.performance

public interface BenchmarkScenario {
    public val id: String
    public val description: String
}

public interface BenchmarkRunner<C, R> {
    public suspend fun run(config: C): R
}

public interface ModelReference {
    public val raw: String
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
    val notes: List<String> = emptyList(),
)

public data class BenchmarkRunResult(
    val scenarioId: String,
    val target: String,
    val startedAtEpochMillis: Long,
    val finishedAtEpochMillis: Long,
    val cases: List<BenchmarkCaseResult>,
)

public data class BenchmarkRunRequest(
    val scenarioId: String,
    val target: String = "jvm",
    val model: String? = null,
    val outputFormat: BenchmarkOutputFormat = BenchmarkOutputFormat.CONSOLE,
)

public data class ResolvedBenchmarkScenario(
    val scenario: BenchmarkScenario,
    val supportedTargets: Set<String>,
)
