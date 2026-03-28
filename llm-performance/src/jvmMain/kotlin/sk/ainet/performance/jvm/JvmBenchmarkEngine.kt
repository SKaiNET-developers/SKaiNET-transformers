package sk.ainet.performance.jvm

import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.Path
import kotlin.io.path.absolutePathString
import kotlin.system.measureNanoTime
import kotlin.time.measureTime
import kotlinx.coroutines.delay
import sk.ainet.apps.kllama.CpuAttentionBackend
import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.apps.kllama.LlamaIngestion
import sk.ainet.apps.kllama.LlamaLoadConfig
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.types.FP32
import sk.ainet.models.llama.LlamaNetworkLoader
import sk.ainet.models.llama.LlamaRuntime
import sk.ainet.performance.BenchmarkCaseResult
import sk.ainet.performance.BenchmarkCaseStatus
import sk.ainet.performance.BenchmarkMetric
import sk.ainet.performance.BenchmarkRunRequest
import sk.ainet.performance.BenchmarkRunResult
import sk.ainet.performance.BenchmarkRunner
import sk.ainet.performance.BenchmarkScenario
import sk.ainet.performance.ResolvedBenchmarkScenario

private const val MODEL_PROPERTY = "skainet.model.path"
private const val MODEL_ENV = "SKAINET_MODEL_PATH"

private interface JvmBenchmarkScenario : BenchmarkScenario {
    suspend fun execute(request: BenchmarkRunRequest, resolver: JvmModelResolver = JvmModelResolver()): BenchmarkRunResult
}

private class JvmBenchmarkRegistry(
    private val scenarios: List<JvmBenchmarkScenario> = listOf(
        JvmSmokeBenchmarkScenario(),
        LlamaRuntimeThroughputScenario(),
    ),
) {
    fun list(): List<ResolvedBenchmarkScenario> = scenarios.map { scenario ->
        ResolvedBenchmarkScenario(scenario = scenario, supportedTargets = setOf("jvm"))
    }

    fun get(id: String): JvmBenchmarkScenario = scenarios.firstOrNull { it.id == id }
        ?: error("Unknown benchmark scenario '$id'. Use list-scenarios to inspect available scenarios.")
}

class JvmBenchmarkOrchestrator(
    private val resolver: JvmModelResolver = JvmModelResolver(),
) : BenchmarkRunner<BenchmarkRunRequest, BenchmarkRunResult> {
    private val registry: JvmBenchmarkRegistry = JvmBenchmarkRegistry()

    override suspend fun run(config: BenchmarkRunRequest): BenchmarkRunResult {
        require(config.target == "jvm") {
            "Unsupported target '${config.target}'. Only 'jvm' is available in the current implementation."
        }
        return registry.get(config.scenarioId).execute(config, resolver)
    }

    fun listScenarios(): List<ResolvedBenchmarkScenario> = registry.list()

    fun resolveModelReference(reference: String?): ResolvedJvmModel = resolver.resolve(reference)
}

data class ResolvedJvmModel(
    val path: Path,
    val source: String,
    val reference: String,
)

class JvmModelResolver(
    private val systemPropertyLookup: (String) -> String? = System::getProperty,
    private val environmentLookup: (String) -> String? = System::getenv,
) {
    fun resolve(explicitReference: String?): ResolvedJvmModel {
        val candidate = when {
            !explicitReference.isNullOrBlank() -> explicitReference to "cli"
            !systemPropertyLookup(MODEL_PROPERTY).isNullOrBlank() -> systemPropertyLookup(MODEL_PROPERTY)!! to "system-property:$MODEL_PROPERTY"
            !environmentLookup(MODEL_ENV).isNullOrBlank() -> environmentLookup(MODEL_ENV)!! to "env:$MODEL_ENV"
            else -> error("Model path not provided. Use --model/--model-path, -D$MODEL_PROPERTY, or $MODEL_ENV.")
        }

        val path = Path(candidate.first)
        require(Files.exists(path)) { "Model path does not exist: $path" }
        require(Files.isRegularFile(path)) { "Model path is not a regular file: $path" }

        return ResolvedJvmModel(path = path, source = candidate.second, reference = candidate.first)
    }
}

private class JvmSmokeBenchmarkScenario : JvmBenchmarkScenario {
    override val id: String = "jvm-smoke"
    override val description: String = "Verifies llm-performance orchestration, timing, and result reporting on JVM."

    override suspend fun execute(request: BenchmarkRunRequest, resolver: JvmModelResolver): BenchmarkRunResult {
        val startedAt = System.currentTimeMillis()
        val durationNanos = measureNanoTime { delay(5) }
        val finishedAt = System.currentTimeMillis()

        val case = BenchmarkCaseResult(
            caseId = "smoke-delay",
            status = BenchmarkCaseStatus.SUCCESS,
            metrics = listOf(BenchmarkMetric("duration", durationNanos / 1_000_000.0, "ms")),
            runtime = "smoke",
            notes = buildList {
                add("Scenario completed successfully on JVM.")
                if (request.modelReference != null) add("Model reference received: ${request.modelReference}")
            },
        )

        return BenchmarkRunResult(
            scenarioId = id,
            target = request.target,
            startedAtEpochMillis = startedAt,
            finishedAtEpochMillis = finishedAt,
            modelReference = request.modelReference,
            cases = listOf(case),
        )
    }
}

private class LlamaRuntimeThroughputScenario : JvmBenchmarkScenario {
    override val id: String = "llama-runtime-throughput"
    override val description: String = "Compare old LlamaRuntime vs DIRECT DSL vs OPTIMIZED runtime throughput."

    private val prompts: List<NamedPrompt> = listOf(
        NamedPrompt("short", "Hello"),
        NamedPrompt("medium", "The capital of France is"),
        NamedPrompt("long", "Explain the theory of relativity in simple terms for a student who has never studied physics before"),
    )

    override suspend fun execute(request: BenchmarkRunRequest, resolver: JvmModelResolver): BenchmarkRunResult {
        val resolvedModel = resolver.resolve(request.modelReference)
        val startedAt = System.currentTimeMillis()

        val tokenizer = JvmRandomAccessSource.open(resolvedModel.path.toString()).use { source ->
            GGUFTokenizer.fromRandomAccessSource(source)
        }
        val promptPlans = prompts.map { prompt ->
            PromptPlan(prompt = prompt, promptTokens = tokenizer.encode(prompt.text))
        }

        val adapters: List<LlamaRuntimeAdapter> = listOf(
            LegacyLlamaAdapter(resolvedModel.path),
            DirectDslLlamaAdapter(resolvedModel.path),
            OptimizedLlamaAdapter(resolvedModel.path),
        )

        val results = mutableListOf<BenchmarkCaseResult>()
        for (adapter in adapters) {
            results += adapter.runAllCases(
                promptPlans = promptPlans,
                stepCounts = request.steps,
                warmupRuns = request.warmupRuns,
                measuredRuns = request.measuredRuns,
            )
            System.gc()
        }

        val finishedAt = System.currentTimeMillis()
        return BenchmarkRunResult(
            scenarioId = id,
            target = request.target,
            startedAtEpochMillis = startedAt,
            finishedAtEpochMillis = finishedAt,
            modelReference = request.modelReference ?: resolvedModel.reference,
            resolvedModelPath = resolvedModel.path.absolutePathString(),
            modelResolutionSource = resolvedModel.source,
            cases = results,
        )
    }
}

private data class NamedPrompt(
    val label: String,
    val text: String,
)

private data class PromptPlan(
    val prompt: NamedPrompt,
    val promptTokens: IntArray,
)

private interface LlamaRuntimeAdapter {
    val runtimeName: String

    suspend fun runAllCases(
        promptPlans: List<PromptPlan>,
        stepCounts: List<Int>,
        warmupRuns: Int,
        measuredRuns: Int,
    ): List<BenchmarkCaseResult>
}

private class LegacyLlamaAdapter(
    private val modelPath: Path,
) : LlamaRuntimeAdapter {
    override val runtimeName: String = "LlamaRuntime"

    override suspend fun runAllCases(
        promptPlans: List<PromptPlan>,
        stepCounts: List<Int>,
        warmupRuns: Int,
        measuredRuns: Int,
    ): List<BenchmarkCaseResult> {
        val ctx = DirectCpuExecutionContext()
        val ingestion = LlamaIngestion<FP32>(
            ctx = ctx,
            dtype = FP32::class,
            config = LlamaLoadConfig(
                quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
                allowQuantized = true,
            ),
        )
        val oldWeights = ingestion.loadStreaming {
            JvmRandomAccessSource.open(modelPath.toString())
        }
        val backend = CpuAttentionBackend<FP32>(ctx, oldWeights, FP32::class)
        @Suppress("DEPRECATION")
        val runtime = LlamaRuntime<FP32>(ctx, oldWeights, backend, FP32::class)

        return benchmarkCases(promptPlans, stepCounts, warmupRuns, measuredRuns) { tokens, steps ->
            runtime.reset()
            runtime.generate(tokens, steps, 0.0f) { _ -> }
        }
    }

    private fun benchmarkCases(
        promptPlans: List<PromptPlan>,
        stepCounts: List<Int>,
        warmupRuns: Int,
        measuredRuns: Int,
        run: (IntArray, Int) -> Unit,
    ): List<BenchmarkCaseResult> {
        val results = mutableListOf<BenchmarkCaseResult>()
        for (steps in stepCounts) {
            for ((prompt, promptTokens) in promptPlans) {
                results += benchmarkSingleCase(prompt, promptTokens, steps, warmupRuns, measuredRuns, run)
            }
        }
        return results
    }

    private fun benchmarkSingleCase(
        prompt: NamedPrompt,
        promptTokens: IntArray,
        steps: Int,
        warmupRuns: Int,
        measuredRuns: Int,
        run: (IntArray, Int) -> Unit,
    ): BenchmarkCaseResult {
        repeat(warmupRuns) { run(promptTokens, steps) }
        val measurements = (1..measuredRuns)
            .map { measureTime { run(promptTokens, steps) }.inWholeMilliseconds }
            .sorted()
        val medianMillis = measurements[measuredRuns / 2].coerceAtLeast(1)
        val throughput = steps.toDouble() / medianMillis * 1000.0
        return BenchmarkCaseResult(
            caseId = "$runtimeName:${prompt.label}:$steps",
            status = BenchmarkCaseStatus.SUCCESS,
            runtime = runtimeName,
            promptLabel = prompt.label,
            promptTokenCount = promptTokens.size,
            steps = steps,
            metrics = listOf(
                BenchmarkMetric("throughput", throughput, "tok/s"),
                BenchmarkMetric("median_duration", medianMillis.toDouble(), "ms"),
            ),
        )
    }
}

private class DirectDslLlamaAdapter(
    private val modelPath: Path,
) : LlamaRuntimeAdapter {
    override val runtimeName: String = "DIRECT"

    override suspend fun runAllCases(
        promptPlans: List<PromptPlan>,
        stepCounts: List<Int>,
        warmupRuns: Int,
        measuredRuns: Int,
    ): List<BenchmarkCaseResult> {
        val ctx = DirectCpuExecutionContext()
        val model = LlamaNetworkLoader.fromGguf(
            randomAccessProvider = { JvmRandomAccessSource.open(modelPath.toString()) },
            quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
        ).load<FP32, Float>(ctx)
        val runtime = OptimizedLLMRuntime(
            model = model,
            ctx = ctx,
            mode = OptimizedLLMMode.DIRECT,
            dtype = FP32::class,
        )

        val results = mutableListOf<BenchmarkCaseResult>()
        for (steps in stepCounts) {
            for ((prompt, promptTokens) in promptPlans) {
                repeat(warmupRuns) {
                    runtime.reset()
                    runtime.generate(promptTokens, steps, 0.0f) { _ -> }
                }
                val measurements = (1..measuredRuns)
                    .map {
                        measureTime {
                            runtime.reset()
                            runtime.generate(promptTokens, steps, 0.0f) { _ -> }
                        }.inWholeMilliseconds
                    }
                    .sorted()
                val medianMillis = measurements[measuredRuns / 2].coerceAtLeast(1)
                val throughput = steps.toDouble() / medianMillis * 1000.0
                results += BenchmarkCaseResult(
                    caseId = "$runtimeName:${prompt.label}:$steps",
                    status = BenchmarkCaseStatus.SUCCESS,
                    runtime = runtimeName,
                    promptLabel = prompt.label,
                    promptTokenCount = promptTokens.size,
                    steps = steps,
                    metrics = listOf(
                        BenchmarkMetric("throughput", throughput, "tok/s"),
                        BenchmarkMetric("median_duration", medianMillis.toDouble(), "ms"),
                    ),
                )
            }
        }
        return results
    }
}

private class OptimizedLlamaAdapter(
    private val modelPath: Path,
) : LlamaRuntimeAdapter {
    override val runtimeName: String = "OPTIMIZED"

    override suspend fun runAllCases(
        promptPlans: List<PromptPlan>,
        stepCounts: List<Int>,
        warmupRuns: Int,
        measuredRuns: Int,
    ): List<BenchmarkCaseResult> {
        val ctx = DirectCpuExecutionContext()
        val runtime = try {
            val model = LlamaNetworkLoader.fromGguf(
                randomAccessProvider = { JvmRandomAccessSource.open(modelPath.toString()) },
                quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
            ).load<FP32, Float>(ctx)
            OptimizedLLMRuntime(
                model = model,
                ctx = ctx,
                mode = OptimizedLLMMode.OPTIMIZED,
                dtype = FP32::class,
            ).also {
                it.compile()
                it.forward(1)
                it.reset()
            }
        } catch (e: Exception) {
            null
        }

        if (runtime == null) {
            return stepCounts.flatMap { steps ->
                promptPlans.map { (prompt, promptTokens) ->
                    BenchmarkCaseResult(
                        caseId = "$runtimeName:${prompt.label}:$steps",
                        status = BenchmarkCaseStatus.SKIPPED,
                        runtime = runtimeName,
                        promptLabel = prompt.label,
                        promptTokenCount = promptTokens.size,
                        steps = steps,
                        metrics = emptyList(),
                        notes = listOf("OPTIMIZED runtime failed to initialize and was skipped."),
                    )
                }
            }
        }

        val results = mutableListOf<BenchmarkCaseResult>()
        for (steps in stepCounts) {
            for ((prompt, promptTokens) in promptPlans) {
                repeat(warmupRuns) {
                    runtime.reset()
                    runtime.generate(promptTokens, steps, 0.0f) { _ -> }
                }
                val measurements = (1..measuredRuns)
                    .map {
                        measureTime {
                            runtime.reset()
                            runtime.generate(promptTokens, steps, 0.0f) { _ -> }
                        }.inWholeMilliseconds
                    }
                    .sorted()
                val medianMillis = measurements[measuredRuns / 2].coerceAtLeast(1)
                val throughput = steps.toDouble() / medianMillis * 1000.0
                results += BenchmarkCaseResult(
                    caseId = "$runtimeName:${prompt.label}:$steps",
                    status = BenchmarkCaseStatus.SUCCESS,
                    runtime = runtimeName,
                    promptLabel = prompt.label,
                    promptTokenCount = promptTokens.size,
                    steps = steps,
                    metrics = listOf(
                        BenchmarkMetric("throughput", throughput, "tok/s"),
                        BenchmarkMetric("median_duration", medianMillis.toDouble(), "ms"),
                    ),
                )
            }
        }
        return results
    }
}

object JvmConsoleReporter {
    fun render(result: BenchmarkRunResult) {
        println("[BENCH] Scenario: ${result.scenarioId}")
        println("[BENCH] Target: ${result.target}")
        result.modelReference?.let { println("[BENCH] Model reference: $it") }
        result.resolvedModelPath?.let { println("[BENCH] Resolved model path: $it") }
        result.modelResolutionSource?.let { println("[BENCH] Model source: $it") }
        println("[BENCH] Started: ${result.startedAtEpochMillis}")
        println("[BENCH] Finished: ${result.finishedAtEpochMillis}")
        println()

        val groupedByCase = result.cases.mapNotNull { case -> case.steps?.let { it to case } }.groupBy({ it.first }, { it.second })
        groupedByCase.toSortedMap().forEach { (steps, cases) ->
            cases.groupBy { it.promptLabel }.forEach { (promptLabel, promptCases) ->
                val tokenCount = promptCases.firstOrNull()?.promptTokenCount
                println("[BENCH] steps=$steps, prompt=$promptLabel (${tokenCount ?: 0} tokens)")
                promptCases.forEach { case ->
                    val throughput = case.metrics.firstOrNull { it.name == "throughput" }?.value
                    val suffix = throughput?.let { "${"%.2f".format(it)} tok/s" } ?: case.status.name
                    println("  ${case.runtime ?: case.caseId}: $suffix")
                    case.notes.forEach { note -> println("  note: $note") }
                }
                println()
            }
        }

        println("[BENCH] ========== SUMMARY ==========")
        println("[BENCH] | Runtime      | Steps | Short  | Medium | Long   |")
        println("[BENCH] |--------------|-------|--------|--------|--------|")
        for (runtime in listOf("LlamaRuntime", "DIRECT", "OPTIMIZED")) {
            for (steps in result.cases.mapNotNull { it.steps }.distinct().sorted()) {
                val columns = listOf("short", "medium", "long").map { prompt ->
                    val case = result.cases.find { it.runtime == runtime && it.promptLabel == prompt && it.steps == steps }
                    val throughput = case?.metrics?.firstOrNull { it.name == "throughput" }?.value
                    throughput?.let { "%.2f".format(it) } ?: (case?.status?.name ?: "N/A")
                }
                println("[BENCH] | %-12s | %5d | %6s | %6s | %6s |".format(runtime, steps, columns[0], columns[1], columns[2]))
            }
        }
        println("[BENCH] ================================")
    }

    fun renderScenarioList(scenarios: List<ResolvedBenchmarkScenario>) {
        scenarios.forEach { resolved ->
            println("${resolved.scenario.id}\t${resolved.supportedTargets.joinToString()}\t${resolved.scenario.description}")
        }
    }

    fun renderResolvedModel(model: ResolvedJvmModel) {
        println("source=${model.source}")
        println("reference=${model.reference}")
        println("path=${model.path.absolutePathString()}")
    }
}
