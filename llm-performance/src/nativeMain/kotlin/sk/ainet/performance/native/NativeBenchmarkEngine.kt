package sk.ainet.performance.native

import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlin.time.measureTime
import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.generate
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.types.FP32
import sk.ainet.lang.nn.dsl.decoder.DecoderGgufWeightLoader
import sk.ainet.models.llama.LlamaNetworkLoader
import sk.ainet.performance.BenchmarkCaseResult
import sk.ainet.performance.BenchmarkCaseStatus
import sk.ainet.performance.BenchmarkMetric
import sk.ainet.performance.BenchmarkRunRequest
import sk.ainet.performance.BenchmarkRunResult
import sk.ainet.performance.BenchmarkRunner
import sk.ainet.performance.BenchmarkScenario
import sk.ainet.performance.ResolvedBenchmarkScenario

private fun log(message: String) {
    println("[BENCH] $message")
}

@OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)
private fun epochMillis(): Long {
    return platform.posix.time(null) * 1000L
}

private fun formatDouble2(value: Double): String {
    val rounded = kotlin.math.round(value * 100.0) / 100.0
    val intPart = rounded.toLong()
    val fracPart = ((rounded - intPart) * 100).toLong().let { kotlin.math.abs(it) }
    return "$intPart.${fracPart.toString().padStart(2, '0')}"
}

private fun formatDouble1(value: Double): String {
    val rounded = kotlin.math.round(value * 10.0) / 10.0
    val intPart = rounded.toLong()
    val fracPart = ((rounded - intPart) * 10).toLong().let { kotlin.math.abs(it) }
    return "$intPart.$fracPart"
}

internal expect fun availableNativeBackends(): List<String>

internal data class NamedPrompt(
    val label: String,
    val text: String,
)

internal data class PromptPlan(
    val prompt: NamedPrompt,
    val promptTokens: IntArray,
)

internal class CpuNativeDslAdapter(
    private val modelPathStr: String,
) {
    val runtimeName: String = "CPU"

    suspend fun runAllCases(
        promptPlans: List<PromptPlan>,
        stepCounts: List<Int>,
        warmupRuns: Int,
        measuredRuns: Int,
    ): List<BenchmarkCaseResult> {
        val ctx = DirectCpuExecutionContext()
        val modelPath = Path(modelPathStr)
        log("  $runtimeName | loading model...")
        val weights = DecoderGgufWeightLoader(
            sourceProvider = { SystemFileSystem.source(modelPath).buffered() },
        ).loadToMap<FP32, Float>(ctx)
        val model = LlamaNetworkLoader.fromWeights(weights)
        val runtime = OptimizedLLMRuntime(
            model = model,
            ctx = ctx,
            mode = OptimizedLLMMode.DIRECT,
            dtype = FP32::class,
            bos = weights.metadata.bosTokenId,
        )
        log("  $runtimeName | model loaded")

        val results = mutableListOf<BenchmarkCaseResult>()
        for (steps in stepCounts) {
            for ((prompt, promptTokens) in promptPlans) {
                log("  $runtimeName | prompt=${prompt.label} steps=$steps | warming up ($warmupRuns runs)...")
                repeat(warmupRuns) { i ->
                    runtime.reset()
                    runtime.generate(promptTokens, steps, 0.0f) { _ -> }
                    log("    warmup ${i + 1}/$warmupRuns done")
                }
                log("  $runtimeName | prompt=${prompt.label} steps=$steps | measuring ($measuredRuns runs)...")
                val measurements = (1..measuredRuns).map { i ->
                    val ms = measureTime {
                        runtime.reset()
                        runtime.generate(promptTokens, steps, 0.0f) { _ -> }
                    }.inWholeMilliseconds
                    log("    measured $i/$measuredRuns: ${ms}ms")
                    ms
                }.sorted()

                val medianMillis = measurements[measuredRuns / 2].coerceAtLeast(1)
                val throughput = steps.toDouble() / medianMillis * 1000.0
                log("  $runtimeName | prompt=${prompt.label} steps=$steps | median=${medianMillis}ms throughput=${formatDouble2(throughput)} tok/s")

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

internal class NativeCpuThroughputScenario : BenchmarkScenario {
    override val id: String = "native-cpu-throughput"
    override val description: String = "DSL CPU throughput on native (macOS)."

    private val prompts: List<NamedPrompt> = listOf(
        NamedPrompt("short", "Hello"),
        NamedPrompt("medium", "The capital of France is"),
        NamedPrompt("long", "Explain the theory of relativity in simple terms for a student who has never studied physics before"),
    )

    suspend fun execute(request: BenchmarkRunRequest): BenchmarkRunResult {
        val modelPathStr = request.modelReference
            ?: error("Model path required. Use --model-path.")
        val modelPath = Path(modelPathStr)
        require(SystemFileSystem.exists(modelPath)) { "Model not found: $modelPathStr" }

        val startedAt = epochMillis()

        log("Resolved model: $modelPathStr")
        log("Tokenizing prompts...")
        val tokenizer = GGUFTokenizer.fromSource(
            SystemFileSystem.source(modelPath).buffered()
        )
        val promptPlans = prompts.map { prompt ->
            PromptPlan(prompt = prompt, promptTokens = tokenizer.encode(prompt.text))
        }
        log("Prompts tokenized: ${promptPlans.joinToString { "${it.prompt.label}(${it.promptTokens.size} tokens)" }}")

        val adapter = CpuNativeDslAdapter(modelPathStr)
        log("=== Backend: ${adapter.runtimeName} ===")
        val results = adapter.runAllCases(
            promptPlans = promptPlans,
            stepCounts = request.steps,
            warmupRuns = request.warmupRuns,
            measuredRuns = request.measuredRuns,
        )
        val successCount = results.count { it.status == BenchmarkCaseStatus.SUCCESS }
        log("${adapter.runtimeName} finished: $successCount/${results.size} cases succeeded")

        val finishedAt = epochMillis()
        val elapsedSec = (finishedAt - startedAt) / 1000.0
        log("Backend complete. Total elapsed: ${formatDouble1(elapsedSec)}s")

        return BenchmarkRunResult(
            scenarioId = id,
            target = request.target,
            startedAtEpochMillis = startedAt,
            finishedAtEpochMillis = finishedAt,
            modelReference = modelPathStr,
            resolvedModelPath = modelPathStr,
            modelResolutionSource = "cli",
            cases = results,
        )
    }
}

class NativeBenchmarkOrchestrator : BenchmarkRunner<BenchmarkRunRequest, BenchmarkRunResult> {
    private val scenario = NativeCpuThroughputScenario()

    override suspend fun run(config: BenchmarkRunRequest): BenchmarkRunResult {
        return scenario.execute(config)
    }

    fun listScenarios(): List<ResolvedBenchmarkScenario> = listOf(
        ResolvedBenchmarkScenario(scenario = scenario, supportedTargets = setOf("macos")),
    )
}

object NativeConsoleReporter {
    fun render(result: BenchmarkRunResult) {
        println("[BENCH] Scenario: ${result.scenarioId}")
        println("[BENCH] Target: ${result.target}")
        result.modelReference?.let { println("[BENCH] Model reference: $it") }
        result.resolvedModelPath?.let { println("[BENCH] Resolved model path: $it") }
        result.modelResolutionSource?.let { println("[BENCH] Model source: $it") }
        println("[BENCH] Started: ${result.startedAtEpochMillis}")
        println("[BENCH] Finished: ${result.finishedAtEpochMillis}")
        println()

        val stepGroups: Map<Int, List<BenchmarkCaseResult>> = result.cases
            .filter { it.steps != null }
            .groupBy { it.steps!! }
        for (steps in stepGroups.keys.sorted()) {
            val casesForStep = stepGroups[steps] ?: continue
            val promptGroups: Map<String?, List<BenchmarkCaseResult>> = casesForStep.groupBy { it.promptLabel }
            for (entry in promptGroups.entries) {
                val promptLabel = entry.key
                val promptCases: List<BenchmarkCaseResult> = entry.value
                val tokenCount = promptCases.firstOrNull()?.promptTokenCount
                println("[BENCH] steps=$steps, prompt=$promptLabel (${tokenCount ?: 0} tokens)")
                for (case in promptCases) {
                    val throughput = case.metrics.firstOrNull { m -> m.name == "throughput" }?.value
                    val suffix = if (throughput != null) "${formatDouble2(throughput)} tok/s" else case.status.name
                    println("  ${case.runtime ?: case.caseId}: $suffix")
                    for (note in case.notes) { println("  note: $note") }
                }
                println()
            }
        }

        val backends = availableNativeBackends()
        println("[BENCH] ========== SUMMARY ==========")
        println("[BENCH] | Backend      | Steps | Short  | Medium | Long   |")
        println("[BENCH] |--------------|-------|--------|--------|--------|")
        val allSteps = result.cases.mapNotNull { it.steps }.distinct().sorted()
        for (backend in backends) {
            for (steps in allSteps) {
                val columns = listOf("short", "medium", "long").map { prompt ->
                    val case = result.cases.find { c -> c.runtime == backend && c.promptLabel == prompt && c.steps == steps }
                    val throughput = case?.metrics?.firstOrNull { m -> m.name == "throughput" }?.value
                    if (throughput != null) formatDouble2(throughput) else (case?.status?.name ?: "N/A")
                }
                println("[BENCH] | ${backend.padEnd(12)} | ${steps.toString().padStart(5)} | ${columns[0].padStart(6)} | ${columns[1].padStart(6)} | ${columns[2].padStart(6)} |")
            }
        }
        println("[BENCH] ================================")
    }

    fun renderScenarioList(scenarios: List<ResolvedBenchmarkScenario>) {
        scenarios.forEach { resolved ->
            println("${resolved.scenario.id}\t${resolved.supportedTargets.joinToString()}\t${resolved.scenario.description}")
        }
    }
}
