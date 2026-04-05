package sk.ainet.performance.native

import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlin.time.measureTime
import sk.ainet.apps.kllama.CpuAttentionBackend
import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.apps.kllama.GpuAttentionBackend
import sk.ainet.apps.kllama.GpuTensorBridge
import sk.ainet.apps.kllama.LlamaIngestion
import sk.ainet.apps.kllama.LlamaLoadConfig
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import sk.ainet.models.llama.LlamaRuntime
import sk.ainet.models.llama.LlamaRuntimeWeights
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

// ── Expect declarations for macOS-specific backend creation ──

internal expect fun createMetalContext(): ExecutionContext?
internal expect fun createMlxContext(): ExecutionContext?
internal expect fun <T : DType> createGpuBridge(ctx: ExecutionContext): GpuTensorBridge<T>?
internal expect fun availableNativeBackends(): List<String>

// ── Data structures ──

internal data class NamedPrompt(
    val label: String,
    val text: String,
)

internal data class PromptPlan(
    val prompt: NamedPrompt,
    val promptTokens: IntArray,
)

// ── Adapter interface ──

internal interface NativeLlamaAdapter {
    val runtimeName: String

    suspend fun runAllCases(
        promptPlans: List<PromptPlan>,
        stepCounts: List<Int>,
        warmupRuns: Int,
        measuredRuns: Int,
    ): List<BenchmarkCaseResult>
}

// ── CPU adapter ──

internal class CpuNativeLlamaAdapter(
    private val modelPathStr: String,
) : NativeLlamaAdapter {
    override val runtimeName: String = "CPU"

    override suspend fun runAllCases(
        promptPlans: List<PromptPlan>,
        stepCounts: List<Int>,
        warmupRuns: Int,
        measuredRuns: Int,
    ): List<BenchmarkCaseResult> {
        val ctx = DirectCpuExecutionContext()
        log("  $runtimeName | loading model...")
        val weights = loadWeights<FP32>(ctx, FP32::class, modelPathStr)
        val backend = CpuAttentionBackend<FP32>(ctx, weights, FP32::class)
        @Suppress("DEPRECATION")
        val runtime = LlamaRuntime<FP32>(ctx, weights, backend, FP32::class)
        log("  $runtimeName | model loaded")

        return benchmarkCases(runtimeName, runtime, promptPlans, stepCounts, warmupRuns, measuredRuns)
    }
}

// ── GPU adapter (Metal or MLX) ──

internal class GpuNativeLlamaAdapter(
    private val modelPathStr: String,
    override val runtimeName: String,
    private val contextFactory: () -> ExecutionContext?,
) : NativeLlamaAdapter {

    override suspend fun runAllCases(
        promptPlans: List<PromptPlan>,
        stepCounts: List<Int>,
        warmupRuns: Int,
        measuredRuns: Int,
    ): List<BenchmarkCaseResult> {
        val ctx = try {
            contextFactory()
        } catch (e: Exception) {
            log("  $runtimeName | failed to create context: ${e.message}")
            null
        }

        if (ctx == null) {
            log("  $runtimeName | backend unavailable — skipping")
            return skipAll(promptPlans, stepCounts)
        }

        log("  $runtimeName | loading model...")
        val weights = try {
            loadWeights<FP32>(ctx, FP32::class, modelPathStr)
        } catch (e: Exception) {
            log("  $runtimeName | model load failed: ${e.message}")
            return skipAll(promptPlans, stepCounts, "Model load failed: ${e.message}")
        }

        val bridge = createGpuBridge<FP32>(ctx)
        val backend = if (bridge != null) {
            log("  $runtimeName | using GPU attention backend")
            GpuAttentionBackend<FP32>(ctx, bridge, weights, FP32::class)
        } else {
            log("  $runtimeName | GPU bridge unavailable, falling back to CPU attention")
            CpuAttentionBackend<FP32>(ctx, weights, FP32::class)
        }

        @Suppress("DEPRECATION")
        val runtime = LlamaRuntime<FP32>(ctx, weights, backend, FP32::class)
        log("  $runtimeName | model loaded")

        return benchmarkCases(runtimeName, runtime, promptPlans, stepCounts, warmupRuns, measuredRuns)
    }

    private fun skipAll(
        promptPlans: List<PromptPlan>,
        stepCounts: List<Int>,
        reason: String = "$runtimeName backend unavailable.",
    ): List<BenchmarkCaseResult> = stepCounts.flatMap { steps ->
        promptPlans.map { (prompt, promptTokens) ->
            BenchmarkCaseResult(
                caseId = "$runtimeName:${prompt.label}:$steps",
                status = BenchmarkCaseStatus.SKIPPED,
                runtime = runtimeName,
                promptLabel = prompt.label,
                promptTokenCount = promptTokens.size,
                steps = steps,
                metrics = emptyList(),
                notes = listOf(reason),
            )
        }
    }
}

// ── Shared helpers ──

internal suspend fun <T : DType> loadWeights(
    ctx: ExecutionContext,
    dtype: kotlin.reflect.KClass<T>,
    modelPathStr: String,
): LlamaRuntimeWeights<T> {
    val modelPath = Path(modelPathStr)
    val ingestion = LlamaIngestion<T>(
        ctx = ctx,
        dtype = dtype,
        config = LlamaLoadConfig(
            quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
            allowQuantized = false,
        ),
    )
    return ingestion.load {
        SystemFileSystem.source(modelPath).buffered()
    }
}

internal fun benchmarkCases(
    runtimeName: String,
    runtime: LlamaRuntime<FP32>,
    promptPlans: List<PromptPlan>,
    stepCounts: List<Int>,
    warmupRuns: Int,
    measuredRuns: Int,
): List<BenchmarkCaseResult> {
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

// ── Scenario ──

internal class NativeBackendThroughputScenario : BenchmarkScenario {
    override val id: String = "native-backend-throughput"
    override val description: String = "Compare CPU vs Metal vs MLX backend throughput on native macOS."

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

        val adapters: List<NativeLlamaAdapter> = buildList {
            add(CpuNativeLlamaAdapter(modelPathStr))
            add(GpuNativeLlamaAdapter(modelPathStr, "Metal", ::createMetalContext))
            add(GpuNativeLlamaAdapter(modelPathStr, "MLX", ::createMlxContext))
        }

        val results = mutableListOf<BenchmarkCaseResult>()
        for ((index, adapter) in adapters.withIndex()) {
            log("=== Backend ${index + 1}/${adapters.size}: ${adapter.runtimeName} ===")
            val adapterResults = adapter.runAllCases(
                promptPlans = promptPlans,
                stepCounts = request.steps,
                warmupRuns = request.warmupRuns,
                measuredRuns = request.measuredRuns,
            )
            results += adapterResults
            val successCount = adapterResults.count { it.status == BenchmarkCaseStatus.SUCCESS }
            log("${adapter.runtimeName} finished: $successCount/${adapterResults.size} cases succeeded")
        }

        val finishedAt = epochMillis()
        val elapsedSec = (finishedAt - startedAt) / 1000.0
        log("All backends complete. Total elapsed: ${formatDouble1(elapsedSec)}s")

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

// ── Orchestrator ──

class NativeBenchmarkOrchestrator : BenchmarkRunner<BenchmarkRunRequest, BenchmarkRunResult> {
    private val scenario = NativeBackendThroughputScenario()

    override suspend fun run(config: BenchmarkRunRequest): BenchmarkRunResult {
        return scenario.execute(config)
    }

    fun listScenarios(): List<ResolvedBenchmarkScenario> = listOf(
        ResolvedBenchmarkScenario(scenario = scenario, supportedTargets = setOf("macos")),
    )
}

// ── Console reporter (matches JVM format) ──

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
