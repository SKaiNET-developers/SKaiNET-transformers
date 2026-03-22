package sk.ainet.apps.llm

import kotlin.reflect.KClass
import sk.ainet.apps.llm.compile.LLMFusionPass
import sk.ainet.apps.llm.graph.LLMFusedOpHandlers
import sk.ainet.compile.opt.GraphOptimizationPipeline
import sk.ainet.compile.opt.passes.DeadCodeEliminationPass
import sk.ainet.compile.opt.passes.OperationFusionPass
import sk.ainet.compile.opt.passes.SharedWeightDeduplicationPass
import sk.ainet.compile.opt.passes.TransposeEliminationPass
import sk.ainet.context.ExecutionContext
import sk.ainet.io.weights.WeightMapper
import sk.ainet.io.weights.WeightNameResolver
import sk.ainet.io.weights.WeightTensor
import sk.ainet.lang.nn.Module
import sk.ainet.lang.types.DType

/**
 * Execution mode for [OptimizedLLMRuntime].
 */
public enum class OptimizedLLMMode {
    /** Direct execution via Module.forward() — no compilation, good for debugging. */
    DIRECT,
    /** Traced + optimized execution via fused graph kernels — production path. */
    OPTIMIZED
}

internal data class ModelInfo(
    val dim: Int,
    val vocabSize: Int,
    val seqLen: Int,
    val nLayers: Int
)

/**
 * Standard LLM optimization settings and registry.
 */
public fun getLLMOptimizationPipeline(): GraphOptimizationPipeline {
    // Register CPU fallback handlers for fused ops so the executor
    // can run them on any backend. Platform-specific handlers (Metal, CUDA)
    // override these when available.
    LLMFusedOpHandlers.registerAll()

    return GraphOptimizationPipeline(
        passes = listOf(
            TransposeEliminationPass(),
            SharedWeightDeduplicationPass(),
            LLMFusionPass(),
            // OperationFusionPass skipped: its fallback decomposition in the graph
            // executor misroutes inputs between sub-ops (skainet issue).
            // LLM-specific fusions cover the performance-critical patterns.
            DeadCodeEliminationPass()
        ),
        maxIterations = 2
    )
}

/**
 * Returns individual pass pipelines for debugging which pass introduces divergence.
 * Each entry is (name, pipeline-with-just-that-pass).
 */
public fun getLLMOptimizationPassesForDebug(): List<Pair<String, GraphOptimizationPipeline>> {
    LLMFusedOpHandlers.registerAll()
    return listOf(
        "TransposeElimination" to GraphOptimizationPipeline(listOf(TransposeEliminationPass()), 1),
        "SharedWeightDedup" to GraphOptimizationPipeline(listOf(SharedWeightDeduplicationPass()), 1),
        "LLMFusion" to GraphOptimizationPipeline(listOf(LLMFusionPass()), 1),
        "OperationFusion" to GraphOptimizationPipeline(listOf(OperationFusionPass()), 1),
        "DeadCodeElimination" to GraphOptimizationPipeline(listOf(DeadCodeEliminationPass()), 1),
        "TransposeElim+LLMFusion" to GraphOptimizationPipeline(
            listOf(TransposeEliminationPass(), LLMFusionPass()), 1
        )
    )
}

/**
 * Primary factory method to create an optimized runtime.
 */
public fun <T : DType> createOptimizedLLMRuntime(
    model: Module<T, Float>,
    tensors: List<WeightTensor<T, Float>>,
    resolver: WeightNameResolver,
    ctx: ExecutionContext,
    dtype: KClass<T>,
    optimized: Boolean = true,
    bosToken: Int = 1
): OptimizedLLMRuntime<T> {
    val mode = if (optimized) OptimizedLLMMode.OPTIMIZED else OptimizedLLMMode.DIRECT
    val runtime = OptimizedLLMRuntime(model, ctx, mode, dtype, bosToken)

    // Load weights into the module tree (needed for both modes)
    val result = runtime.loadWeights(tensors, resolver)
    WeightMapper.validateAllParametersMapped(result)

    if (optimized) {
        runtime.compile(bosToken)
    }

    return runtime
}
