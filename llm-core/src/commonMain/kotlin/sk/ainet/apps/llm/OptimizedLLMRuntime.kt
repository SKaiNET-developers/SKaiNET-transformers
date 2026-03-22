package sk.ainet.apps.llm

import kotlin.random.Random
import kotlin.reflect.KClass
import sk.ainet.context.ExecutionContext
import sk.ainet.io.weights.MappingConfig
import sk.ainet.io.weights.WeightMapper
import sk.ainet.io.weights.WeightNameResolver
import sk.ainet.io.weights.WeightTensor
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.exec.ComputeGraphExecutor
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.types.DType

/**
 * Unified, optimized LLM runtime that replaces per-architecture runtimes
 * (`LlamaRuntime`, `BertRuntime`, `ApertusRuntime`).
 *
 * The runtime supports two execution modes from the same `network {}` definition:
 *
 * 1. **Direct mode** (development/debugging): The `Module<T,V>` tree executes forward
 *    passes directly. No compilation step needed.
 *
 * 2. **Optimized mode** (production): The module is traced into a DAG, optimization
 *    passes (transpose elimination, weight dedup, RMSNorm/SwiGLU/QKV fusion, DCE) are
 *    applied, and the resulting graph is executed with fused kernels.
 *
 * Adding a new model family (Gemma, Qwen, etc.) requires only:
 * - A `network {}` function (e.g., `gemmaNetwork(config)`)
 * - A config parser
 * No hand-coded runtime, custom weight loader, or custom tensor names.
 *
 * @param T The data type for model weights (FP32, FP16, Q8_0, etc.)
 * @param model The `Module<T, V>` tree built from a `network {}` definition
 * @param ctx Execution context providing tensor operations
 * @param mode Execution mode: [OptimizedLLMMode.DIRECT] or [OptimizedLLMMode.OPTIMIZED]
 * @param dtype KClass for the DType (needed to create tensors)
 * @param bos Beginning-of-sequence token ID
 * @param random Random generator for sampling
 */
public class OptimizedLLMRuntime<T : DType>(
    private val model: Module<T, Float>,
    private val ctx: ExecutionContext,
    private val mode: OptimizedLLMMode = OptimizedLLMMode.DIRECT,
    private val dtype: KClass<T>,
    public val bos: Int = 1,
    private val random: Random = Random.Default
) : InferenceRuntime<T> {

    public val dim: Int get() = modelDim
    public val vocabSize: Int get() = modelVocabSize
    public val seqLen: Int get() = modelSeqLen
    public val nLayers: Int get() = modelNLayers

    /** Current position in the sequence (incremented on each forward call). */
    public var position: Int = 0
        private set

    // Extracted from the model's module tree structure
    private val modelDim: Int
    private val modelVocabSize: Int
    private val modelSeqLen: Int
    private val modelNLayers: Int

    // Optimized graph (only populated in OPTIMIZED mode after compile())
    private var optimizedGraph: ComputeGraph? = null
    private var graphExecutor: ComputeGraphExecutor? = null
    // Maps graph input node IDs to weight tensors for execution-time resolution
    private var weightTensorMap: Map<String, Tensor<T, Float>> = emptyMap()
    // The graph node ID that receives the dynamic token input (vs static weights)
    private var inputNodeId: String? = null

    init {
        val info = extractModelInfo(model)
        modelDim = info.dim
        modelVocabSize = info.vocabSize
        modelSeqLen = info.seqLen
        modelNLayers = info.nLayers
    }

    // ---- InferenceRuntime implementation ----

    /**
     * Single-token forward pass.
     *
     * In DIRECT mode, runs the module tree's forward pass.
     * In OPTIMIZED mode, runs the compiled graph with fused kernels.
     */
    override fun forward(tokenId: Int): Tensor<T, Float> {
        require(position < seqLen) { "Context length exceeded: pos=$position seqLen=$seqLen" }

        val logits = when (mode) {
            OptimizedLLMMode.DIRECT, OptimizedLLMMode.HYBRID -> {
                // HYBRID uses the same model.forward() path as DIRECT, but
                // HybridTransformerBlock instances internally dispatch to
                // compiled subgraphs for compute-heavy operations.
                val input = createTokenTensor(tokenId)
                model.forward(input, ctx)
            }
            OptimizedLLMMode.OPTIMIZED -> {
                executeOptimized(tokenId)
            }
        }

        position++
        return logits
    }

    /** Reset to initial state (clear KV caches, rewind position to 0). */
    override fun reset() {
        resetModuleState(model)
        position = 0
    }

    // ---- Convenience generation ----

    /**
     * Auto-regressive generation loop.
     *
     * Delegates to the standalone [generate] extension on [InferenceRuntime].
     */
    public fun generate(
        prompt: IntArray,
        steps: Int,
        temperature: Float,
        onToken: (Int) -> Unit
    ) {
        (this as InferenceRuntime<T>).generate(
            prompt = prompt,
            steps = steps,
            temperature = temperature,
            bosToken = bos,
            random = random,
            onToken = onToken
        )
    }

    // ---- Compilation / Optimization ----

    /**
     * Test each optimization pass individually and report which ones introduce divergence.
     * Returns a list of (passName, maxDiff) pairs for diagnostic purposes.
     */
    public fun compileDiagnostic(dummyTokenId: Int = bos): List<Pair<String, Float>> {
        // First get the DIRECT reference
        val directInput = createTokenTensor(dummyTokenId)
        val directLogits = model.forward(directInput, ctx).data.copyToFloatArray()
        // Reset state after DIRECT forward
        resetModuleState(model)

        val results = mutableListOf<Pair<String, Float>>()
        for ((name, pipeline) in getLLMOptimizationPassesForDebug()) {
            // Re-trace for each pass (tracing mutates model state)
            resetModuleState(model)
            compileWith(dummyTokenId, pipeline)
            val optimizedLogits = executeOptimized(dummyTokenId).data.copyToFloatArray()
            var maxDiff = 0f
            for (i in directLogits.indices) {
                val d = kotlin.math.abs(directLogits[i] - optimizedLogits[i])
                if (d > maxDiff) maxDiff = d
            }
            results.add(name to maxDiff)
            // Reset for next iteration
            position = 0
        }
        // Leave runtime in uncompiled state
        optimizedGraph = null
        graphExecutor = null
        weightTensorMap = emptyMap()
        inputNodeId = null
        position = 0
        resetModuleState(model)
        return results
    }

    /**
     * Compile the model for optimized execution.
     *
     * Creates a tracing [DefaultGraphExecutionContext], runs a forward pass through the
     * module tree to capture the computation graph, applies the LLM optimization pipeline,
     * and prepares a [ComputeGraphExecutor] for subsequent forward passes.
     *
     * @param dummyTokenId A token ID to use for the tracing forward pass
     * @return Diagnostics from the optimization passes
     */
    public fun compile(dummyTokenId: Int = bos): List<String> {
        return if (mode == OptimizedLLMMode.HYBRID) {
            compileHybrid()
        } else {
            compileWith(dummyTokenId, getLLMOptimizationPipeline())
        }
    }

    /**
     * Compile all [HybridTransformerBlock] instances in the module tree.
     * Each block compiles its own per-layer subgraphs (attn_compute, ffn_compute).
     */
    private fun compileHybrid(): List<String> {
        val diagnostics = mutableListOf<String>()
        val pipeline = getLLMOptimizationPipeline()
        val blocks = findHybridBlocks(model)
        diagnostics.add("Found ${blocks.size} HybridTransformerBlock(s)")
        for (block in blocks) {
            diagnostics.addAll(block.compile(ctx, dtype, pipeline))
        }
        return diagnostics
    }

    private fun findHybridBlocks(module: Module<*, *>): List<HybridTransformerBlock<T, Float>> {
        val result = mutableListOf<HybridTransformerBlock<T, Float>>()
        val queue = mutableListOf<Module<*, *>>(module)
        var i = 0
        while (i < queue.size) {
            val m = queue[i++]
            @Suppress("UNCHECKED_CAST")
            if (m is HybridTransformerBlock<*, *>) {
                result.add(m as HybridTransformerBlock<T, Float>)
            }
            queue.addAll(m.modules)
        }
        return result
    }

    @PublishedApi
    internal fun compileWith(
        dummyTokenId: Int = bos,
        pipeline: sk.ainet.compile.opt.GraphOptimizationPipeline
    ): List<String> {
        val diagnostics = mutableListOf<String>()

        // 1. Create a tape-recording context that traces operations with
        //    full tensor reference tracking (needed to wire edges correctly)
        val tapingCtx = DefaultGraphExecutionContext.tape(baseOps = ctx.ops)

        // 2. Record the forward pass
        tapingCtx.startRecording()
        val input = createTokenTensor(dummyTokenId)
        model.forward(input, tapingCtx)
        val tape = tapingCtx.stopRecording()

        // 3. Convert tape to ComputeGraph.
        //    We save the session for weight resolution at execution time, then
        //    clear it before graph conversion to avoid embedding weight float arrays
        //    (which would OOM for large models). External inputs become lightweight
        //    placeholder nodes instead.
        val tapeObj = tape as? DefaultExecutionTape
            ?: error("Expected DefaultExecutionTape but got ${tape?.let { it::class.simpleName }}")

        // Keep the original session for graph building (needed for correct edge wiring
        // of ops with List<Tensor> params like concat). Use embedConstants=false to avoid
        // OOM from embedding weight float arrays into graph nodes.
        val tracingSession = tapeObj.session

        val rawGraph = tapeObj.toComputeGraph(
            synthesizeExternalInputs = true,
            inputTensorIds = emptySet(),
            embedConstants = false
        )

        diagnostics.add("Traced graph: ${rawGraph.nodes.size} nodes, ${rawGraph.edges.size} edges")

        // 4. Validate the raw traced graph before optimization
        try {
            rawGraph.getTopologicalOrder()
        } catch (e: IllegalStateException) {
            diagnostics.add("WARNING: Raw traced graph has cycles — skipping optimization passes")
            optimizedGraph = rawGraph
            graphExecutor = ComputeGraphExecutor(rawGraph, ctx.ops)
            val (wMap, inNodeId) = buildWeightTensorMap(tracingSession, rawGraph, input)
            weightTensorMap = wMap
            inputNodeId = inNodeId
            diagnostics.add("Input node: $inNodeId, weight map: ${wMap.size} entries")
            return diagnostics
        }

        // 5. Apply the LLM optimization pipeline
        val result = pipeline.optimize(rawGraph)
        optimizedGraph = result.graph
        diagnostics.addAll(result.passResults.flatMap { it.diagnostics })
        val optimizedGraphForExec = result.graph

        // 6. Create executor for the optimized graph
        graphExecutor = ComputeGraphExecutor(optimizedGraphForExec, ctx.ops)

        // 7. Build the weight tensor map for execution-time resolution
        val (wMap, inNodeId) = buildWeightTensorMap(tracingSession, optimizedGraphForExec, input)
        weightTensorMap = wMap
        inputNodeId = inNodeId
        diagnostics.add("Input node: $inNodeId, weight map: ${wMap.size} entries")

        return diagnostics
    }

    /**
     * Alternative compilation path that skips optimization passes.
     *
     * Traces the forward pass into a ComputeGraph but only applies the graph
     * executor without any optimization. Useful for debugging when optimization
     * passes cause issues.
     *
     * @param dummyTokenId A token ID to use for the tracing forward pass
     * @return Diagnostics
     */
    public fun compileUnoptimized(dummyTokenId: Int = bos): List<String> {
        val diagnostics = mutableListOf<String>()

        val tapingCtx = DefaultGraphExecutionContext.tape(baseOps = ctx.ops)
        tapingCtx.startRecording()
        val input = createTokenTensor(dummyTokenId)
        model.forward(input, tapingCtx)
        val tape = tapingCtx.stopRecording()

        val tapeObj = tape as? DefaultExecutionTape
            ?: error("Expected DefaultExecutionTape but got ${tape?.let { it::class.simpleName }}")

        val tracingSession = tapeObj.session

        val rawGraph = tapeObj.toComputeGraph(
            synthesizeExternalInputs = true,
            inputTensorIds = emptySet(),
            embedConstants = false
        )

        diagnostics.add("Traced graph (unoptimized): ${rawGraph.nodes.size} nodes, ${rawGraph.edges.size} edges")

        optimizedGraph = rawGraph
        graphExecutor = ComputeGraphExecutor(rawGraph, ctx.ops)
        val (wMap, inNodeId) = buildWeightTensorMap(tracingSession, rawGraph, input)
        weightTensorMap = wMap
        inputNodeId = inNodeId
        diagnostics.add("Input node: $inNodeId, weight map: ${wMap.size} entries")

        return diagnostics
    }

    @Suppress("UNCHECKED_CAST")
    private fun executeOptimized(tokenId: Int): Tensor<T, Float> {
        val executor = graphExecutor
            ?: error("Model not compiled. Call compile() before using OPTIMIZED mode.")
        val nodeId = inputNodeId
            ?: error("Input node ID not resolved during compilation. Graph may be missing the token input.")

        val input = createTokenTensor(tokenId)

        // Build input map: token input (at its exact node ID) + all weight tensors
        val inputMap = mutableMapOf<String, Tensor<DType, Float>>()
        inputMap[nodeId] = input as Tensor<DType, Float>
        for ((wNodeId, tensor) in weightTensorMap) {
            inputMap[wNodeId] = tensor as Tensor<DType, Float>
        }

        val results = executor.execute(inputMap)

        // The last output is the logits tensor
        return results.values.lastOrNull() as? Tensor<T, Float>
            ?: error("Graph execution produced no outputs")
    }

    // ---- Weight Loading ----

    /**
     * Load weights from model file tensors using a name resolver.
     *
     * This is the Module-based path (works for both DIRECT and pre-compile OPTIMIZED mode).
     */
    public fun loadWeights(
        tensors: List<WeightTensor<T, Float>>,
        resolver: WeightNameResolver
    ): WeightMapper.MappingResult {
        return WeightMapper.applyWeights(
            model, tensors,
            MappingConfig(
                usePathBasedMatching = true,
                fallbackToShapeMatching = true,
                nameResolver = resolver
            )
        )
    }

    // ---- Internals ----

    /**
     * Build a map from graph input node IDs to actual weight tensors.
     *
     * During compilation, unresolved tensor refs become input placeholder nodes.
     * This method resolves those placeholders back to the actual weight tensors
     * from the tracing session, so they can be provided at execution time.
     *
     * The [tracingInputTensor] (the token tensor used during tracing) is excluded
     * from the weight map and its node ID is returned separately so that
     * [executeOptimized] can feed the dynamic token input to the correct node.
     *
     * @return Pair of (weightMap, inputNodeId) where inputNodeId is the graph node
     *   that should receive the dynamic token input at execution time.
     */
    @Suppress("UNCHECKED_CAST")
    private fun buildWeightTensorMap(
        session: sk.ainet.lang.trace.TraceSession,
        graph: ComputeGraph,
        tracingInputTensor: Tensor<T, Float>
    ): Pair<Map<String, Tensor<T, Float>>, String?> {
        val result = mutableMapOf<String, Tensor<T, Float>>()
        var detectedInputNodeId: String? = null

        // Count input/weight nodes vs total for diagnostics
        val inputWeightNodes = graph.nodes.filter {
            it.operationName in setOf("input", "weight", "parameter", "constant")
        }

        for (node in inputWeightNodes) {
            val outputSpec = node.outputs.firstOrNull() ?: continue
            val tensorId = outputSpec.name
            val tensor = session.resolve(tensorId)
            // Identify the token input node: it has a small shape (typically [1] for a
            // single token ID) while weight tensors are much larger. The embedding module
            // converts the FP32 token tensor to Int32 internally, so we can't match by
            // object identity with the original input — we match by shape instead.
            val resolvedShape = tensor?.shape?.dimensions
            if (resolvedShape != null && resolvedShape.contentEquals(tracingInputTensor.shape.dimensions)
                && (tensor?.volume ?: 0) <= 1) {
                detectedInputNodeId = node.id
                continue // Don't add the dynamic input to the static weight map
            }
            @Suppress("UNCHECKED_CAST")
            val typedTensor = tensor as? Tensor<T, Float> ?: continue
            result[node.id] = typedTensor
        }
        return result to detectedInputNodeId
    }

    private fun createTokenTensor(tokenId: Int): Tensor<T, Float> {
        val shape = Shape(intArrayOf(1))
        val data = DenseFloatArrayTensorData<T>(shape, floatArrayOf(tokenId.toFloat()))
        return VoidOpsTensor(data = data, dtype = dtype)
    }

    private fun resetModuleState(module: Module<*, *>) {
        for (child in module.modules) {
            resetModuleState(child)
        }
        if (module is sk.ainet.lang.nn.transformer.KVCache<*, *>) {
            module.reset()
        }
    }

    private fun extractModelInfo(module: Module<*, *>): ModelInfo {
        var dim = 0
        var vocabSize = 0
        var seqLen = 4096
        var nLayers = 0

        val modules = mutableListOf<Module<*, *>>()
        modules.add(module)
        var i = 0
        while (i < modules.size) {
            val m = modules[i++]

            if (m is sk.ainet.lang.nn.layers.EmbeddingAdapter<*, *>) {
                if (vocabSize == 0) {
                    vocabSize = m.numEmbeddings
                    dim = m.embeddingDim
                }
            } else if (m is sk.ainet.lang.nn.transformer.KVCache<*, *>) {
                seqLen = m.maxSeqLen
            }

            modules.addAll(m.modules)
        }

        for (m in module.modules) {
            val name = m.name ?: ""
            if (name.startsWith("blk.") || name.contains("layer")) {
                nLayers++
            }
        }

        return ModelInfo(dim, vocabSize, seqLen, nLayers)
    }
}
