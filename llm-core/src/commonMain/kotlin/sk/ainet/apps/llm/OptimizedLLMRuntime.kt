package sk.ainet.apps.llm

import kotlin.random.Random
import kotlin.reflect.KClass
import sk.ainet.compile.opt.GraphOptimizationPipeline
import sk.ainet.compile.opt.passes.DeadCodeEliminationPass
import sk.ainet.compile.opt.passes.LLMFusionPass
import sk.ainet.compile.opt.passes.OperationFusionPass
import sk.ainet.compile.opt.passes.SharedWeightDeduplicationPass
import sk.ainet.compile.opt.passes.TransposeEliminationPass
import sk.ainet.context.ExecutionContext
import sk.ainet.io.weights.MappingConfig
import sk.ainet.io.weights.WeightMapper
import sk.ainet.io.weights.WeightNameResolver
import sk.ainet.io.weights.WeightTensor
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.exec.ComputeGraphExecutor
import sk.ainet.lang.graph.exec.LLMFusedOpHandlers
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.types.DType

/**
 * Unified, optimized LLM runtime that replaces per-architecture runtimes
 * (`LlamaRuntime`, `BertRuntime`, `ApertusRuntime`).
 *
 * The runtime supports two execution modes from the same `network {}` definition:
 *
 * 1. **Direct mode** (development/debugging): The `Module<T,V>` tree executes forward
 *    passes directly, identical to how CNNs work today. No compilation step needed.
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
 * @param mode Execution mode: [Mode.DIRECT] or [Mode.OPTIMIZED]
 * @param dtype KClass for the DType (needed to create tensors)
 * @param bos Beginning-of-sequence token ID
 * @param random Random generator for sampling
 */
public class OptimizedLLMRuntime<T : DType>(
    private val model: Module<T, Float>,
    private val ctx: ExecutionContext,
    private val mode: Mode = Mode.DIRECT,
    private val dtype: KClass<T>,
    private val bos: Int = 1,
    random: Random = Random.Default
) : DecoderRuntime<T>(random) {

    /**
     * Execution mode.
     */
    public enum class Mode {
        /** Direct execution via Module.forward() — no compilation, good for debugging. */
        DIRECT,
        /** Traced + optimized execution via fused graph kernels — production path. */
        OPTIMIZED
    }

    override val dim: Int get() = modelDim
    override val vocabSize: Int get() = modelVocabSize
    override val seqLen: Int get() = modelSeqLen
    override val nLayers: Int get() = modelNLayers
    override val bosToken: Int get() = bos

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

    // ---- DecoderRuntime template methods ----

    override fun embedToken(tokenId: Int): Tensor<T, Float> {
        val inputTensor = createTokenTensor(tokenId)
        return model.forward(inputTensor, ctx)
    }

    override fun runLayer(layerIdx: Int, x: Tensor<T, Float>): Tensor<T, Float> {
        // The full forward pass runs through the module tree in embedToken.
        // This method exists only for DecoderRuntime compatibility.
        return x
    }

    override fun outputNorm(x: Tensor<T, Float>): Tensor<T, Float> = x

    override fun outputProject(x: Tensor<T, Float>): Tensor<T, Float> = x

    override fun resetState() {
        resetModuleState(model)
    }

    /**
     * Single-token forward pass.
     *
     * In DIRECT mode, runs the module tree's forward pass.
     * In OPTIMIZED mode, runs the compiled graph with fused kernels.
     */
    override fun forward(tokenId: Int): Tensor<T, Float> {
        require(position < seqLen) { "Context length exceeded: pos=$position seqLen=$seqLen" }

        val logits = when (mode) {
            Mode.DIRECT -> {
                val input = createTokenTensor(tokenId)
                model.forward(input, ctx)
            }
            Mode.OPTIMIZED -> {
                executeOptimized(tokenId)
            }
        }

        position++
        return logits
    }

    // ---- Compilation / Optimization ----

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
        val result = LLM_PIPELINE.optimize(rawGraph)
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
        for (node in graph.nodes) {
            if (node.operationName != "input" && node.operationName != "weight") continue
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

    private data class ModelInfo(
        val dim: Int,
        val vocabSize: Int,
        val seqLen: Int,
        val nLayers: Int
    )

    private fun extractModelInfo(module: Module<*, *>): ModelInfo {
        var dim = 0
        var vocabSize = 0
        var seqLen = 4096
        var nLayers = 0

        fun walk(m: Module<*, *>) {
            when (m) {
                is sk.ainet.lang.nn.layers.EmbeddingAdapter<*, *> -> {
                    if (vocabSize == 0) {
                        vocabSize = m.numEmbeddings
                        dim = m.embeddingDim
                    }
                }
                is sk.ainet.lang.nn.transformer.KVCache<*, *> -> {
                    seqLen = m.maxSeqLen
                }
            }
            for (child in m.modules) {
                walk(child)
            }
        }

        walk(module)

        for (child in module.modules) {
            if (child.name.startsWith("blk.") || child.name.matches(Regex("encoder\\.layer\\.\\d+"))) {
                nLayers++
            }
        }

        return ModelInfo(dim, vocabSize, seqLen, nLayers)
    }

    public companion object {
        /**
         * The standard LLM optimization pipeline.
         *
         * Pass ordering:
         * 1. TransposeElimination — fold transposes into matmuls before fusion sees them
         * 2. SharedWeightDedup — deduplicate tied weights before pattern matching
         * 3. LLMFusion — fuse RMSNorm, SwiGLU, QKV patterns on clean graph
         * 4. OperationFusion — fuse remaining elementwise chains
         * 5. DeadCodeElimination — clean up nodes orphaned by fusion passes
         */
        init {
            // Register CPU fallback handlers for fused ops so the executor
            // can run them on any backend. Platform-specific handlers (Metal, CUDA)
            // override these when available.
            LLMFusedOpHandlers.registerAll()
        }

        public val LLM_PIPELINE: GraphOptimizationPipeline = GraphOptimizationPipeline(
            passes = listOf(
                TransposeEliminationPass(),
                SharedWeightDeduplicationPass(),
                LLMFusionPass(),
                OperationFusionPass(),
                DeadCodeEliminationPass()
            ),
            maxIterations = 2
        )

        /**
         * Create an optimized runtime from a network definition and weights.
         *
         * This is the primary factory method for production use.
         *
         * @param model Module tree from a `network {}` definition (e.g., `llamaNetwork(config)`)
         * @param tensors Weight tensors from GGUF/SafeTensors
         * @param resolver Weight name resolver for the model format
         * @param ctx Execution context
         * @param dtype KClass for the DType
         * @param optimized Whether to compile for optimized execution
         * @param bosToken BOS token ID
         * @return A ready-to-use runtime
         */
        public fun <T : DType> create(
            model: Module<T, Float>,
            tensors: List<WeightTensor<T, Float>>,
            resolver: WeightNameResolver,
            ctx: ExecutionContext,
            dtype: KClass<T>,
            optimized: Boolean = true,
            bosToken: Int = 1
        ): OptimizedLLMRuntime<T> {
            val mode = if (optimized) Mode.OPTIMIZED else Mode.DIRECT
            val runtime = OptimizedLLMRuntime(model, ctx, mode, dtype, bosToken)

            // Load weights into the module tree (needed for both modes —
            // OPTIMIZED mode traces with real weights for accurate graph capture)
            val result = runtime.loadWeights(tensors, resolver)
            WeightMapper.validateAllParametersMapped(result)

            if (optimized) {
                runtime.compile(bosToken)
            }

            return runtime
        }
    }
}
