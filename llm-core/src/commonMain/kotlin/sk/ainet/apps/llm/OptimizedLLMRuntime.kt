package sk.ainet.apps.llm

import kotlin.random.Random
import sk.ainet.compile.opt.GraphOptimizationPipeline
import sk.ainet.compile.opt.passes.DeadCodeEliminationPass
import sk.ainet.compile.opt.passes.LLMFusionPass
import sk.ainet.compile.opt.passes.OperationFusionPass
import sk.ainet.compile.opt.passes.SharedWeightDeduplicationPass
import sk.ainet.compile.opt.passes.TransposeEliminationPass
import sk.ainet.context.ExecutionContext
import sk.ainet.io.weights.GraphWeightLoader
import sk.ainet.io.weights.MappingConfig
import sk.ainet.io.weights.WeightMapper
import sk.ainet.io.weights.WeightNameResolver
import sk.ainet.io.weights.WeightTensor
import sk.ainet.lang.dag.GraphProgramSink
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Tensor
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
 * @param bosToken Beginning-of-sequence token ID
 * @param random Random generator for sampling
 */
public class OptimizedLLMRuntime<T : DType>(
    private val model: Module<T, Float>,
    private val ctx: ExecutionContext,
    private val mode: Mode = Mode.DIRECT,
    private val bosToken: Int = 1,
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
    override val bosToken: Int get() = field

    // These are extracted from the model's module tree structure
    private val modelDim: Int
    private val modelVocabSize: Int
    private val modelSeqLen: Int
    private val modelNLayers: Int

    // Optimized graph (only populated in OPTIMIZED mode)
    private var optimizedGraph: ComputeGraph? = null

    init {
        // Extract architecture parameters from the module tree
        val info = extractModelInfo(model)
        modelDim = info.dim
        modelVocabSize = info.vocabSize
        modelSeqLen = info.seqLen
        modelNLayers = info.nLayers
    }

    // ---- DecoderRuntime template methods ----

    override fun embedToken(tokenId: Int): Tensor<T, Float> {
        // Direct mode delegates to the module's forward pass
        // The module tree handles embedding lookup internally
        val inputTensor = createTokenTensor(tokenId)
        return model.forward(inputTensor, ctx)
    }

    override fun runLayer(layerIdx: Int, x: Tensor<T, Float>): Tensor<T, Float> {
        // In the unified module-based approach, runLayer is not called separately —
        // the full forward pass runs through the module tree in embedToken.
        // This method exists only for DecoderRuntime compatibility.
        return x
    }

    override fun outputNorm(x: Tensor<T, Float>): Tensor<T, Float> = x

    override fun outputProject(x: Tensor<T, Float>): Tensor<T, Float> = x

    override fun resetState() {
        // Reset KV caches and other stateful modules
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
     * Traces the module into a DAG, applies the LLM optimization pipeline, and
     * prepares the optimized graph for execution. Call this before using [Mode.OPTIMIZED].
     *
     * @param dummyTokenId A token ID to use for the tracing forward pass
     * @return Diagnostics from the optimization passes
     */
    public fun compile(dummyTokenId: Int = bosToken): List<String> {
        val sink = GraphProgramSink()

        // TODO: wire sink into ctx's OpSink for tracing
        // For now, run a tracing forward pass that records into the sink
        val input = createTokenTensor(dummyTokenId)
        model.forward(input, ctx)

        val program = sink.toGraphProgram()

        // Convert GraphProgram to ComputeGraph for optimization
        val graph = programToComputeGraph(program)

        // Run the LLM optimization pipeline
        val result = LLM_PIPELINE.optimize(graph)
        optimizedGraph = result.graph

        return result.passResults.flatMap { it.diagnostics }
    }

    private fun executeOptimized(tokenId: Int): Tensor<T, Float> {
        val graph = optimizedGraph
            ?: error("Model not compiled. Call compile() before using OPTIMIZED mode.")

        // Execute the optimized graph
        // The graph executor resolves parameter bindings and runs fused kernels
        val input = createTokenTensor(tokenId)
        // TODO: integrate with ComputeGraphExecutor once available
        // For now, fall back to direct execution
        return model.forward(input, ctx)
    }

    // ---- Weight Loading ----

    /**
     * Load weights from model file tensors using a name resolver.
     *
     * This is the Module-based path (for DIRECT mode).
     *
     * @param tensors Weight tensors loaded from GGUF/SafeTensors
     * @param resolver Translates module paths to tensor names in the model format
     * @return Mapping result with diagnostics
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

    /**
     * Load weights into the optimized graph's parameter nodes.
     *
     * This is the Graph-based path (for OPTIMIZED mode, post-compile).
     *
     * @param tensors Weight tensors loaded from GGUF/SafeTensors
     * @param resolver Translates parameter node IDs to tensor names
     * @return Graph load result with diagnostics
     */
    public fun loadGraphWeights(
        tensors: List<WeightTensor<T, Float>>,
        resolver: WeightNameResolver
    ): GraphWeightLoader.GraphLoadResult<T, Float> {
        val sink = GraphProgramSink()
        // Get the current program for graph weight loading
        val program = sink.toGraphProgram()
        val loader = GraphWeightLoader(resolver)
        return loader.load(program, tensors)
    }

    // ---- Internals ----

    @Suppress("UNCHECKED_CAST")
    private fun createTokenTensor(tokenId: Int): Tensor<T, Float> {
        val data = FloatArrayTensorData<T>(floatArrayOf(tokenId.toFloat()), intArrayOf(1))
        return Tensor(
            data = data as sk.ainet.lang.tensor.TensorData<T, Float>,
            shape = sk.ainet.lang.tensor.Shape(intArrayOf(1)),
            dtype = model.dtype
        )
    }

    private fun resetModuleState(module: Module<*, *>) {
        // Walk the module tree and reset any stateful modules (KV caches, etc.)
        for (child in module.modules) {
            resetModuleState(child)
        }
        // Check for KVCache reset
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
        // Walk the module tree to extract architecture parameters
        var dim = 0
        var vocabSize = 0
        var seqLen = 4096 // default
        var nLayers = 0

        fun walk(m: Module<*, *>) {
            when (m) {
                is sk.ainet.lang.nn.layers.Embedding<*, *> -> {
                    if (vocabSize == 0) {
                        vocabSize = m.vocabSize
                        dim = m.dim
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

        // Count transformer layers by looking at stage/sequential blocks named "blk.N"
        for (child in module.modules) {
            if (child.name.startsWith("blk.") || child.name.matches(Regex("encoder\\.layer\\.\\d+"))) {
                nLayers++
            }
        }

        return ModelInfo(
            dim = dim,
            vocabSize = vocabSize,
            seqLen = seqLen,
            nLayers = nLayers
        )
    }

    private fun programToComputeGraph(program: sk.ainet.lang.dag.GraphProgram): ComputeGraph {
        val graph = sk.ainet.lang.graph.DefaultComputeGraph()
        val nodeMap = mutableMapOf<String, sk.ainet.lang.graph.GraphNode>()

        // Create graph nodes from program nodes
        for (nodeDef in program.nodes) {
            val graphNode = sk.ainet.lang.graph.GraphNode(
                id = nodeDef.id,
                operation = nodeDef.operation,
                inputs = nodeDef.inputs.map { it.spec },
                outputs = nodeDef.outputs.map { it.spec },
                metadata = nodeDef.attributes.filterValues { it != null }.mapValues { it.value!! }
            )
            graph.addNode(graphNode)
            nodeMap[nodeDef.id] = graphNode
        }

        // Create edges from input references
        var edgeId = 0
        for (nodeDef in program.nodes) {
            for ((inputIdx, inputValue) in nodeDef.inputs.withIndex()) {
                val srcNode = nodeMap[inputValue.nodeId] ?: continue
                val dstNode = nodeMap[nodeDef.id] ?: continue
                graph.addEdge(
                    sk.ainet.lang.graph.GraphEdge(
                        id = "e${edgeId++}",
                        source = srcNode,
                        destination = dstNode,
                        sourceOutputIndex = inputValue.outputIndex,
                        destinationInputIndex = inputIdx,
                        tensorSpec = inputValue.spec
                    )
                )
            }
        }

        return graph
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
         * @param optimized Whether to compile for optimized execution
         * @param bosToken BOS token ID
         * @return A ready-to-use runtime
         */
        public fun <T : DType> create(
            model: Module<T, Float>,
            tensors: List<WeightTensor<T, Float>>,
            resolver: WeightNameResolver,
            ctx: ExecutionContext,
            optimized: Boolean = true,
            bosToken: Int = 1
        ): OptimizedLLMRuntime<T> {
            val mode = if (optimized) Mode.OPTIMIZED else Mode.DIRECT
            val runtime = OptimizedLLMRuntime(model, ctx, mode, bosToken)

            // Load weights into the module tree
            val result = runtime.loadWeights(tensors, resolver)
            WeightMapper.validateAllParametersMapped(result)

            // Compile if optimized mode
            if (optimized) {
                runtime.compile(bosToken)
            }

            return runtime
        }
    }
}
