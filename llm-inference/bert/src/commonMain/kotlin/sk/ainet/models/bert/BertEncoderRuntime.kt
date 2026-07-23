package sk.ainet.models.bert

import sk.ainet.apps.llm.getLLMOptimizationPipeline
import sk.ainet.apps.llm.graph.LLMFusedOpHandlers
import sk.ainet.apps.llm.weights.BertSafeTensorsNameResolver
import sk.ainet.context.ExecutionContext
import sk.ainet.io.weights.MappingConfig
import sk.ainet.io.weights.WeightMapper
import sk.ainet.io.weights.WeightNameResolver
import sk.ainet.io.weights.WeightTensor
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.graph.exec.ComputeGraphExecutor
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.div
import sk.ainet.lang.tensor.matmul
import sk.ainet.lang.tensor.mean
import sk.ainet.lang.tensor.plus
import sk.ainet.lang.tensor.reshape
import sk.ainet.lang.tensor.sqrt
import sk.ainet.lang.tensor.sum
import sk.ainet.lang.tensor.t
import sk.ainet.lang.tensor.times
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.Int32
import kotlin.reflect.KClass

/** Execution strategy for [BertEncoderRuntime]. */
public enum class BertExecutionMode {
    /** Run the module tree eagerly — the primary JVM path. */
    DIRECT,

    /** Trace the encoder into an optimized ComputeGraph and execute the graph. */
    OPTIMIZED,
}

/**
 * Sentence-pooling strategy applied by [BertEncoderRuntime.encode] on top of
 * the encoder's hidden states — the sentence-transformers `1_Pooling` module.
 * Pooling stays outside the traced graph, so the choice has no effect on the
 * OPTIMIZED path or the StableHLO export.
 */
public enum class BertPooling {
    /** Mask-weighted mean over token positions (LEAF, E5, MiniLM). */
    MEAN,

    /** Hidden state of the first token — `[CLS]` (BGE, GTE). */
    CLS,
}

/**
 * Encoder runtime for BERT sentence embeddings on the DSL path.
 *
 * Wraps a [bertNetwork] module (a complete `tokens → hidden-states` encoder)
 * and adds what sentence embedding needs on top of the pure encoder graph:
 * masked mean pooling, the optional sentence-transformers dense projection
 * (`2_Dense`), and L2 normalization. Pooling and projection stay outside the
 * DSL network on purpose — the traced/exported graph remains a clean encoder,
 * and the pooling mask is dynamic per call.
 *
 * Intended use is one unpadded sequence per [encode] call; the attention mask
 * only affects pooling (attention itself is bidirectional over the full
 * sequence, exactly like the eager runtime this class replaces).
 *
 * Construction goes through [createBertEncoderRuntime], which maps checkpoint
 * tensors into the module via [WeightMapper].
 */
public class BertEncoderRuntime<T : DType>(
    private val model: Module<T, Float>,
    public val config: BertModelConfig,
    private val ctx: ExecutionContext,
    private val dtype: KClass<T>,
    private val projectionWeight: Tensor<T, Float>? = null,
    private val projectionBias: Tensor<T, Float>? = null,
    private val mode: BertExecutionMode = BertExecutionMode.DIRECT,
    private val pooling: BertPooling = BertPooling.MEAN,
) {

    /** Output dimensionality of [encode]: projection out-features when present, else hidden size. */
    public val dimensions: Int get() = config.projectionDim ?: config.hiddenSize

    /**
     * Full-sequence encoder forward: `[L]` token ids → `[L, hiddenSize]`
     * hidden states.
     */
    public fun forward(tokenIds: IntArray): Tensor<T, Float> = ctx.scratch.scope {
        require(tokenIds.isNotEmpty()) { "BertEncoderRuntime: tokenIds must not be empty" }
        when (mode) {
            BertExecutionMode.DIRECT -> model.forward(tokenTensor(tokenIds), ctx)
            BertExecutionMode.OPTIMIZED -> forwardOptimized(tokenIds)
        }
    }

    /**
     * Encode tokens into a single embedding vector: forward → pooling
     * ([BertPooling.MEAN], mask-weighted when [attentionMask] is given, or
     * [BertPooling.CLS]) → optional dense projection → L2 normalization.
     *
     * @param tokenIds token IDs including `[CLS]` and `[SEP]`
     * @param attentionMask 1 for real tokens, 0 for padding; affects MEAN pooling only
     * @return normalized vector of size [dimensions]
     */
    public fun encode(tokenIds: IntArray, attentionMask: IntArray? = null): FloatArray {
        val hiddenStates = forward(tokenIds)
        val seqLen = tokenIds.size

        var pooled = when (pooling) {
            BertPooling.CLS ->
                // First row of [L, hidden] — the [CLS] position — as a [hidden] vector.
                ctx.ops.narrow(hiddenStates, 0, 0, 1).reshape(Shape(config.hiddenSize))
            BertPooling.MEAN -> if (attentionMask != null) {
                require(attentionMask.size == seqLen) {
                    "BertEncoderRuntime: attentionMask size ${attentionMask.size} != tokenIds size $seqLen"
                }
                val maskTensor = ctx.fromFloatArray<T, Float>(
                    Shape(seqLen, 1), dtype,
                    FloatArray(seqLen) { attentionMask[it].toFloat() }
                )
                val masked = hiddenStates * maskTensor
                val summed = masked.sum(dim = 0)
                val count = attentionMask.sumOf { it }.toFloat().coerceAtLeast(1f)
                summed / count
            } else {
                hiddenStates.mean(dim = 0)
            }
        }

        if (projectionWeight != null) {
            // sentence-transformers Dense head; bias is optional (LEAF models
            // ship bias=false). The legacy eager runtime required both tensors
            // and silently skipped the projection on bias-free heads — fixed here.
            pooled = pooled.matmul(projectionWeight.t())
            if (projectionBias != null) pooled = pooled + projectionBias
        }

        pooled = l2Normalize(pooled)

        val out = FloatArray(pooled.volume)
        for (i in out.indices) out[i] = pooled.data[i]
        return out
    }

    // ------------------------------------------------------------------
    // OPTIMIZED mode: trace → ComputeGraph → fused execution.
    //
    // Traced graphs are shape-specialized: one compiled graph per sequence
    // length, kept in a small LRU cache (re-tracing a 6×384 encoder is cheap,
    // and embedding workloads cluster around few lengths after chunking).
    // ------------------------------------------------------------------

    private class CompiledEncoder(
        val executor: ComputeGraphExecutor,
        val weightMap: Map<String, Tensor<*, *>>,
        val inputNodeId: String,
    )

    private val compiledBySeqLen = LinkedHashMap<Int, CompiledEncoder>()

    /**
     * Pre-warm the OPTIMIZED cache for [sampleSeqLen] and return compile
     * diagnostics. Optional — [forward] compiles lazily per sequence length.
     */
    public fun compile(sampleSeqLen: Int = 8): List<String> {
        val diagnostics = mutableListOf<String>()
        compiledFor(IntArray(sampleSeqLen) { it % config.vocabSize }, diagnostics)
        return diagnostics
    }

    /**
     * Trace one encoder forward at [seqLen] and return the recorded tape —
     * the entry point for graph export (StableHLO and friends).
     */
    public fun exportTape(seqLen: Int): DefaultExecutionTape {
        val tapingCtx = DefaultGraphExecutionContext.tape(baseOps = ctx.ops)
        tapingCtx.startRecording()
        model.forward(tokenTensor(IntArray(seqLen) { it % config.vocabSize }), tapingCtx)
        val tape = tapingCtx.stopRecording()
        return tape as? DefaultExecutionTape
            ?: error("Expected DefaultExecutionTape but got ${tape?.let { it::class.simpleName }}")
    }

    private fun forwardOptimized(tokenIds: IntArray): Tensor<T, Float> {
        val compiled = compiledBySeqLen.getOrPut(tokenIds.size) {
            if (compiledBySeqLen.size >= COMPILED_CACHE_LIMIT) {
                compiledBySeqLen.remove(compiledBySeqLen.keys.first())
            }
            compiledFor(tokenIds, mutableListOf())
        }

        // Feed the Int32 index tensor the traced gather consumes, plus every
        // weight leaf resolved at compile time.
        val input = ctx.fromIntArray<Int32, Float>(Shape(tokenIds.size), Int32::class, tokenIds)
        val inputMap = mutableMapOf<String, Tensor<DType, Float>>()
        @Suppress("UNCHECKED_CAST")
        inputMap[compiled.inputNodeId] = input as Tensor<DType, Float>
        for ((nodeId, tensor) in compiled.weightMap) {
            @Suppress("UNCHECKED_CAST")
            inputMap[nodeId] = tensor as Tensor<DType, Float>
        }

        val results = compiled.executor.execute(inputMap)
        @Suppress("UNCHECKED_CAST")
        return results.values.lastOrNull() as? Tensor<T, Float>
            ?: error("BertEncoderRuntime: graph execution produced no outputs")
    }

    private fun compiledFor(sampleTokenIds: IntArray, diagnostics: MutableList<String>): CompiledEncoder {
        // Fused-op replay handlers (RMSNorm / SwiGLU / QKV decompositions).
        LLMFusedOpHandlers.registerAll()

        // 1. Record one eager forward through a tape-recording context.
        val tapingCtx = DefaultGraphExecutionContext.tape(baseOps = ctx.ops)
        tapingCtx.startRecording()
        model.forward(tokenTensor(sampleTokenIds), tapingCtx)
        val tape = tapingCtx.stopRecording()
        val tapeObj = tape as? DefaultExecutionTape
            ?: error("Expected DefaultExecutionTape but got ${tape?.let { it::class.simpleName }}")
        val session = tapeObj.session

        // 2. Tape → raw ComputeGraph (weights stay placeholder inputs).
        val rawGraph = tapeObj.toComputeGraph(
            synthesizeExternalInputs = true,
            inputTensorIds = emptySet(),
            embedConstants = false,
        )
        diagnostics.add("Traced graph: ${rawGraph.nodes.size} nodes, ${rawGraph.edges.size} edges")

        // 3. Optimize unless the raw graph fails validation.
        val graph = try {
            rawGraph.getTopologicalOrder()
            val result = getLLMOptimizationPipeline().optimize(rawGraph)
            diagnostics.addAll(result.passResults.flatMap { it.diagnostics })
            result.graph
        } catch (e: IllegalStateException) {
            diagnostics.add("WARNING: raw traced graph failed validation (${e.message}) — executing unoptimized")
            rawGraph
        }

        // 4. Split graph leaves into the token input vs static tensors.
        //    Module parameters are known by identity. Non-parameter leaves are
        //    either trace-time constants the modules created during forward
        //    (e.g. attention scale/mask tensors — valid for this seqLen, so
        //    they ride along in the static map) or the token input. The token
        //    input is the unique Int32 leaf: the index tensor Embedding
        //    creates for gather. (BertEmbeddings keeps position/type lookups
        //    index-free precisely so this stays unambiguous.)
        val paramTensors = HashSet<Tensor<*, *>>()
        collectParams(model, paramTensors)

        val weightMap = mutableMapOf<String, Tensor<*, *>>()
        val inputCandidates = mutableListOf<String>()
        for (node in graph.nodes) {
            if (node.operationName !in setOf("input", "weight", "parameter", "constant")) continue
            val outputSpec = node.outputs.firstOrNull() ?: continue
            val tensor = session.resolve(outputSpec.name) ?: continue
            when {
                paramTensors.any { it === tensor } -> weightMap[node.id] = tensor
                tensor.dtype == Int32::class -> inputCandidates += node.id
                else -> weightMap[node.id] = tensor
            }
        }
        require(inputCandidates.size == 1) {
            "BertEncoderRuntime: expected exactly one Int32 graph leaf (the token input), " +
                "found ${inputCandidates.size}: $inputCandidates"
        }
        diagnostics.add("Input node: ${inputCandidates.first()}, weight map: ${weightMap.size} entries")

        return CompiledEncoder(
            executor = ComputeGraphExecutor(graph, ctx.ops),
            weightMap = weightMap,
            inputNodeId = inputCandidates.first(),
        )
    }

    private fun collectParams(module: Module<*, *>, out: MutableSet<Tensor<*, *>>) {
        if (module is ModuleParameters<*, *>) {
            module.params.forEach { out.add(it.value) }
        }
        module.modules.forEach { collectParams(it, out) }
    }

    private companion object {
        private const val COMPILED_CACHE_LIMIT = 16
    }

    private fun tokenTensor(tokenIds: IntArray): Tensor<T, Float> =
        ctx.fromFloatArray(
            Shape(tokenIds.size), dtype,
            FloatArray(tokenIds.size) { tokenIds[it].toFloat() },
        )

    private fun l2Normalize(tensor: Tensor<T, Float>): Tensor<T, Float> {
        val squared = tensor * tensor
        val sumSquared = squared.sum()
        val norm = (sumSquared + 1e-12).sqrt()
        return tensor / norm
    }
}

/**
 * Build a [BertEncoderRuntime] from checkpoint tensors: constructs
 * `bertNetwork(config)`, maps every DSL parameter by name via [resolver]
 * (strict — shape fallback stays off so same-shaped Q/K/V tensors can't
 * cross-wire), and pulls the optional `2_Dense` projection pair
 * (`linear.weight` / `linear.bias`) out of [tensors] for the runtime.
 *
 * @param tensors as produced by [BertNetworkLoader.loadWeightTensors]
 */
public inline fun <reified T : DType> createBertEncoderRuntime(
    config: BertModelConfig,
    tensors: List<WeightTensor<T, Float>>,
    ctx: ExecutionContext,
    resolver: WeightNameResolver = BertSafeTensorsNameResolver(),
    mode: BertExecutionMode = BertExecutionMode.DIRECT,
    debug: Boolean = false,
    pooling: BertPooling = BertPooling.MEAN,
): BertEncoderRuntime<T> {
    val model = bertNetwork<T, Float>(config)

    val result = WeightMapper.applyWeights(
        model, tensors,
        MappingConfig(
            usePathBasedMatching = false,
            fallbackToShapeMatching = false,
            debug = debug,
            nameResolver = resolver,
        )
    )
    require(result.mapped == result.total) {
        buildString {
            appendLine("Failed to map ${result.total - result.mapped}/${result.total} BERT parameters:")
            result.missingParams.forEach { appendLine("  - $it") }
            if (result.unusedTensors.isNotEmpty()) {
                appendLine("Unused tensors (${result.unusedTensors.size}):")
                result.unusedTensors.take(10).forEach { appendLine("  - $it") }
            }
        }.trim()
    }

    val projectionWeight = tensors.firstOrNull { it.name == BertNetworkLoader.PROJECTION_WEIGHT }?.tensor
    val projectionBias = tensors.firstOrNull { it.name == BertNetworkLoader.PROJECTION_BIAS }?.tensor
    if (config.projectionDim != null) {
        requireNotNull(projectionWeight) {
            "config.projectionDim=${config.projectionDim} but no ${BertNetworkLoader.PROJECTION_WEIGHT} tensor was provided"
        }
    }
    require(projectionBias == null || projectionWeight != null) {
        "${BertNetworkLoader.PROJECTION_BIAS} provided without ${BertNetworkLoader.PROJECTION_WEIGHT}"
    }

    return BertEncoderRuntime(
        model = model,
        config = config,
        ctx = ctx,
        dtype = T::class,
        projectionWeight = projectionWeight,
        projectionBias = projectionBias,
        mode = mode,
        pooling = pooling,
    )
}
