package sk.ainet.apps.llm

import sk.ainet.compile.opt.GraphOptimizationPipeline
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.graph.exec.ComputeGraphExecutor
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.normalization.RMSNormalization
import sk.ainet.lang.nn.transformer.KVCache
import sk.ainet.lang.nn.transformer.MultiHeadAttention
import sk.ainet.lang.nn.transformer.ResidualAdd
import sk.ainet.lang.nn.transformer.RoPE
import sk.ainet.lang.nn.transformer.SwiGLUFFN
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.trace.TraceSession
import sk.ainet.lang.types.DType
import kotlin.math.sqrt
import kotlin.reflect.KClass

/**
 * Transformer block that supports hybrid execution: compiled subgraphs for
 * pure-compute operations + imperative execution for stateful operations
 * (RoPE, KVCache, SDPA).
 *
 * In DIRECT mode, behaves identically to [TransformerBlock].
 * In HYBRID mode (after [compile]), executes:
 * 1. Compiled attn_compute subgraph: RMSNorm → Q/K/V projections
 * 2. Imperative attention: RoPE → KVCache → SDPA → O_proj
 * 3. Residual add
 * 4. Compiled ffn_compute subgraph: RMSNorm → SwiGLU FFN
 * 5. Residual add
 *
 * Weight mapping is preserved: [modules] returns the original module list,
 * so [WeightMapper] finds MultiHeadAttention params unchanged.
 */
public class HybridTransformerBlock<T : DType, V>(
    private val modulesList: List<Module<T, V>>,
    override val name: String = "HybridTransformerBlock"
) : Module<T, V>() {

    override val modules: List<Module<T, V>>
        get() = modulesList

    // --- Module references extracted from the module list ---
    private val attnNorm: RMSNormalization<T, V>?
    private val mha: MultiHeadAttention<T, V>?
    private val ffnNorm: RMSNormalization<T, V>?
    private val ffn: SwiGLUFFN<T, V>?

    // Residual block boundaries (same logic as TransformerBlock)
    private val residualBlockStarts: Map<Int, Int> = buildMap {
        var blockStart = 0
        for (i in modulesList.indices) {
            if (modulesList[i] is ResidualAdd<*, *>) {
                put(i, blockStart)
                blockStart = i + 1
            }
        }
    }

    init {
        // Extract typed references from the module list.
        // Expected structure: [RMSNorm, MHA, ResidualAdd, RMSNorm, SwiGLUFFN, ResidualAdd]
        @Suppress("UNCHECKED_CAST")
        attnNorm = modulesList.filterIsInstance<RMSNormalization<T, V>>().getOrNull(0)
        @Suppress("UNCHECKED_CAST")
        mha = modulesList.filterIsInstance<MultiHeadAttention<T, V>>().firstOrNull()
        @Suppress("UNCHECKED_CAST")
        ffnNorm = modulesList.filterIsInstance<RMSNormalization<T, V>>().getOrNull(1)
        @Suppress("UNCHECKED_CAST")
        ffn = modulesList.filterIsInstance<SwiGLUFFN<T, V>>().firstOrNull()
    }

    // --- Compiled state (populated by compile()) ---
    private var hybridMode = false

    // Attn compute subgraph: norm → Q/K/V projections
    private var attnExecutor: ComputeGraphExecutor? = null
    private var attnWeightMap: Map<String, Tensor<*, *>> = emptyMap()
    private var attnInputNodeId: String? = null

    // FFN compute subgraph: norm → SwiGLU
    private var ffnExecutor: ComputeGraphExecutor? = null
    private var ffnWeightMap: Map<String, Tensor<*, *>> = emptyMap()
    private var ffnInputNodeId: String? = null

    /**
     * Compile subgraphs for pure-compute modules.
     *
     * After compilation, [onForward] uses the compiled subgraphs for
     * norm + projection / norm + FFN, and runs attention imperatively.
     */
    public fun compile(ctx: ExecutionContext, dtype: KClass<T>, pipeline: GraphOptimizationPipeline): List<String> {
        val diagnostics = mutableListOf<String>()
        val norm = attnNorm
        val attention = mha
        val fNorm = ffnNorm
        val feedForward = ffn

        if (norm == null || attention == null || fNorm == null || feedForward == null) {
            diagnostics.add("$name: cannot compile — missing expected modules")
            return diagnostics
        }

        val dim = attention.dim
        val sampleInput = createSampleInput(dim, dtype, ctx)

        // --- Compile attn_compute subgraph: norm → Q/K/V projections ---
        val attnResult = traceAttnCompute(norm, attention, sampleInput, ctx, pipeline)
        if (attnResult != null) {
            attnExecutor = attnResult.executor
            attnWeightMap = attnResult.weightMap
            attnInputNodeId = attnResult.inputNodeId
            diagnostics.add("$name/attn_compute: ${attnResult.diagnostics}")
        }

        // --- Compile ffn_compute subgraph: norm → SwiGLU FFN ---
        val ffnResult = traceFFNCompute(fNorm, feedForward, sampleInput, ctx, pipeline)
        if (ffnResult != null) {
            ffnExecutor = ffnResult.executor
            ffnWeightMap = ffnResult.weightMap
            ffnInputNodeId = ffnResult.inputNodeId
            diagnostics.add("$name/ffn_compute: ${ffnResult.diagnostics}")
        }

        hybridMode = attnExecutor != null && ffnExecutor != null
        return diagnostics
    }

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        return if (hybridMode) {
            hybridForward(input, ctx)
        } else {
            directForward(input, ctx)
        }
    }

    // --- DIRECT mode: same as TransformerBlock ---

    private fun directForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        // Diagnostic gates — JVM-only, always false on JS/wasm/native.
        // See `sk.ainet.apps.llm.diag.envFlag` / `dumpStats`.
        val dumpInner = sk.ainet.apps.llm.diag.envFlag("GEMMA4_DUMP_INNER") && name == "blk.0"
        val dumpMha = sk.ainet.apps.llm.diag.envFlag("GEMMA4_DUMP_MHA") && name == "blk.0"
        // GEMMA4_DUMP_BLOCKS=1 → one line per block (attn output + block output)
        // for every block in the model. Used to bisect against HF's per-layer
        // dump (`/tmp/dump_gemma4_intermediate.py`).
        val dumpBlocks = sk.ainet.apps.llm.diag.envFlag("GEMMA4_DUMP_BLOCKS")
        val outputs = arrayOfNulls<Any>(modulesList.size + 1)
        outputs[0] = input
        var tmp = input
        if (dumpInner) sk.ainet.apps.llm.diag.dumpStats("[blk.0 input]                  ", tmp)
        for (i in modulesList.indices) {
            val module = modulesList[i]
            val blockStart = residualBlockStarts[i]
            if (blockStart != null) {
                @Suppress("UNCHECKED_CAST")
                (module as ResidualAdd<T, V>).savedInput =
                    outputs[blockStart] as Tensor<T, V>
            }
            // Set the MHA substep-dump gate ONLY around this block's MHA call.
            // The MHA module is named just "attn" — every block has one with
            // the same name — so MHA can't gate its own dump on the block id.
            // Toggle the static flag from here, where we know which block we're in.
            val isMhaCall = dumpMha && module is MultiHeadAttention<*, *>
            if (isMhaCall) sk.ainet.lang.nn.transformer.MultiHeadAttentionDiag.shouldDumpThisCall = true
            tmp = module.forward(tmp, ctx)
            if (isMhaCall) sk.ainet.lang.nn.transformer.MultiHeadAttentionDiag.shouldDumpThisCall = false
            outputs[i + 1] = tmp
            if (dumpInner) sk.ainet.apps.llm.diag.dumpStats("[blk.0 after ${module::class.simpleName}/${module.name}]", tmp)
            if (dumpBlocks && module is MultiHeadAttention<*, *>) {
                sk.ainet.apps.llm.diag.dumpStats("[$name attn-out] ", tmp)
            }
        }
        if (dumpBlocks) sk.ainet.apps.llm.diag.dumpStats("[$name block-out]", tmp)
        return tmp
    }

    // --- HYBRID mode: compiled subgraphs + imperative attention ---

    private fun hybridForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        val ops = ctx.ops
        val attention = mha!!

        // Step 1: Compiled attn_compute → Q, K, V
        val qkvConcat = executeSubgraph(attnExecutor!!, attnWeightMap, attnInputNodeId!!, input, ctx)
        val (q, k, v) = splitQKV(qkvConcat, attention, ops)

        // Step 2: Imperative attention (RoPE, KVCache, SDPA, O_proj)
        val attnOut = imperativeAttention(q, k, v, attention, ctx)

        // Step 3: Residual add
        val afterAttn = ops.add(attnOut, input)

        // Step 4: Compiled ffn_compute → FFN output
        val ffnOut = executeSubgraph(ffnExecutor!!, ffnWeightMap, ffnInputNodeId!!, afterAttn, ctx)

        // Step 5: Residual add
        return ops.add(ffnOut, afterAttn)
    }

    private fun splitQKV(
        qkvConcat: Tensor<T, V>,
        mha: MultiHeadAttention<T, V>,
        ops: sk.ainet.lang.tensor.ops.TensorOps
    ): Triple<Tensor<T, V>, Tensor<T, V>, Tensor<T, V>> {
        val dim = mha.dim
        val kvDim = mha.kvDim
        // qkvConcat shape: [seqLen, dim + kvDim + kvDim]
        // Use narrow to extract each projection's slice along the last dimension
        val lastDim = qkvConcat.rank - 1
        val q = ops.narrow(qkvConcat, lastDim, 0, dim)
        val k = ops.narrow(qkvConcat, lastDim, dim, kvDim)
        val v = ops.narrow(qkvConcat, lastDim, dim + kvDim, kvDim)
        return Triple(q, k, v)
    }

    /**
     * Reimplements MultiHeadAttention lines 135-191 (stateful portion):
     * reshape → QK-norm → RoPE → KVCache → GQA → SDPA → merge → O_proj
     */
    private fun imperativeAttention(
        qRaw: Tensor<T, V>,
        kRaw: Tensor<T, V>,
        vRaw: Tensor<T, V>,
        mha: MultiHeadAttention<T, V>,
        ctx: ExecutionContext
    ): Tensor<T, V> {
        val ops = ctx.ops
        val scale = 1.0f / sqrt(mha.headDim.toFloat())
        val seqLen = if (qRaw.rank >= 2) qRaw.shape[qRaw.rank - 2] else 1

        // Reshape to multi-head
        var q = ops.reshape(qRaw, Shape(mha.nHeads, seqLen, mha.headDim))
        var k = ops.reshape(kRaw, Shape(mha.nKVHeads, seqLen, mha.headDim))
        val vReshaped = ops.reshape(vRaw, Shape(mha.nKVHeads, seqLen, mha.headDim))

        // Optional QK-Norm
        val qNormMod = mha.qNorm
        val kNormMod = mha.kNorm
        if (qNormMod != null && kNormMod != null) {
            q = qNormMod.forward(q, ctx)
            k = kNormMod.forward(k, ctx)
        }

        // RoPE (position-dependent — the key stateful interaction)
        val ropeModule = mha.rope
        if (ropeModule != null) {
            val position = mha.kvCache?.position ?: 0
            q = ropeModule.forward(q, position, ctx)
            k = ropeModule.forward(k, position, ctx)
        }

        // KV Cache update
        val (fullK, fullV) = if (mha.kvCache != null) {
            mha.kvCache!!.update(k, vReshaped, ctx)
        } else {
            k to vReshaped
        }

        // Expand KV heads for GQA
        val expandedK = if (mha.nKVHeads < mha.nHeads) {
            repeatKVHeads(fullK, mha.nHeads / mha.nKVHeads, ops)
        } else fullK
        val expandedV = if (mha.nKVHeads < mha.nHeads) {
            repeatKVHeads(fullV, mha.nHeads / mha.nKVHeads, ops)
        } else fullV

        // SDPA
        val qBatched = ops.unsqueeze(q, 0)
        val kBatched = ops.unsqueeze(expandedK, 0)
        val vBatched = ops.unsqueeze(expandedV, 0)

        val attnOut = ops.scaledDotProductAttention(
            query = qBatched,
            key = kBatched,
            value = vBatched,
            mask = null,
            scale = scale,
            causal = mha.causal
        )

        // Merge heads
        val squeezed = ops.squeeze(attnOut, 0)
        val merged = ops.reshape(squeezed, Shape(seqLen, mha.dim))

        // Output projection
        val mhaParams = (mha as ModuleParameters<T, V>).params
        val oWIdx = if (mha.bias) 6 else 3
        val wO = mhaParams[oWIdx].value
        var output = ops.matmul(merged, ops.transpose(wO))
        if (mha.bias) {
            output = ops.add(output, mhaParams[oWIdx + 1].value)
        }
        return output
    }

    private fun repeatKVHeads(
        t: Tensor<T, V>,
        repeats: Int,
        ops: sk.ainet.lang.tensor.ops.TensorOps
    ): Tensor<T, V> {
        if (repeats == 1) return t
        // Repeat each KV head individually so head mapping matches GQA:
        // head h uses KV head h/repeats → [kv0]*repeats ++ [kv1]*repeats ++ ...
        val nKVHeads = t.shape[0]
        val expanded = mutableListOf<Tensor<T, V>>()
        for (h in 0 until nKVHeads) {
            val headSlice = ops.narrow(t, 0, h, 1) // [1, seqLen, headDim]
            repeat(repeats) { expanded.add(headSlice) }
        }
        return ops.concat(expanded, dim = 0)
    }

    // --- Subgraph tracing and compilation ---

    private data class SubgraphResult(
        val executor: ComputeGraphExecutor,
        val weightMap: Map<String, Tensor<*, *>>,
        val inputNodeId: String?,
        val diagnostics: String
    )

    private fun traceAttnCompute(
        norm: RMSNormalization<T, V>,
        mha: MultiHeadAttention<T, V>,
        sampleInput: Tensor<T, V>,
        ctx: ExecutionContext,
        pipeline: GraphOptimizationPipeline
    ): SubgraphResult? {
        val tapingCtx = DefaultGraphExecutionContext.tape(baseOps = ctx.ops)
        tapingCtx.startRecording()

        // Create the input through the taping context so that operator-overloaded
        // ops in modules (like RMSNorm's `input * input`) dispatch through the
        // tracing wrapper, not the tensor's bound CPU ops.
        val tracedInput: Tensor<T, V> = tapingCtx.fromFloatArray(
            sampleInput.shape, sampleInput.dtype, sampleInput.data.copyToFloatArray()
        )

        // Trace: norm → Q/K/V projections → concat
        var x = norm.forward(tracedInput, tapingCtx)
        val tracingOps = tapingCtx.ops
        val mhaParams = (mha as ModuleParameters<T, V>).params
        val qWIdx = 0
        val kWIdx = if (mha.bias) 2 else 1
        val vWIdx = if (mha.bias) 4 else 2

        val q = tracingOps.matmul(x, tracingOps.transpose(mhaParams[qWIdx].value))
        val k = tracingOps.matmul(x, tracingOps.transpose(mhaParams[kWIdx].value))
        val v = tracingOps.matmul(x, tracingOps.transpose(mhaParams[vWIdx].value))

        // Add bias if enabled
        val qFinal = if (mha.bias) tracingOps.add(q, mhaParams[qWIdx + 1].value) else q
        val kFinal = if (mha.bias) tracingOps.add(k, mhaParams[kWIdx + 1].value) else k
        val vFinal = if (mha.bias) tracingOps.add(v, mhaParams[vWIdx + 1].value) else v

        // Concat Q, K, V for single-output graph
        tracingOps.concat(listOf(qFinal, kFinal, vFinal), dim = qFinal.rank - 1)

        val tape = tapingCtx.stopRecording()
        return buildSubgraph(tape, sampleInput, ctx, pipeline, "attn_compute")
    }

    private fun traceFFNCompute(
        norm: Module<T, V>,
        ffn: Module<T, V>,
        sampleInput: Tensor<T, V>,
        ctx: ExecutionContext,
        pipeline: GraphOptimizationPipeline
    ): SubgraphResult? {
        val tapingCtx = DefaultGraphExecutionContext.tape(baseOps = ctx.ops)
        tapingCtx.startRecording()

        // Create input through taping context (see traceAttnCompute for rationale)
        val tracedInput: Tensor<T, V> = tapingCtx.fromFloatArray(
            sampleInput.shape, sampleInput.dtype, sampleInput.data.copyToFloatArray()
        )

        // Trace: norm → FFN
        var x = norm.forward(tracedInput, tapingCtx)
        ffn.forward(x, tapingCtx)

        val tape = tapingCtx.stopRecording()
        return buildSubgraph(tape, sampleInput, ctx, pipeline, "ffn_compute")
    }

    private fun buildSubgraph(
        tape: Any?,
        sampleInput: Tensor<T, V>,
        ctx: ExecutionContext,
        pipeline: GraphOptimizationPipeline,
        label: String
    ): SubgraphResult? {
        val tapeObj = tape as? DefaultExecutionTape ?: return null
        val session = tapeObj.session

        val rawGraph = tapeObj.toComputeGraph(
            synthesizeExternalInputs = true,
            inputTensorIds = emptySet(),
            embedConstants = false
        )

        // Validate
        try {
            rawGraph.getTopologicalOrder()
        } catch (_: IllegalStateException) {
            return null
        }

        // Optimize
        val result = pipeline.optimize(rawGraph)
        val graph = result.graph
        val diagnostics = result.passResults.flatMap { it.diagnostics }

        // Build weight map and detect input node
        val (weightMap, inputNodeId) = buildSubgraphWeightMap(session, graph, sampleInput)

        val executor = ComputeGraphExecutor(graph, ctx.ops)

        return SubgraphResult(
            executor = executor,
            weightMap = weightMap,
            inputNodeId = inputNodeId,
            diagnostics = "${graph.nodes.size} nodes, ${graph.edges.size} edges" +
                if (diagnostics.isNotEmpty()) ", fusions: ${diagnostics.size}" else ""
        )
    }

    @Suppress("UNCHECKED_CAST")
    private fun buildSubgraphWeightMap(
        session: TraceSession,
        graph: sk.ainet.lang.graph.ComputeGraph,
        sampleInput: Tensor<T, V>
    ): Pair<Map<String, Tensor<*, *>>, String?> {
        val result = mutableMapOf<String, Tensor<*, *>>()
        var inputNodeId: String? = null

        val inputOps = setOf("input", "weight", "parameter", "constant")
        for (node in graph.nodes) {
            if (node.operationName !in inputOps) continue
            val outputSpec = node.outputs.firstOrNull() ?: continue
            val tensor = session.resolve(outputSpec.name) ?: continue

            // The input node matches the sample input's shape.
            // Weight/parameter tensors have different shapes.
            if (inputNodeId == null
                && tensor.shape.dimensions.contentEquals(sampleInput.shape.dimensions)
            ) {
                inputNodeId = node.id
                continue
            }
            result[node.id] = tensor
        }
        return result to inputNodeId
    }

    @Suppress("UNCHECKED_CAST")
    private fun executeSubgraph(
        executor: ComputeGraphExecutor,
        weightMap: Map<String, Tensor<*, *>>,
        inputNodeId: String,
        input: Tensor<T, V>,
        ctx: ExecutionContext
    ): Tensor<T, V> {
        val inputMap = mutableMapOf<String, Tensor<DType, Float>>()
        inputMap[inputNodeId] = input as Tensor<DType, Float>
        for ((nodeId, tensor) in weightMap) {
            inputMap[nodeId] = tensor as Tensor<DType, Float>
        }

        val results = executor.execute(inputMap)
        return results.values.lastOrNull() as? Tensor<T, V>
            ?: error("Subgraph execution produced no outputs")
    }

    @Suppress("UNCHECKED_CAST")
    private fun createSampleInput(dim: Int, dtype: KClass<T>, ctx: ExecutionContext): Tensor<T, V> {
        // Use the execution context to create a real tensor (not VoidOpsTensor)
        // so that operator-overloaded ops in modules (like RMSNorm's input * input)
        // correctly dispatch through the ops backend during tracing.
        return ctx.fromFloatArray(Shape(1, dim), dtype, FloatArray(dim) { 0.1f })
    }
}
