package sk.ainet.lang.nn.dsl.decoder

import sk.ainet.compile.hlo.toStableHlo
import sk.ainet.compile.opt.passes.BlockQuantMatmulFusionPass
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.operators.bind
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.trace.PackedConstantHandling
import sk.ainet.lang.types.DType
import sk.ainet.tape.Execution
import kotlin.reflect.KClass

/**
 * The shape of a single-token decode step with a static past of [past] positions: the token
 * embedding `[1, hidden]`, RoPE tables `[1, headDim]` (a second pair when [slidingRopeBase]), and
 * K/V `[1, nKVHeads, past, headDim]` per layer.
 */
public class DecodeStepSpec(
    public val hidden: Int,
    public val headDim: Int,
    public val nKVHeads: Int,
    public val nLayers: Int,
    public val past: Int,
    public val slidingRopeBase: Boolean = false,
)

/** The traced graph inputs a decode step runs on (see [DecodeStepExport.export]). */
public class DecodeStepInputs<T : DType, V>(
    public val embedded: Tensor<T, V>,
    public val rope: RopeCosSin<T, V>,
    public val selfK: List<Tensor<T, V>>,
    public val selfV: List<Tensor<T, V>>,
)

/** A function argument or result of an exported step: what it is ([role]) and its shape (`null` if dynamic). */
public class DecodeStepPort(public val role: String, public val shape: List<Int>?)

/**
 * An exported decode step: the StableHLO module text, its arguments and results in function order,
 * and what [BlockQuantMatmulFusionPass] did (`BLOCK_QUANT_FUSED` / `BLOCK_QUANT_SKIPPED` lines).
 */
public class ExportedDecodeStep(
    public val mlir: String,
    public val inputs: List<DecodeStepPort>,
    public val outputs: List<DecodeStepPort>,
    public val diagnostics: List<String>,
) {
    public val foldedBlockQuantMatmuls: Int get() = diagnostics.count { it.startsWith("BLOCK_QUANT_FUSED") }

    /** One line per port, `in <i> <role> <shape>` then `out <i> <role> <shape>`. */
    public fun ioTable(): String =
        (inputs.mapIndexed { i, p -> "in $i ${p.role} ${p.shape}" } + outputs.mapIndexed { i, p -> "out $i ${p.role} ${p.shape}" })
            .joinToString("\n", postfix = "\n")
}

/**
 * Trace one decode step of any KV-cache decoder ([DecoderKvModel.forwardEmbeddedWithPast],
 * `GemmaModel.forwardEmbeddedWithPast`) and export it to StableHLO, for a compiler that wants the
 * whole step as one graph with the embedding lookup left to the host.
 *
 * Ports are named after the model contracts: `embedded`, `cosGlobal`/`sinGlobal` (and
 * `cosSliding`/`sinSliding`), `l<i>.k`/`l<i>.v` in; `logits`, `l<i>.k`/`l<i>.v` out. The emitter orders
 * them by graph topology, so read the order from [ExportedDecodeStep.inputs].
 *
 * With [PackedConstantHandling.KEEP_PACKED] (the default) block-quantized weights stay packed and
 * [BlockQuantMatmulFusionPass] folds every `x · Wᵀ` over one; a packed weight it cannot fold fails
 * the export.
 */
public object DecodeStepExport {

    public fun <T : DType, V> export(
        spec: DecodeStepSpec,
        dtype: KClass<T>,
        functionName: String = "decode_step",
        packedConstants: PackedConstantHandling = PackedConstantHandling.KEEP_PACKED,
        step: (DecodeStepInputs<T, V>, ExecutionContext) -> KvStepOutput<T, V>,
    ): ExportedDecodeStep {
        val kvShape = Shape(1, spec.nKVHeads, spec.past, spec.headDim)
        val ropeShape = Shape(1, spec.headDim)
        val ports = buildList {
            add("embedded" to Shape(1, spec.hidden))
            add("cosGlobal" to ropeShape); add("sinGlobal" to ropeShape)
            if (spec.slidingRopeBase) { add("cosSliding" to ropeShape); add("sinSliding" to ropeShape) }
            for (i in 0 until spec.nLayers) add("l$i.k" to kvShape)
            for (i in 0 until spec.nLayers) add("l$i.v" to kvShape)
        }
        return exportPorts<T, V>(ports, dtype, functionName, packedConstants) { p, ctx ->
            val global = RopeTables(p.getValue("cosGlobal"), p.getValue("sinGlobal"))
            val rope = if (spec.slidingRopeBase) {
                RopeCosSin(global, RopeTables(p.getValue("cosSliding"), p.getValue("sinSliding")))
            } else {
                RopeCosSin(global)
            }
            val inputs = DecodeStepInputs(
                p.getValue("embedded"), rope,
                List(spec.nLayers) { p.getValue("l$it.k") },
                List(spec.nLayers) { p.getValue("l$it.v") },
            )
            val result = step(inputs, ctx)
            linkedMapOf<String, Tensor<T, V>>("logits" to result.logits).apply {
                result.selfK.forEachIndexed { i, t -> put("l$i.k", t) }
                result.selfV.forEachIndexed { i, t -> put("l$i.v", t) }
            }
        }
    }

    /**
     * Trace a step over arbitrary named ports and export it like [export]: one graph input per entry
     * of [inputs] (bound to the trace, in this order), one result per entry [step] returns, each
     * named by its key in [ExportedDecodeStep.inputs] / [ExportedDecodeStep.outputs]. For a decoder
     * whose state is more than K/V (a convolution history, a fixed-capacity cache with its masks).
     */
    public fun <T : DType, V> exportPorts(
        inputs: List<Pair<String, Shape>>,
        dtype: KClass<T>,
        functionName: String = "decode_step",
        packedConstants: PackedConstantHandling = PackedConstantHandling.KEEP_PACKED,
        step: (Map<String, Tensor<T, V>>, ExecutionContext) -> Map<String, Tensor<T, V>>,
    ): ExportedDecodeStep {
        require(inputs.map { it.first }.distinct().size == inputs.size) { "DecodeStepExport: input port names must be unique" }
        // Inputs are bound to the tape context: a bare VoidOpsTensor runs operator overloads
        // (`x * x` in RMSNorm) on its own untraced ops, and the result is captured as a constant.
        val tapeCtx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val roles = LinkedHashMap<String, String>()
        val bound = LinkedHashMap<String, Tensor<T, V>>()
        for ((role, shape) in inputs) {
            val t = graphInput<T, V>(shape, dtype).bind(tapeCtx)
            roles[tapeCtx.session.refOf(t as Tensor<*, *>).id] = role
            bound[role] = t
        }

        var out: Map<String, Tensor<T, V>>? = null
        val tape = tapeCtx.record {
            val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("DecodeStepExport: no tape")
            Execution.tapeStack.pushTape(ct)
            try {
                out = step(bound, this as ExecutionContext)
            } finally {
                Execution.tapeStack.popTape()
            }
        }.first
        val outRoles = HashMap<String, String>()
        for ((role, t) in out!!) {
            val id = tapeCtx.session.refOf(t as Tensor<*, *>).id
            require(outRoles.put(id, role) == null) { "DecodeStepExport: results '${outRoles[id]}' and '$role' are the same tensor" }
        }

        var graph = (tape as DefaultExecutionTape).toComputeGraph(synthesizeExternalInputs = true, packedConstants = packedConstants)
        val diagnostics = if (packedConstants == PackedConstantHandling.KEEP_PACKED) {
            val fused = BlockQuantMatmulFusionPass().apply(graph)
            graph = fused.graph
            fused.diagnostics
        } else {
            emptyList()
        }
        val module = toStableHlo(graph, functionName)
        return ExportedDecodeStep(
            mlir = module.content,
            inputs = module.inputSpecs.map { DecodeStepPort(roles[it.name] ?: "?", it.shape) },
            outputs = module.outputSpecs.map { DecodeStepPort(outRoles[it.name] ?: "?", it.shape) },
            diagnostics = diagnostics,
        )
    }

    private fun <T : DType, V> graphInput(shape: Shape, dtype: KClass<T>): Tensor<T, V> = VoidOpsTensor(
        object : TensorData<T, V> {
            override val shape: Shape = shape
            // Unreadable, like the shape-only results of VoidTensorOps: an eager shortcut that
            // reads its input (RMSNormalization's fused path) must not bake this input in as a constant.
            override fun get(vararg indices: Int): V = throw UnsupportedOperationException("graph input has no values")
            override fun set(vararg indices: Int, value: V) {}
        },
        dtype,
    )
}
