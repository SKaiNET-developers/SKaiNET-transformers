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
        // Inputs are bound to the tape context: a bare VoidOpsTensor runs operator overloads
        // (`x * x` in RMSNorm) on its own untraced ops, and the result is captured as a constant.
        val tapeCtx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val roles = LinkedHashMap<String, String>()
        fun input(role: String, shape: Shape): Tensor<T, V> =
            graphInput<T, V>(shape, dtype).bind(tapeCtx).also { roles[tapeCtx.session.refOf(it as Tensor<*, *>).id] = role }

        val kvShape = Shape(1, spec.nKVHeads, spec.past, spec.headDim)
        val ropeShape = Shape(1, spec.headDim)
        val embedded = input("embedded", Shape(1, spec.hidden))
        val global = RopeTables(input("cosGlobal", ropeShape), input("sinGlobal", ropeShape))
        val rope = if (spec.slidingRopeBase) {
            RopeCosSin(global, RopeTables(input("cosSliding", ropeShape), input("sinSliding", ropeShape)))
        } else {
            RopeCosSin(global)
        }
        val inputs = DecodeStepInputs(
            embedded, rope,
            List(spec.nLayers) { input("l$it.k", kvShape) },
            List(spec.nLayers) { input("l$it.v", kvShape) },
        )

        var out: KvStepOutput<T, V>? = null
        val tape = tapeCtx.record {
            val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("DecodeStepExport: no tape")
            Execution.tapeStack.pushTape(ct)
            try {
                out = step(inputs, this as ExecutionContext)
            } finally {
                Execution.tapeStack.popTape()
            }
        }.first
        val result = out!!
        val outRoles = HashMap<String, String>()
        fun outRole(t: Tensor<T, V>, role: String) { outRoles[tapeCtx.session.refOf(t as Tensor<*, *>).id] = role }
        outRole(result.logits, "logits")
        result.selfK.forEachIndexed { i, t -> outRole(t, "l$i.k") }
        result.selfV.forEachIndexed { i, t -> outRole(t, "l$i.v") }

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
