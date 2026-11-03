package sk.ainet.models.lfm2.export

import sk.ainet.compile.hlo.ConstantMaterializationPolicy
import sk.ainet.compile.hlo.ConversionErrorPolicy
import sk.ainet.compile.hlo.StableHloConverterFactory
import sk.ainet.compile.hlo.StableHloModule
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.types.FP32
import sk.ainet.models.lfm2.*

/** Names describe the caller ABI; StableHLO specs retain the tracer's tensor identifiers. */
data class Lfm2TracedFunction(
    val module: StableHloModule,
    val inputNames: List<String>,
    val outputNames: List<String>,
    val operationCount: Int,
    val inputShapes: List<Shape>,
    val outputShapes: List<Shape>,
)

/** Trace explicit state with unreadable runtime data and fail on unsupported lowering. */
fun traceLfm2Function(
    functionName: String,
    inputs: LinkedHashMap<String, Shape>,
    constantPolicy: ConstantMaterializationPolicy = ConstantMaterializationPolicy.ExternalAlways(scope = "model"),
    forward: (Map<String, Tensor<FP32, Float>>, ExecutionContext) -> LinkedHashMap<String, Tensor<FP32, Float>>,
): Lfm2TracedFunction {
    require(functionName.matches(Regex("[A-Za-z_][A-Za-z0-9_]*")))
    require(inputs.isNotEmpty() && inputs.all { (name, shape) -> name.isNotBlank() && shape.dimensions.all { it > 0 } })
    val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
    val tensors = inputs.mapValuesTo(linkedMapOf()) { (_, shape) -> runtimeTensor(shape) }
    val inputIds = tensors.values.map { ctx.session.refOf(it).id }
    val (tape, outputs) = ctx.record { forward(tensors, this) }
    require(outputs.isNotEmpty() && outputs.keys.all { it.isNotBlank() })
    val outputIds = outputs.values.map { ctx.session.refOf(it).id }
    require(outputIds.distinct().size == outputIds.size) { "Each named result must have a distinct tensor identity" }
    val graph = (tape as DefaultExecutionTape).toComputeGraph(
        synthesizeExternalInputs = true, inputTensorIds = inputIds.toSet(), embedConstants = true,
    )
    fun producer(id: String): GraphNode = graph.nodes.single { node -> node.outputs.any { it.name == id } }
    val inputNodes = inputIds.map(::producer)
    val inputNodeIds = inputNodes.map { it.id }.toSet()
    val ordered = object : ComputeGraph by graph {
        override fun getOutputNodes(): List<GraphNode> = outputIds.map(::producer)
        override fun getTopologicalOrder(): List<GraphNode> =
            inputNodes + graph.getTopologicalOrder().filterNot { it.id in inputNodeIds }
    }
    val module = StableHloConverterFactory.createBasic(policy = constantPolicy, errorPolicy = ConversionErrorPolicy.STRICT)
        .convert(ordered, functionName)
    check(module.inputSpecs.map { it.name } == inputIds && module.outputSpecs.map { it.name } == outputIds) {
        "Export argument/result order differs from the declared LFM2 ABI"
    }
    return Lfm2TracedFunction(module, inputs.keys.toList(), outputs.keys.toList(),
        graph.nodes.count { it.operation.name != "input" && it.operation.name != "constant" },
        tensors.values.map { it.shape }, outputs.values.map { it.shape })
}

fun traceLfm2ShortConv(operation: ShortConvTensorOp, length: Int,
    policy: ConstantMaterializationPolicy = ConstantMaterializationPolicy.ExternalAlways(scope = "model"),
): Lfm2TracedFunction = traceConvolutionState(operation.width, operation.kernel, length, "short_conv", policy) { x, history, ctx ->
    operation.forward(x, history, ctx)
}

fun traceLfm2ConvBlock(operation: BlockTensorOp, length: Int,
    policy: ConstantMaterializationPolicy = ConstantMaterializationPolicy.ExternalAlways(scope = "model"),
): Lfm2TracedFunction = traceConvolutionState(operation.width, operation.kernel, length, "decoder_block", policy) { x, history, ctx ->
    val result = operation.forward(x, history, ctx)
    ShortConvTensorResult(result.output, result.nextHistory)
}

private fun traceConvolutionState(width: Int, kernel: Int, length: Int, name: String,
    policy: ConstantMaterializationPolicy,
    forward: (Tensor<FP32, Float>, Tensor<FP32, Float>, ExecutionContext) -> ShortConvTensorResult,
): Lfm2TracedFunction {
    require(width > 0 && kernel > 0 && length > 0)
    return traceLfm2Function(name, linkedMapOf("input" to Shape(length, width), "history" to Shape(width, kernel)), policy) { inputs, ctx ->
        val result = forward(inputs.getValue("input"), inputs.getValue("history"), ctx)
        linkedMapOf("output" to result.output, "next_history" to result.nextHistory)
    }
}

fun traceLfm2Attention(operation: AttentionTensorOp, past: Int, length: Int,
    policy: ConstantMaterializationPolicy = ConstantMaterializationPolicy.ExternalAlways(scope = "model"),
): Lfm2TracedFunction {
    require(past >= 0 && length > 0 && past.toLong() + length <= Int.MAX_VALUE)
    val shapes = linkedMapOf("input" to Shape(length, operation.width))
    if (past > 0) {
        shapes["past_key"] = Shape(operation.kvHeads, past, operation.headDim)
        shapes["past_value"] = Shape(operation.kvHeads, past, operation.headDim)
    }
    shapes["cos"] = Shape(length, operation.headDim)
    shapes["sin"] = Shape(length, operation.headDim)
    shapes["mask"] = Shape(length, past + length)
    val traced = traceLfm2Function("attention_block", shapes, policy) { inputs, ctx ->
        val r = operation.forward(inputs.getValue("input"), inputs["past_key"], inputs["past_value"],
            inputs.getValue("cos"), inputs.getValue("sin"), inputs.getValue("mask"), ctx)
        linkedMapOf("output" to r.output, "next_key" to r.nextKey, "next_value" to r.nextValue)
    }
    check(traced.module.content.contains("across dimensions = [2]")) { "Attention softmax must lower over the key axis" }
    return traced
}

fun traceLfm2Prefill(network: Lfm2PrefillNetwork, length: Int): Lfm2TracedFunction {
    val d = network.dimensions
    require(length in 1..d.contextLimit)
    val shapes = linkedMapOf("embeddings" to Shape(length, d.width), "cos" to Shape(length, d.headDim),
        "sin" to Shape(length, d.headDim), "mask" to Shape(length, length))
    for ((layer, kind) in d.layerTypes.withIndex()) if (kind == "conv") shapes["history_$layer"] = Shape(d.width, d.kernel)
    val traced = traceLfm2Function("lfm2_prefill", shapes) { inputs, ctx ->
        val histories = d.layerTypes.indices.filter { d.layerTypes[it] == "conv" }.associateWith { inputs.getValue("history_$it") }
        val r = network.forward(inputs.getValue("embeddings"), inputs.getValue("cos"), inputs.getValue("sin"), inputs.getValue("mask"), histories, ctx)
        linkedMapOf("logits" to r.logits).apply { putAll(r.states) }
    }
    check(traced.module.externalParameters.size == expectedModelWeights(d).size) { "Prefill must reference every source weight exactly once" }
    return traced
}

private fun runtimeTensor(shape: Shape): Tensor<FP32, Float> = VoidOpsTensor(object : TensorData<FP32, Float> {
    override val shape = shape
    override fun get(vararg indices: Int): Float = error("Runtime input data cannot be read while exporting")
    override fun set(vararg indices: Int, value: Float): Unit = error("Runtime input data cannot be written while exporting")
}, FP32::class)
