package sk.ainet.models.gemma

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.types.FP32
import sk.ainet.tape.Execution
import java.io.File
import kotlin.test.Test

/**
 * Dumps the REAL FunctionGemma-270M gemma3 config (640/18/heads4/kv1/headDim256/
 * ffn2048/vocab262153/sliding512) to StableHLO for iree-compile, AND dumps the
 * ordered graph input nodes (id, shape) — the data we need to map gguf weights
 * onto the vmfb function args (trace input nodes carry tensor ids, not names).
 */
class RealGemmaMlirDumpTest {
    @Test
    fun dumpRealGemmaMlir() {
        val meta = GemmaModelMetadata(
            architecture = "gemma3",
            embeddingLength = 640,
            contextLength = 512,
            blockCount = 18,
            headCount = 4,
            kvHeadCount = 1,
            intermediateSize = 2048,
            headDim = 256,
            globalHeadDim = 256,
            vocabSize = 262153,
            slidingWindow = 512,
            kvSharedLayers = 0,
            layerTypes = List(18) { "full_attention" },
            ropeParametersFull = GemmaRopeConfig(base = 1000000.0f),
            ropeParametersSliding = GemmaRopeConfig(base = 10000.0f),
            maxPositionEmbeddings = 512,
        )
        val seqLen = (System.getProperty("seqLen") ?: "4").toInt()
        val model = gemmaNetwork<FP32, Float>(meta, FP32::class, maxInferenceLen = seqLen)
        val input = VoidOpsTensor(
            object : TensorData<FP32, Float> {
                override val shape = Shape(1, seqLen)
                override fun get(vararg indices: Int): Float = 0.0f
                override fun set(vararg indices: Int, value: Float) {}
            },
            FP32::class,
        )
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val tape = ctx.record {
            val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
            Execution.tapeStack.pushTape(ct)
            try {
                model.forward(input, this as ExecutionContext)
            } finally {
                Execution.tapeStack.popTape()
            }
        }.first
        // embedConstants=false: every leaf weight (incl. block-level RMSNorm
        // weights, which otherwise self-resolve to default zeros/ones and get
        // baked as constants) becomes an external input -> a complete
        // weights-as-args interface to bind real gguf values onto.
        val embed = (System.getProperty("embedConstants") ?: "false").toBoolean()
        val graph = (tape as DefaultExecutionTape).toComputeGraph(
            synthesizeExternalInputs = true,
            embedConstants = embed,
        )

        // Ordered graph input nodes = the vmfb function-arg order. Dump id+shape
        // so we can build the gguf-tensor -> arg map.
        val inputs = graph.nodes.filter { it.operationName.equals("input", ignoreCase = true) }
        println("INPUTS ${inputs.size}")
        inputs.forEachIndexed { i, n ->
            val shp = n.outputs.firstOrNull()?.shape
            println("  ARG$i id=${n.id} shape=$shp")
        }
        println("NODES ${graph.nodes.size}")

        val mlir = sk.ainet.compile.hlo.toStableHlo(graph, "gemma").content
        val unsupported = mlir.lines().count { it.contains("Unsupported op", ignoreCase = true) }
        val arity = mlir.lines().count { it.contains("arity", ignoreCase = true) }
        println("MLIR lines=${mlir.lines().size} unsupported=$unsupported arity=$arity")
        val out = File(System.getProperty("gemmaMlirOut") ?: "build/build-mlir/gemma-real.mlir")
        out.parentFile?.mkdirs()
        out.writeText(mlir)
        println("WROTE_MLIR ${out.absolutePath}")
    }
}
