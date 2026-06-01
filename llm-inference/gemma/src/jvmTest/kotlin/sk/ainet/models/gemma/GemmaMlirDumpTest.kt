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

/** Dumps the lowered gemma3 StableHLO to a file for iree-compile (CPU/NEON). */
class GemmaMlirDumpTest {
    @Test
    fun dumpGemmaMlir() {
        val meta = Gemma4ModelMetadata(
            architecture = "gemma3",
            embeddingLength = 64,
            contextLength = 128,
            blockCount = 1,
            headCount = 2,
            kvHeadCount = 1,
            intermediateSize = 128,
            headDim = 32,
            globalHeadDim = 32,
            vocabSize = 32,
            slidingWindow = 64,
            kvSharedLayers = 0,
            layerTypes = listOf("full_attention"),
            ropeParametersFull = Gemma4RopeConfig(base = 10000.0f),
            ropeParametersSliding = Gemma4RopeConfig(base = 10000.0f),
            maxPositionEmbeddings = 128,
        )
        val model = gemmaNetwork<FP32, Float>(meta, FP32::class, maxInferenceLen = 8)
        val input = VoidOpsTensor(
            object : TensorData<FP32, Float> {
                override val shape = Shape(1, 4)
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
        val graph = (tape as DefaultExecutionTape).toComputeGraph(synthesizeExternalInputs = true)
        val mlir = sk.ainet.compile.hlo.toStableHlo(graph, "gemma").content

        val out = File(System.getProperty("gemmaMlirOut") ?: "/home/miso/projects/coral/build-mlir/gemma.mlir")
        out.parentFile?.mkdirs()
        out.writeText(mlir)
        println("WROTE_MLIR ${out.absolutePath} (${mlir.lines().size} lines)")
    }
}
