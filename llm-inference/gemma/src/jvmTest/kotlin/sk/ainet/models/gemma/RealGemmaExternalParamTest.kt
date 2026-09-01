package sk.ainet.models.gemma

import org.junit.jupiter.api.Tag
import kotlinx.coroutines.runBlocking
import sk.ainet.compile.hlo.ConstantMaterializationPolicy
import sk.ainet.compile.hlo.StableHloConverterFactory
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.io.JvmRandomAccessSource
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
 * The production weight path: build the REAL FunctionGemma network via
 * GemmaNetworkLoader.fromWeights (correct auto-detected topology + real weights
 * bound to params), trace it, lower with the ExternalAlways materialization
 * policy so every weight is lifted to a `util.global.load` + ExternalParameterRef
 * (keyed by tensor name). Reports: does the real-topology graph lower cleanly,
 * how many external params, and are they keyed by gguf-style names — the input
 * to the .irpa packager + iree-compile --iree-opt-import-parameters.
 */
@Tag("integration")
class RealGemmaExternalParamTest {
    @Test
    fun externalizeRealGemmaWeights() = runBlocking {
        val path = FunctionGemmaFixture.gguf
        val ctx = DirectCpuExecutionContext.create()
        val weights = GemmaWeightLoader(
            randomAccessProvider = { JvmRandomAccessSource.open(path) },
            weightForm = GEMMA_DEQUANTIZE_ALL,
        ).loadToMapStreaming<FP32, Float>(ctx, FP32::class)

        // Correct topology (qkNorm/sandwich/PLE/softcap auto-detected) + real weights.
        val model = GemmaNetworkLoader.fromWeights(ctx, weights, FP32::class)

        val seqLen = (System.getProperty("seqLen") ?: "4").toInt()
        val input = VoidOpsTensor(
            object : TensorData<FP32, Float> {
                override val shape = Shape(1, seqLen)
                override fun get(vararg indices: Int): Float = 0.0f
                override fun set(vararg indices: Int, value: Float) {}
            },
            FP32::class,
        )
        val tapeCtx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val tape = tapeCtx.record {
            val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
            Execution.tapeStack.pushTape(ct)
            try {
                model.forward(input, this as ExecutionContext)
            } finally {
                Execution.tapeStack.popTape()
            }
        }.first
        // embedConstants=false: weights become func ARGS (shapes only, no
        // boxing). Baking real weights via finalize().toList() OOMs at real
        // vocab (262153x640 -> ~2.7GB boxed List<Float>); the zero-copy .irpa
        // bake path needs a FileBacked mmap BufferHandle (not yet in SKaiNET).
        // Weights-as-args is the deployable interface: bind weights at runtime.
        val graph = (tape as DefaultExecutionTape).toComputeGraph(
            synthesizeExternalInputs = true, embedConstants = false,
        )

        val module = StableHloConverterFactory.createBasic().convert(graph, "gemma")
        val mlir = module.content
        val inputs = graph.nodes.filter { it.operationName.equals("input", ignoreCase = true) }
        println("INPUTS ${inputs.size}")
        inputs.take(14).forEach { n -> println("  ARG id=${n.id} shape=${n.outputs.firstOrNull()?.shape}") }
        val ext = module.externalParameters
        println("EXTPARAMS ${ext.size}")
        val unsupported = mlir.lines().count { it.contains("Unsupported op", ignoreCase = true) }
        val arity = mlir.lines().count { it.contains("arity", ignoreCase = true) }
        val globals = mlir.lines().count { it.trimStart().startsWith("util.global") }
        println("MLIR lines=${mlir.lines().size} bytes=${mlir.length} unsupported=$unsupported arity=$arity util.global=$globals")
        val out = File("/home/miso/projects/coral/build-mlir/gemma-ext.mlir")
        out.parentFile?.mkdirs()
        out.writeText(mlir)
        println("WROTE_MLIR ${out.absolutePath}")
    }
}
