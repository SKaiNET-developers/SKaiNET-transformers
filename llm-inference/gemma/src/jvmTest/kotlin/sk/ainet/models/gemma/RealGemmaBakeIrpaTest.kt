package sk.ainet.models.gemma

import kotlinx.coroutines.runBlocking
import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import sk.ainet.compile.hlo.ConstantMaterializationPolicy
import sk.ainet.compile.hlo.StableHloConverterFactory
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.irpa.IrpaWriter
import sk.ainet.io.model.QuantPolicy
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
 * Bake the REAL FunctionGemma-270M to a self-contained IREE artifact:
 * fromWeights (real topology + real weights) -> trace (embedConstants=true,
 * resolves the bound weights) -> StableHLO with ExternalAlways so every weight
 * becomes a `util.global.load` + ExternalParameterRef -> IrpaWriter writes the
 * .irpa. The emitted vmfb then takes ONLY the token input; weights resolve from
 * the archive via `iree-run-module --parameters=model=gemma.irpa`. No 361-arg
 * mapping, no host-side RoPE reproduction. Boxing-free (FloatArray) path.
 */
class RealGemmaBakeIrpaTest {
    @Test
    fun bakeRealGemmaToIrpa() = runBlocking {
        val path = "/home/miso/projects/coral/sl2610-voice-cc-kt/models/functiongemma-physical-ai-v10-Q5_K_M.gguf"
        val ctx = DirectCpuExecutionContext.create()
        val weights = Gemma4WeightLoader(
            randomAccessProvider = { JvmRandomAccessSource.open(path) },
            quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
        ).loadToMapStreaming<FP32, Float>(ctx, FP32::class)
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
            try { model.forward(input, this as ExecutionContext) } finally { Execution.tapeStack.popTape() }
        }.first
        // embedConstants=true: finalize resolves the bound weights (now stored
        // as primitive FloatArray, no .toList() boxing). ExternalAlways lifts
        // every weight into the .irpa.
        val graph = (tape as DefaultExecutionTape).toComputeGraph(
            synthesizeExternalInputs = true, embedConstants = true,
        )
        val module = StableHloConverterFactory
            .createBasic(ConstantMaterializationPolicy.ExternalAlways(scope = "model"))
            .convert(graph, "gemma")

        val outDir = File("/home/miso/projects/coral/build-mlir").apply { mkdirs() }
        File(outDir, "gemma-baked.mlir").writeText(module.content)

        val ext = module.externalParameters
        val funcArgs = module.content.lineSequence().firstOrNull { it.contains("func.func @gemma(") }
            ?.let { Regex("%arg\\d+").findAll(it).count() } ?: -1
        val globals = module.content.lineSequence().count { it.trimStart().startsWith("util.global ") }
        val totalBytes = ext.sumOf { it.source.sizeInBytes }
        println("EXTPARAMS ${ext.size} totalMiB=${totalBytes / (1024 * 1024)}")
        println("FUNCARGS $funcArgs UTILGLOBALS $globals MLIRlines=${module.content.lines().size}")

        val irpa = File(outDir, "gemma.irpa")
        SystemFileSystem.sink(Path(irpa.absolutePath)).buffered().use { sink ->
            IrpaWriter().write(ext, sink)
        }
        println("WROTE_IRPA ${irpa.absolutePath} sizeMiB=${irpa.length() / (1024 * 1024)}")
    }
}
