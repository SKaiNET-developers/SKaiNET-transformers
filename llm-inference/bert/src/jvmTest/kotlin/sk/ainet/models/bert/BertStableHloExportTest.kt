package sk.ainet.models.bert

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.compile.hlo.toStableHlo
import java.io.File
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * StableHLO export smoke test: the DSL BERT encoder traces to a ComputeGraph
 * that lowers to StableHLO text (the entry point of the MLIR/IREE path).
 * Compilation/execution with iree-compile is out of scope here — this gates
 * that the network definition stays export-clean.
 */
class BertStableHloExportTest {

    private val ctx = DirectCpuExecutionContext()

    @Test
    fun bertEncoderExportsToStableHlo() {
        val cfg = BertSyntheticFixtures.tinyConfig(layers = 1)
        val tensors = BertSyntheticFixtures.weightTensors(ctx, cfg)
        val runtime = createBertEncoderRuntime(cfg, tensors, ctx)

        val tape = runtime.exportTape(seqLen = 4)
        val graph = tape.toComputeGraph(synthesizeExternalInputs = true)
        val mlir = toStableHlo(graph, "bert_encoder").content

        assertTrue(mlir.contains("module"), "no MLIR module emitted")
        assertTrue(mlir.contains("func.func"), "no function emitted")
        // The ops that carry BERT's structure must survive lowering:
        assertTrue(mlir.contains("stablehlo.gather") || mlir.contains("\"stablehlo.gather\""), "no gather (word embeddings) in export")
        assertTrue(mlir.contains("stablehlo.dot") || mlir.contains("stablehlo.dot_general"), "no matmul in export")
        assertTrue(mlir.contains("stablehlo.add"), "no add in export")

        val out = File(System.getProperty("bertMlirOut") ?: "build/build-mlir/bert_encoder.mlir")
        out.parentFile?.mkdirs()
        out.writeText(mlir)
        println("WROTE_MLIR ${out.absolutePath} (${mlir.lines().size} lines)")
    }
}
