package sk.ainet.models.bert

import kotlinx.coroutines.runBlocking
import sk.ainet.compile.hlo.toStableHlo
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.safetensors.SafeTensorsParametersLoader
import java.io.File
import java.nio.file.Path
import kotlin.io.path.readText
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Export harness for the on-box semantic embedder cartridge: bakes the real
 * sentence-transformers MiniLM weights into the DSL BERT encoder and emits token-level
 * StableHLO graphs at fixed sequence buckets (weights folded to constants — the same
 * pattern as the Moonshine v2 cartridge exports). Pooling/L2 stay host-side (the runtime
 * does the same), so the graphs output `[seq, hidden]`.
 *
 * Gated on MINILM_DIR (snapshot dir); MINILM_MLIR_OUT selects the output directory.
 * Also writes reference token ids + eager `encode` vectors for cross-target verification.
 */
class MiniLmExportHarness {

    @Test
    fun exportBuckets() {
        val dir = System.getenv("MINILM_DIR") ?: run {
            println("SKIP exportBuckets: set MINILM_DIR"); return
        }
        val out = File(System.getenv("MINILM_MLIR_OUT") ?: "build/build-mlir/minilm")
        out.mkdirs()
        val modelDir = Path.of(dir)

        val ctx = DirectCpuExecutionContext()
        val config = BertConfigParser.parse(modelDir.resolve("config.json").readText(), null)
        val loaders = listOf(
            SafeTensorsParametersLoader(sourceProvider = {
                JvmRandomAccessSource.open(modelDir.resolve("model.safetensors").toString())
            }),
        )
        val tensors = runBlocking {
            BertNetworkLoader.loadWeightTensors(loaders, ctx, sk.ainet.lang.types.FP32::class)
        }
        val runtime = createBertEncoderRuntime(config, tensors, ctx)
        val tokenizer = HuggingFaceTokenizer.fromVocabTxt(modelDir.resolve("vocab.txt").readText())

        for (seq in listOf(8, 16, 24, 32)) {
            val tape = runtime.exportTape(seqLen = seq)
            val graph = tape.toComputeGraph(synthesizeExternalInputs = true, embedConstants = true)
            val mlir = toStableHlo(graph, "minilm_encoder").content
            assertTrue(mlir.contains("stablehlo.constant"), "weights must fold to constants")
            val argCount = Regex("""%arg\d+""").findAll(mlir.substringBefore(") ->"))
                .map { it.value }.toSet().size
            assertEquals(1, argCount, "only the token-ids input should remain (seq=$seq)")
            val f = File(out, "minilm_s$seq.mlir")
            f.writeText(mlir)
            println("WROTE_MLIR ${f.absolutePath} (${mlir.lines().size} lines)")
        }

        // Reference probes: eager encode (unpadded, mask-free) — the quality baseline the
        // padded fixed-bucket vmfbs are compared against (cosine).
        val probes = listOf(
            "switch to the next channel",
            "make the tv silent",
            "i need some quiet time",
            "show me whats playing tonight",
        )
        File(out, "probes.txt").writeText(probes.joinToString("\n"))
        for ((i, p) in probes.withIndex()) {
            val ids = tokenizer.encode(p)
            File(out, "probe${i}_ids.txt").writeText(ids.joinToString(","))
            val vec = runtime.encode(ids)
            val bb = java.nio.ByteBuffer.allocate(vec.size * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN)
            for (v in vec) bb.putFloat(v)
            File(out, "probe${i}_ref.bin").writeBytes(bb.array())
            println("probe$i \"$p\" ids=${ids.size} dim=${vec.size}")
        }
    }
}
