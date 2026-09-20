package sk.ainet.models.functiongemma

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * G2a — real-checkpoint export smoke for the FunctionGemma module (whisper `WhisperExportTest`
 * pattern): run [FunctionGemmaExportHarness.exportAll] and assert the emitted MLIR signatures
 * match [FunctionGemmaContract]'s pinned arg/result orders, and `manifest.json` carries the
 * contract shapes. Skips (JUnit assumption) without the GGUF checkpoint or with too small a heap.
 *
 * Run with:
 *   ./gradlew :llm-inference:functiongemma:jvmTest --tests "*FunctionGemmaExportDumpTest*"
 */
class FunctionGemmaExportDumpTest {

    private val spec = FunctionGemmaSpec(gguf = FunctionGemmaFixture.gguf, seq = 24)

    @Test
    fun exportAll_emitsContractShapesAndManifest() {
        FunctionGemmaFixture.assumeRealCheckpointRunnable()
        val outDir = File(System.getProperty("java.io.tmpdir"), "functiongemma-export-dump").absolutePath
        val r = FunctionGemmaExportHarness.exportAll(spec, outDir)

        // ---- redecode (`gemma`): fully static, single arg -> single result ----
        val redecodeMlir = File(r.redecode.mlirPath).readText()
        assertTrue(!redecodeMlir.contains("tensor<?"), "gemma: dynamic shape leaked into the fixed redecode graph")
        assertTrue(redecodeMlir.contains("func.func @${FunctionGemmaContract.FN_REDECODE}("), "entry func present")
        assertTrue(redecodeMlir.contains("${spec.seq}xi32>"), "argMax+squeeze tail -> ${spec.seq}xi32")
        assertTrue(redecodeMlir.contains("xbf16>"), "bf16 weight globals present (default quant policy)")

        // ---- prefill: static seq, per-layer K/V outputs THEN tokens last (contract order) ----
        val prefillMlir = File(outDir, "gemma-prefill.mlir").readText()
        assertTrue(!prefillMlir.contains("tensor<?"), "gemma_prefill: must stay fully static (fixed seq)")
        checkArgCount(prefillMlir, FunctionGemmaContract.FN_PREFILL, FunctionGemmaContract.prefillArgs().size)
        checkResultCount(prefillMlir, FunctionGemmaContract.FN_PREFILL, FunctionGemmaContract.prefillOutputs(spec).size)

        // ---- with_past: token-step graph, dynamic self-cache seq dim by design ----
        val withPastMlir = File(outDir, "gemma-with-past.mlir").readText()
        assertTrue(withPastMlir.contains("x?x${spec.headDim}"), "gemma_with_past: dynamic self-cache dim must be present")
        checkArgCount(withPastMlir, FunctionGemmaContract.FN_WITH_PAST, FunctionGemmaContract.withPastArgs(spec).size)
        checkResultCount(withPastMlir, FunctionGemmaContract.FN_WITH_PAST, FunctionGemmaContract.withPastOutputs(spec).size)

        // ---- manifest.json: contract fields present and internally consistent ----
        val manifest = r.manifest.readText()
        assertTrue(manifest.contains("\"contractVersion\": ${FunctionGemmaContract.CONTRACT_VERSION}"))
        assertTrue(manifest.contains("\"nLayers\": ${spec.nLayers}"))
        assertEquals(FunctionGemmaContract.manifestJson(spec), manifest, "manifest.json must equal the pure contract emission for the same spec")

        println("G2a OK: redecode=${r.redecode.weightMiB}MiB prefill=${File(outDir, "gemma-prefill.safetensors").length() / (1 shl 20)}MiB " +
            "withPast=${File(outDir, "gemma-with-past.safetensors").length() / (1 shl 20)}MiB manifest=${r.manifest}")
    }

    @Test
    fun positionSelectedGraphs_emitContractShapes() {
        FunctionGemmaFixture.assumeRealCheckpointRunnable()
        val outDir = File(System.getProperty("java.io.tmpdir"), "functiongemma-export-dump-at").absolutePath

        // ---- gemma_at: (tokens 1xseq i32, select 1xseq f32) -> token 1xi32, fully static ----
        val r = FunctionGemmaExportHarness.exportRedecodeAt(spec, outDir)
        val atMlir = File(r.mlirPath).readText()
        assertTrue(!atMlir.contains("tensor<?"), "gemma_at: dynamic shape leaked into the fixed graph")
        assertTrue(atMlir.contains("func.func @${FunctionGemmaContract.FN_REDECODE_AT}("), "entry func present")
        assertTrue(atMlir.contains("tensor<1x${spec.seq}xf32>"), "one-hot select input present")
        assertTrue(!atMlir.contains("x${spec.seq}x262144xf32>") && !atMlir.contains("x${spec.seq}x262144xi32>"), "gemma_at: no seq x vocab logits/argmax scratch")
        checkArgCount(atMlir, FunctionGemmaContract.FN_REDECODE_AT, FunctionGemmaContract.selectArgs().size)
        checkResultCount(atMlir, FunctionGemmaContract.FN_REDECODE_AT, 1)

        // ---- gemma_prefill_at: (tokens seq i32, select 1xseq f32) -> per-layer K/V THEN token last ----
        val prefillAtMlir = File(outDir, "gemma-prefill-at.mlir").also { FunctionGemmaExportHarness.exportPrefillAt(spec, outDir) }.readText()
        assertTrue(!prefillAtMlir.contains("tensor<?"), "gemma_prefill_at: must stay fully static (fixed seq)")
        assertTrue(!prefillAtMlir.contains("x${spec.seq}x262144xf32>"), "gemma_prefill_at: no seq x vocab logits")
        checkArgCount(prefillAtMlir, FunctionGemmaContract.FN_PREFILL_AT, FunctionGemmaContract.selectArgs().size)
        checkResultCount(prefillAtMlir, FunctionGemmaContract.FN_PREFILL_AT, FunctionGemmaContract.prefillAtOutputs(spec).size)
    }

    @Test
    fun chunkPrefillWithPast_emitsContractShapes() {
        FunctionGemmaFixture.assumeRealCheckpointRunnable()
        val outDir = File(System.getProperty("java.io.tmpdir"), "functiongemma-export-dump-chunk").absolutePath
        FunctionGemmaExportHarness.exportPrefillWithPast(spec, outDir)
        val mlir = File(outDir, "gemma-prefill-with-past.mlir").readText()
        val c = FunctionGemmaContract.DEFAULT_CHUNK
        assertTrue(mlir.contains("func.func @${FunctionGemmaContract.FN_PREFILL_WITH_PAST}("), "entry func present")
        assertTrue(mlir.contains("x?x${spec.headDim}"), "dynamic past cache dim must be present")
        assertTrue(Regex("tensor<1x\\d+x${c}x\\?xf32>").containsMatchIn(mlir), "additive mask inputs [1,nHeads,C,?] present")
        assertTrue(mlir.contains("tensor<${c}x${spec.headDim}xf32>"), "per-position cos/sin tables present")
        checkArgCount(mlir, FunctionGemmaContract.FN_PREFILL_WITH_PAST, FunctionGemmaContract.prefillWithPastArgs(spec).size)
        checkResultCount(mlir, FunctionGemmaContract.FN_PREFILL_WITH_PAST, FunctionGemmaContract.prefillWithPastOutputs(spec).size)
    }

    private fun checkArgCount(mlir: String, func: String, expected: Int) {
        val sig = signatureLine(mlir, func)
        val argsPart = sig.substringAfter("@$func(").substringBefore(") ->")
        val args = Regex("tensor<[^>]+>").findAll(argsPart).count()
        assertEquals(expected, args, "$func: arg count\n$sig")
    }

    private fun checkResultCount(mlir: String, func: String, expected: Int) {
        val sig = signatureLine(mlir, func)
        val resultsPart = sig.substringAfter("-> ")
        val results = Regex("tensor<[^>]+>").findAll(resultsPart).count()
        assertEquals(expected, results, "$func: result count\n$sig")
    }

    private fun signatureLine(mlir: String, func: String): String =
        mlir.lineSequence().firstOrNull { it.contains("func.func") && it.contains("@$func(") }
            ?: error("$func: no func.func signature line found")
}
