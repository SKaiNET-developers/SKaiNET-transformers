package sk.ainet.models.qwen

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.jupiter.api.Assumptions.assumeTrue

/**
 * Real-checkpoint export smoke for [QwenExportHarness] (SKaiNET-transformers#411), the
 * FunctionGemma `FunctionGemmaExportDumpTest` pattern: run the harness against a real GGUF and
 * check the emitted MLIR's arg/result counts against the RAW (pre-host-gather) formula the trace
 * actually produces — `3 + 2*nLayers` for `qwen_with_past` (token, cos, sin, then K/V per layer;
 * no separate `emb` arg at this stage, see [QwenExportHarness]'s class doc on why that differs
 * from [QwenKvContract]'s arg lists, which describe the contract AFTER a later host-gather
 * rewrite) — not [QwenKvContract]'s own lists, to keep this test honest about what this harness
 * alone actually emits.
 *
 * Qwen3-0.6B only: Qwen2.5-0.5B-Instruct's attention bias does not export correctly today (a
 * verified, documented SKaiNET core tracer gap — see [QwenExportHarness]'s "Known gap" doc);
 * exercising it here would just pin a known-broken shape.
 *
 * Skips (JUnit assumption) without the checkpoint on disk — set `QWEN3_GGUF` to a Qwen3 GGUF path.
 * Run with:
 *   QWEN3_GGUF=/path/to/Qwen3-0.6B-Q8_0.gguf ./gradlew :llm-inference:qwen:jvmTest --tests "*QwenExportDumpTest*"
 */
class QwenExportDumpTest {
    private val gguf: String? = System.getenv("QWEN3_GGUF")

    private fun assumeCheckpoint() {
        assumeTrue(gguf != null && File(gguf).exists(), "QWEN3_GGUF not set or file missing -- skipping (see class doc)")
    }

    @Test
    fun exportAll_producesRawPreHostGatherArgCounts() {
        assumeCheckpoint()
        val outDir = File(System.getProperty("java.io.tmpdir"), "qwen3-export-dump").absolutePath
        val r = QwenExportHarness.exportAll(gguf!!, outDir, seq = 64, chunk = 32)
        val nLayers = r.arch.nLayers

        // qwen_prefill_at: tokens, select -- no emb, no K/V inputs (stateless one-pass prefill).
        val prefillAtMlir = r.prefillAtMlir.readText()
        checkArgCount(prefillAtMlir, QwenKvContract.FN_PREFILL_AT, 2)
        checkResultCount(prefillAtMlir, QwenKvContract.FN_PREFILL_AT, 2 * nLayers + 1)

        // qwen_prefill_with_past: tokens, cos, sin, [K,V]*nLayers, mask, select.
        val chunkMlir = r.prefillWithPastMlir.readText()
        checkArgCount(chunkMlir, QwenKvContract.FN_PREFILL_WITH_PAST, 3 + 2 * nLayers + 1 + 1)
        checkResultCount(chunkMlir, QwenKvContract.FN_PREFILL_WITH_PAST, 2 * nLayers + 1)
        assertTrue(chunkMlir.contains("x?x"), "qwen_prefill_with_past: dynamic past-cache dim must be present")

        // qwen_with_past: token, cos, sin, [K,V]*nLayers. Dynamic self-cache by design (one vmfb
        // serves every decode position).
        val withPastMlir = r.withPastMlir.readText()
        checkArgCount(withPastMlir, QwenKvContract.FN_WITH_PAST, 3 + 2 * nLayers)
        checkResultCount(withPastMlir, QwenKvContract.FN_WITH_PAST, 2 * nLayers + 1)
        assertTrue(withPastMlir.contains("x?x${r.arch.headDim}"), "qwen_with_past: dynamic self-cache dim must be present")

        // No bias leaking into the signature for this model (Qwen3 has none) -- the one axis a
        // real SKaiNET tracer gap (see class doc) would silently break if it regressed further.
        assertEquals(3 + 2 * nLayers, argCount(withPastMlir, QwenKvContract.FN_WITH_PAST), "qwen_with_past arg count must match exactly (no leaked bias args)")

        // manifest.json: contract fields present and internally consistent with the derived arch.
        val manifest = r.manifest.readText()
        assertTrue(manifest.contains("\"contract\": \"${QwenKvContract.CONTRACT_ID}\""))
        assertTrue(manifest.contains("\"nLayers\": $nLayers"))
        assertTrue(manifest.contains("\"qkNorm\": true"), "Qwen3 must report qkNorm=true")
        assertTrue(manifest.contains("\"attnBias\": false"), "Qwen3 must report attnBias=false")
        assertEquals(QwenKvContract.manifestJson(r.arch, 32), manifest, "manifest.json must equal the pure contract emission for the same arch")

        println(
            "Qwen3 export OK: prefillAt=${sizeMiB(outDir, "qwen-prefill-at.safetensors")}MiB " +
                "prefillWithPast=${sizeMiB(outDir, "qwen-prefill-with-past.safetensors")}MiB " +
                "withPast=${sizeMiB(outDir, "qwen-with-past.safetensors")}MiB manifest=${r.manifest}",
        )
    }

    private fun sizeMiB(dir: String, name: String) = File(dir, name).length() / (1 shl 20)

    private fun argCount(mlir: String, fn: String): Int {
        val m = Regex("""func\.func @$fn\(([^)]*)\)""", RegexOption.DOT_MATCHES_ALL).find(mlir)
            ?: error("no func.func @$fn( found")
        return m.groupValues[1].split(",").count { it.isNotBlank() }
    }

    private fun checkArgCount(mlir: String, fn: String, expected: Int) =
        assertEquals(expected, argCount(mlir, fn), "$fn: arg count")

    private fun checkResultCount(mlir: String, fn: String, expected: Int) {
        val m = Regex("""func\.func @$fn\([^)]*\) -> \(([^)]*)\)""", RegexOption.DOT_MATCHES_ALL).find(mlir)
            ?: error("no func.func @$fn(...) -> (...) found")
        assertEquals(expected, m.groupValues[1].split(",").count { it.isNotBlank() }, "$fn: result count")
    }
}
