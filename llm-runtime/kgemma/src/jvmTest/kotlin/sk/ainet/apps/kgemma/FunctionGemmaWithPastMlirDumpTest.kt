package sk.ainet.apps.kgemma

import java.io.File
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Phase-2 structure/probe: trace `GemmaModel.forwardWithPast` to StableHLO via
 * `FunctionGemmaExport.exportWithPast` and confirm the `decoder_with_past` graph emits, then probe
 * the KEY open question — does the tracer produce a clean DYNAMIC self-cache seq dim (`1x1x?x256`)
 * so one board vmfb serves every decode position, or must we fall back to fixed-pad / sentinel-regex.
 *
 * Skips without the GGUF or with too small a heap. 12g is the module default heap — override
 * with -PkgemmaTestMaxHeap. Run with:
 *   ./gradlew -PuseLocalSkainet=true \
 *     :llm-runtime:kgemma:jvmTest --tests "*FunctionGemmaWithPastMlirDumpTest*"
 */
class FunctionGemmaWithPastMlirDumpTest {
    private val gguf = FunctionGemmaFixture.gguf

    @Test
    fun emits_with_past_graph_and_probes_dynamic_dim() {
        FunctionGemmaFixture.assumeRealCheckpointRunnable()
        val outDir = File(System.getProperty("java.io.tmpdir"), "gemma-wp-dump").absolutePath

        // Static probe (fixed past=7): the per-layer cache seq dim should appear concretely (1x1x7x256).
        val staticMlir = FunctionGemmaExport.exportWithPast(gguf, outDir, past = 7, dynamicPast = false, bf16 = true)
        val staticIn = staticMlir.contains("x7x256")
        val staticOut = staticMlir.contains("x8x256")
        println("[wp-dump] STATIC past=7: cache-in(x7x256)=$staticIn  cache-out(x8x256)=$staticOut  lines=${staticMlir.lines().size}")

        // Dynamic probe (sentinel-relax): BOTH the input caches AND the output/return caches must be
        // dynamic `?` — and crucially NOT the broken `1x1x0x256` a naive `-1` placeholder produces.
        val dynMlir = FunctionGemmaExport.exportWithPast(gguf, outDir, dynamicPast = true, bf16 = true)
        val ret = dynMlir.lineSequence().firstOrNull { it.trimStart().startsWith("return ") } ?: ""
        val inQ = dynMlir.contains("x?x256")
        val outQ = ret.contains("x?x256")
        val outBroken = ret.contains("x0x256")
        val sentinelLeaked = dynMlir.contains("x7919x") || dynMlir.contains("x7920x")
        println("[wp-dump] DYNAMIC: input `x?x256`=$inQ  return `x?x256`=$outQ  return-broken(`x0x256`)=$outBroken  sentinel-leaked=$sentinelLeaked")
        println("[wp-dump] DECISION: ${if (outQ && !outBroken) "sentinel-relax dynamic OK — one vmfb serves every position (iree-compile must confirm on board)" else "dynamic relax INCOMPLETE — inspect return sig; consider fixed-pad fallback"}")

        // Structure gate (independent of the dynamic question): the graph must emit with the func,
        // the bf16 external weight loads, and the argMax token tail.
        assertTrue(staticMlir.contains("func.func @gemma_with_past") || staticMlir.contains("@gemma_with_past"),
            "with_past func not emitted")
        assertTrue(staticMlir.contains("bf16"), "expected bf16 external weight globals")
        // Static: K/V correctly threaded (cache-in 7 -> cache-out 8).
        assertTrue(staticIn && staticOut, "static with_past must thread cache-in x7x256 -> cache-out x8x256")
        // Dynamic: caches relax to `?` on BOTH ends, with no leftover sentinel and no broken 0-dim.
        assertTrue(inQ && outQ, "dynamic with_past must relax input AND return caches to x?x256")
        assertTrue(!outBroken, "return caches must not be the broken x0x256 (naive -1 placeholder)")
        assertTrue(!sentinelLeaked, "sentinel dim leaked into the emitted MLIR")

        // Prefill graph: emits `func @gemma_prefill` with fixed seq=16 initial K/V outputs `1x1x16x256`.
        val prefill = FunctionGemmaExport.exportPrefill(gguf, outDir, seq = 16, bf16 = true)
        val prefIn = prefill.contains("x16x256")
        println("[wp-dump] PREFILL seq=16: func=${prefill.contains("@gemma_prefill")} initial-KV(x16x256)=$prefIn  lines=${prefill.lines().size}")
        assertTrue(prefill.contains("@gemma_prefill"), "prefill func not emitted")
        assertTrue(prefIn, "prefill must emit per-layer initial K/V x16x256")
    }
}
