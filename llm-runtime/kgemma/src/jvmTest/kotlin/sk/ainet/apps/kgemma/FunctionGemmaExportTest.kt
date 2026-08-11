package sk.ainet.apps.kgemma

import java.io.File
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Integration: run the real FunctionGemma-270M compiled export and assert the emitted
 * StableHLO is the gemma-gen contract with the DSL argMax tail + bf16 externals.
 * Skips (JUnit assumption) when the GGUF checkpoint isn't present (CI) or the heap is too small.
 * 12g is the module default heap — override with -PkgemmaTestMaxHeap. Run with:
 *   ./gradlew -PuseLocalSkainet=true \
 *     :llm-runtime:kgemma:jvmTest --tests "*FunctionGemmaExportTest*"
 */
class FunctionGemmaExportTest {
    private val gguf = FunctionGemmaFixture.gguf

    @Test
    fun exports_gemma_gen_mlir_with_argmax_tail_and_bf16_externals() {
        FunctionGemmaFixture.assumeRealCheckpointRunnable()
        val outDir = File(System.getProperty("java.io.tmpdir"), "fgemma-export").absolutePath
        val r = FunctionGemmaExport.export(gguf = gguf, outDir = outDir, seq = 24, bf16 = true)
        println("EXPORT extParams=${r.externalParamCount} weightMiB=${r.weightMiB} mlir=${r.mlirPath}")

        assertTrue(r.externalParamCount > 100, "270M weights should lift to many external params")

        val mlir = File(r.mlirPath).readText()

        // Entry + argMax tail: func @gemma returns rank-1 i32 token ids for seq=24.
        assertTrue(mlir.contains("func.func @gemma("), "entry func @gemma present")
        assertTrue(mlir.contains("24xi32>"), "argMax+squeeze tail -> 24xi32")
        assertTrue(
            mlir.contains("stablehlo.iota") &&
                mlir.contains("stablehlo.maximum") &&
                mlir.contains("stablehlo.minimum"),
            "argMax lowering (iota + reduce-max + reduce-min) present",
        )

        // bf16 externals: weight globals are bf16 with a convert-on-load; no f32 weight globals remain.
        assertTrue(mlir.contains("xbf16>"), "bf16 weight globals present")
        assertTrue(mlir.contains("stablehlo.convert"), "convert-on-load present")
        val f32Global = Regex(
            """util\.global private @\w+ = #flow\.parameter\.named<[^>]*> : tensor<[0-9x]*xf32>""",
        ).find(mlir)
        assertTrue(f32Global == null, "no f32 weight globals should remain after the bf16 fold")

        // Tied-embedding dedup (#260): token_embd/output.weight are one tensor, so the trace must
        // externalize the 262153x640 embedding exactly ONCE (was twice = 77% of the archive).
        val embedGlobals = Regex("""util\.global private @\w+ = [^\n]*tensor<262153x640x""").findAll(mlir).count()
        kotlin.test.assertEquals(
            1, embedGlobals,
            "tied 262153x640 embedding must be externalized exactly once (#260), got $embedGlobals globals",
        )

        val st = File(r.safetensorsPath)
        assertTrue(st.exists() && st.length() > 0L, "bf16 safetensors written")
        // bf16 (2 bytes) halves the f32 archive; with the tied embedding deduplicated (#260) the
        // 268.3M-param model lands at ~512 MiB (was ~831 MiB with the duplicate).
        assertTrue(r.weightMiB in 450..600, "bf16 deduped weight archive ~512 MiB, got ${r.weightMiB} MiB")
    }
}
