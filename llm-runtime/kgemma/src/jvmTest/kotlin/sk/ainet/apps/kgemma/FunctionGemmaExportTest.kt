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

        val st = File(r.safetensorsPath)
        assertTrue(st.exists() && st.length() > 0L, "bf16 safetensors written")
        // bf16 (2 bytes) halves the f32 archive -> ~831 MiB, matching the proven f16 .irpa size.
        assertTrue(r.weightMiB in 700..900, "bf16 weight archive ~831 MiB, got ${r.weightMiB} MiB")
    }
}
