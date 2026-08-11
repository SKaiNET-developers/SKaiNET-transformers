package sk.ainet.apps.kgemma

import java.io.File
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Phase-5 host verification: `quantizeInt8` emits the 2D matmul weights as per-row int8 in the
 * compiled graph — `tensor<rows x cols x i8>` global + a `tensor<rows x f32>` scale, dequant'd in
 * graph (`convert i8->f32` + `broadcast_in_dim` scale + `multiply`) — while norms stay bf16, and the
 * archive halves vs bf16 (a real RAM win on the 1.9 GB board). (iree-compile acceptance + numeric
 * quality of per-row int8 from Q5_K + the actual speed are on-board.)
 *
 * Skips without the GGUF or with too small a heap. 12g is the module default heap — override
 * with -PkgemmaTestMaxHeap. Run with:
 *   ./gradlew -PuseLocalSkainet=true \
 *     :llm-runtime:kgemma:jvmTest --tests "*FunctionGemmaInt8QuantTest*"
 */
class FunctionGemmaInt8QuantTest {
    private val gguf = FunctionGemmaFixture.gguf

    @Test
    fun int8_quant_emits_i8_weights_and_halves_archive() {
        FunctionGemmaFixture.assumeRealCheckpointRunnable()
        val out = File(System.getProperty("java.io.tmpdir"), "gemma-int8").absolutePath
        val r = FunctionGemmaExport.export(gguf, out, seq = 16, quantizeInt8 = true)
        val mlir = File(out, "gemma-gen.mlir").readText()

        val i8Globals = Regex("""util\.global private @\w+ = [^\n]*: tensor<\d+x\d+xi8>""").findAll(mlir).count()
        val scaleGlobals = Regex("""util\.global private @\w+_scale = [^\n]*: tensor<\d+xf32>""").findAll(mlir).count()
        val hasDequant = mlir.contains("stablehlo.broadcast_in_dim") && mlir.contains("stablehlo.multiply") &&
            Regex("""stablehlo\.convert[^\n]*i8[^\n]*f32""").containsMatchIn(mlir)
        val normsBf16 = mlir.contains("xbf16>")
        val stSize = File(out, "gemma.safetensors").length() / (1024 * 1024)
        println("[int8] weightMiB=${r.weightMiB} safetensorsMiB=$stSize i8Globals=$i8Globals scaleGlobals=$scaleGlobals dequant=$hasDequant normsBf16=$normsBf16")

        assertTrue(i8Globals > 100, "expected the ~126 2D matmul weights as int8, got $i8Globals")
        assertTrue(i8Globals == scaleGlobals, "each int8 weight needs a per-row scale global ($i8Globals vs $scaleGlobals)")
        assertTrue(hasDequant, "in-graph dequant (convert i8->f32 + broadcast scale + multiply) missing")
        assertTrue(normsBf16, "1-D norm globals should stay bf16")
        // int8 weights ~1 B/elem + tiny scales; the deduped bf16 archive is ~512 MiB (#260 removed
        // the duplicated tied embedding), so int8 ~= half of that (~256 MiB).
        assertTrue(r.weightMiB in 200..350, "int8 archive should be ~half the deduped bf16 512 MiB (got ${r.weightMiB})")
    }
}
