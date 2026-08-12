package sk.ainet.models.smollm2

import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Tag
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Exercises [SmolLm2ExportHarness.export] against the real SmolLM2-135M-Instruct
 * checkpoint and pins the shape of the emitted redecode graph (transformers#305):
 * a single `1x{seq}xi32 -> {seq}xi32` function (the DSL argMax tail — see the
 * harness doc), weights externalized under scope "model".
 *
 * Gated like the #272 reproducer spike: skips cleanly unless SMOLLM2_GGUF points
 * at an existing `.gguf`.
 */
@Tag("integration")
class SmolLm2ExportHarnessTest {
    @Test
    fun exportsRedecodeGraphWithArgMaxTail() {
        val path = System.getenv("SMOLLM2_GGUF")?.trim().orEmpty()
        assumeTrue(path.isNotEmpty() && File(path).isFile) {
            "[skip] SMOLLM2_GGUF not set (or not an existing file) — skipping SmolLM2 export harness test."
        }

        val outDir = System.getenv("SMOLLM2_TEST_OUT_DIR") ?: "build/mlir-test"
        val seq = 24
        val result = SmolLm2ExportHarness.export(gguf = path, outDir = outDir, seq = seq, bf16 = true)

        assertEquals(seq, result.seq)
        assertTrue(File(result.mlirPath).isFile, "MLIR file not written: ${result.mlirPath}")
        assertTrue(File(result.safetensorsPath).isFile, "safetensors file not written: ${result.safetensorsPath}")
        // SmolLM2-135M has 393 distinct weight tensors after tied-embedding dedup.
        assertEquals(393, result.externalParamCount)
        // bf16 halves the ~621 MiB dense-FP32 floor for this checkpoint.
        assertTrue(result.weightMiB in 300..320, "unexpected bf16 archive size: ${result.weightMiB} MiB")

        val mlir = File(result.mlirPath).readText()
        assertTrue(
            mlir.contains("func.func @smollm2(%arg0: tensor<1x${seq}xi32>) -> (tensor<${seq}xi32>)"),
            "unexpected function signature in ${result.mlirPath}",
        )
    }
}
