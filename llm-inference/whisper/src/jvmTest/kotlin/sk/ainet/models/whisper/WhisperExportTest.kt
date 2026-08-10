package sk.ainet.models.whisper

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * G2a — StableHLO export smoke for the whisper contract-v2 triple.
 *
 * Runs [WhisperExportHarness.export] (real weights, env-gated on
 * WHISPER_SAFETENSORS; output dir WHISPER_EXPORT_OUT, default
 * `build/whisper-export`) and asserts the emitted MLIR is compile-clean:
 *
 *  - fully static (no `tensor<?`), no `stablehlo.custom_call`
 *  - `whisper_encoder`: 1 arg `[1,80,400]` f32 → 1 result `[1,200,384]` f32
 *  - `whisper_prefill`: 2 args (promptIds `[4]` i32, feat `[1,200,384]`) →
 *    1+16 results (logits + per layer selfK/selfV `[1,48,384]`,
 *    crossK/crossV `[1,200,384]`) in the pinned contract order
 *  - `whisper_step`: 4+16 args → 1+8 results, pinned order
 *  - one merged `params.irpa` (keys deduped, byte-identity asserted inside
 *    the harness merge)
 */
class WhisperExportTest {

    private val cfg = WhisperExportHarness.cfg
    private val maxP = WhisperExportHarness.MAX_P

    @Test
    fun exportContractV2() {
        if (WhisperExportHarness.snapshotDir() == null) {
            println("SKIP: WHISPER_SAFETENSORS not set"); return
        }
        val out = WhisperExportHarness.exportOutDir()
        val r = WhisperExportHarness.export(out)

        val fSk = "tensor<1x${maxP}x${cfg.dim}xf32>"
        val fCross = "tensor<1x${cfg.audioCtx}x${cfg.dim}xf32>"

        // ---- encoder ----
        checkMlir(r.encoderMlir, "whisper_encoder",
            expectedArgs = listOf("tensor<1x${cfg.nMels}x${cfg.melFrames}xf32>"),
            expectedResults = listOf(fCross),
        )

        // ---- prefill ----
        checkMlir(r.prefillMlir, "whisper_prefill",
            expectedArgs = listOf("tensor<4xi32>", fCross),
            expectedResults = listOf("tensor<1x4x${cfg.vocabSize}xf32>") +
                (0 until cfg.decoderLayers).flatMap { listOf(fSk, fSk, fCross, fCross) },
        )

        // ---- step ----
        checkMlir(r.stepMlir, "whisper_step",
            expectedArgs = listOf(
                "tensor<1xi32>", "tensor<1xi32>",
                "tensor<1x1x1x${maxP}xf32>", "tensor<1x${maxP}x1xf32>",
            ) + (0 until cfg.decoderLayers).flatMap { listOf(fSk, fSk, fCross, fCross) },
            expectedResults = listOf("tensor<1x1x${cfg.vocabSize}xf32>") +
                (0 until cfg.decoderLayers).flatMap { listOf(fSk, fSk) },
        )

        assertTrue(r.paramsIrpa.length() > 100L * (1 shl 20), "params.irpa suspiciously small")
        assertTrue(r.mergedParamCount <= r.prefillParamCount + r.stepParamCount)
        assertTrue(r.manifest.readText().contains("\"contractVersion\": 2"))
        println("G2a OK: ${r.outDir}")
    }

    private fun checkMlir(file: File, func: String, expectedArgs: List<String>, expectedResults: List<String>) {
        val mlir = file.readText()
        assertTrue(!mlir.contains("tensor<?"), "$func: dynamic shape leaked into export")
        assertTrue(!mlir.contains("stablehlo.custom_call"), "$func: custom_call in export")

        val sigLine = mlir.lineSequence().firstOrNull { it.contains("func.func") && it.contains("@$func(") }
            ?: error("$func: no func.func signature line found")
        val argsPart = sigLine.substringAfter("@$func(").substringBefore(") ->")
        val resultsPart = sigLine.substringAfter("-> ")
        val args = Regex("tensor<[^>]+>").findAll(argsPart).map { it.value }.toList()
        val results = Regex("tensor<[^>]+>").findAll(resultsPart).map { it.value }.toList()

        assertEquals(expectedArgs.size, args.size, "$func: arg count\n$sigLine")
        assertEquals(expectedResults.size, results.size, "$func: result count\n$sigLine")
        assertEquals(expectedArgs, args, "$func: arg types/order")
        assertEquals(expectedResults, results, "$func: result types/order")
    }
}
