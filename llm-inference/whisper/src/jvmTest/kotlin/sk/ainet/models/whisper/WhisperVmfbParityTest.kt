package sk.ainet.models.whisper

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * G2b — host-CPU vmfb greedy parity. Compiles the three exported StableHLO
 * modules with the pinned dockerized IREE toolchain (llvm-cpu backend), then
 * drives the FULL greedy decode through `iree-run-module` — encoder from the
 * golden mel, prefill, then steps with host-built masks — and asserts the
 * decoded tokens equal the G1 golden sequence 37,1248,2804,5553,635,20314,13.
 *
 * Env-gated on WHISPER_SAFETENSORS + docker with the toolchain image
 * (skainet/iree-compiler:3.11.0). All artifacts land in WHISPER_EXPORT_OUT
 * (default build/whisper-export), which is mounted at /work in the container.
 */
class WhisperVmfbParityTest {

    private val cfg = WhisperExportHarness.cfg
    private val maxP = WhisperExportHarness.MAX_P
    private val nL = cfg.decoderLayers

    @Test
    fun vmfbGreedyMatchesGolden() {
        if (WhisperExportHarness.snapshotDir() == null) {
            println("SKIP: WHISPER_SAFETENSORS not set"); return
        }
        val out = WhisperExportHarness.exportOutDir()
        val tc = WhisperExportHarness.Toolchain(out)
        if (!tc.available()) {
            println("SKIP: docker/toolchain image unavailable"); return
        }

        val export = WhisperExportHarness.export(out)

        // ---- compile all three for host CPU ----
        for ((mlir, vmfb) in listOf(
            "whisper_encoder.mlir" to "whisper_encoder.vmfb",
            "whisper_prefill.mlir" to "whisper_prefill.vmfb",
            "whisper_step.mlir" to "whisper_step.vmfb",
        )) {
            val (code, log) = tc.compileCpu(mlir, vmfb)
            assertTrue(code == 0, "iree-compile failed for $mlir:\n${log.takeLast(4000)}")
            println("COMPILED $vmfb (${File(out, vmfb).length() / (1 shl 10)} KiB)")
        }

        // ---- encoder ----
        val mel = WhisperExportHarness.resourceFloats("mel_de4s.bin")
        WhisperExportHarness.writeNpyFloat(File(out, "mel.npy"), mel, intArrayOf(1, cfg.nMels, cfg.melFrames))
        runModule(
            tc, "whisper_encoder.vmfb", "whisper_encoder",
            inputs = listOf("@mel.npy"), outputs = listOf("feat.npy"), params = false,
        )
        val feat = WhisperExportHarness.readNpyFloat(File(out, "feat.npy"))
        val golden = WhisperExportHarness.resourceFloats("golden_feat.bin")
        var dot = 0.0; var na = 0.0; var nb = 0.0
        for (i in golden.indices) {
            dot += feat[i] * golden[i]; na += feat[i] * feat[i]; nb += golden[i] * golden[i]
        }
        val cos = dot / (kotlin.math.sqrt(na) * kotlin.math.sqrt(nb))
        println("vmfb encoder cosine vs golden: $cos")
        assertTrue(cos > 0.999, "vmfb encoder features diverged: cosine=$cos")

        // ---- prefill ----
        val tokens = WhisperSpecialTokens.forVocab(cfg.vocabSize)
        val prompt = tokens.transcribePrompt("de")
        val s = prompt.size
        WhisperExportHarness.writeNpyInt(File(out, "prompt.npy"), prompt, intArrayOf(s))
        // contract order: logits, then per layer selfK, selfV, crossK, crossV
        val prefillOuts = listOf("p_logits.npy") + (0 until nL).flatMap { l ->
            listOf("p_sk$l.npy", "p_sv$l.npy", "p_ck$l.npy", "p_cv$l.npy")
        }
        runModule(
            tc, "whisper_prefill.vmfb", "whisper_prefill",
            inputs = listOf("@prompt.npy", "@feat.npy"), outputs = prefillOuts, params = true,
        )
        val pLogits = WhisperExportHarness.readNpyFloat(File(out, "p_logits.npy"))
        assertEquals(s * cfg.vocabSize, pLogits.size, "prefill logits size")
        var tok = WhisperExportHarness.argmax(pLogits, from = (s - 1) * cfg.vocabSize, len = cfg.vocabSize)
        val decoded = mutableListOf(tok)
        println("prefill first token: $tok")

        // ---- greedy steps ----
        var selfK = (0 until nL).map { "p_sk$it.npy" }
        var selfV = (0 until nL).map { "p_sv$it.npy" }
        val crossK = (0 until nL).map { "p_ck$it.npy" }
        val crossV = (0 until nL).map { "p_cv$it.npy" }
        var pos = s
        var step = 0
        while (tok != tokens.eot && pos < maxP && decoded.size < 12) {
            WhisperExportHarness.writeNpyInt(File(out, "tok$step.npy"), intArrayOf(tok), intArrayOf(1))
            WhisperExportHarness.writeNpyInt(File(out, "pos$step.npy"), intArrayOf(pos), intArrayOf(1))
            WhisperExportHarness.writeNpyFloat(
                File(out, "addmask$step.npy"),
                WhisperMasks.stepAddMask(pos, maxP), intArrayOf(1, 1, 1, maxP),
            )
            WhisperExportHarness.writeNpyFloat(
                File(out, "wf$step.npy"),
                WhisperMasks.stepWriteVector(pos, maxP), intArrayOf(1, maxP, 1),
            )
            // contract order: tok, pos, addMask, wf, then per layer selfK, selfV, crossK, crossV
            val inputs = listOf("@tok$step.npy", "@pos$step.npy", "@addmask$step.npy", "@wf$step.npy") +
                (0 until nL).flatMap { l ->
                    listOf("@${selfK[l]}", "@${selfV[l]}", "@${crossK[l]}", "@${crossV[l]}")
                }
            // contract order: logits, then per layer updated selfK, selfV
            val outputs = listOf("s${step}_logits.npy") + (0 until nL).flatMap { l ->
                listOf("s${step}_sk$l.npy", "s${step}_sv$l.npy")
            }
            runModule(tc, "whisper_step.vmfb", "whisper_step", inputs, outputs, params = true)

            val logits = WhisperExportHarness.readNpyFloat(File(out, "s${step}_logits.npy"))
            assertEquals(cfg.vocabSize, logits.size, "step logits size")
            tok = WhisperExportHarness.argmax(logits)
            decoded += tok
            selfK = (0 until nL).map { "s${step}_sk$it.npy" }
            selfV = (0 until nL).map { "s${step}_sv$it.npy" }
            pos++; step++
            println("step ${step - 1}: token $tok")
        }

        println("vmfb greedy tokens: $decoded")
        val goldenTokens = WhisperExportHarness.GOLDEN_TOKENS.toList()
        assertTrue(decoded.size >= goldenTokens.size, "decode too short: $decoded")
        assertEquals(goldenTokens, decoded.take(goldenTokens.size), "vmfb greedy diverged from golden")
        println("G2b OK: golden ${goldenTokens.joinToString(",")} reproduced from vmfbs (export: ${export.outDir})")
    }

    private fun runModule(
        tc: WhisperExportHarness.Toolchain,
        vmfb: String,
        func: String,
        inputs: List<String>,
        outputs: List<String>,
        params: Boolean,
    ) {
        val args = buildList {
            add("run-module")
            add("--device=local-task")
            add("--module=$vmfb")
            add("--function=$func")
            if (params) add("--parameters=model=params.irpa")
            inputs.forEach { add("--input=$it") }
            outputs.forEach { add("--output=@$it") }
        }
        val (code, log) = tc.run(*args.toTypedArray())
        assertTrue(code == 0, "iree-run-module $func failed:\n${log.takeLast(4000)}")
    }
}
