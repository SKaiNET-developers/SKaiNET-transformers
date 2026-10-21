package sk.ainet.models.whisper

import java.io.File

/**
 * CLI entry for the Whisper export: a Hugging Face Whisper snapshot in, the three StableHLO graphs
 * (`whisper_encoder`, `whisper_prefill`, `whisper_step`) + one merged `params.irpa` + `manifest.json` out.
 *
 *   WHISPER_MODEL_DIR=<snapshot dir> WHISPER_OUT_DIR=build/mlir \
 *     ./gradlew :llm-inference:whisper:exportWhisper
 *
 * Env:
 *   WHISPER_MODEL_DIR      (required) directory with `config.json` and `model.safetensors`
 *   WHISPER_OUT_DIR        output directory (default `build/mlir/whisper`; or the first argument)
 *   WHISPER_AUDIO_CTX      encoder positions after the stride-2 conv (default 200 = a 4 s window; 1500 = 30 s)
 *   WHISPER_MAX_POSITIONS  decoder self-attention KV length, prompt + generated tokens (default 48)
 *   WHISPER_LANGUAGE       language code of the manifest's transcribe prompt (default `en`; ignored for `*.en`)
 *
 * Geometry and vocabulary are read from the snapshot — tiny, base, small and the English-only variants alike.
 */
public fun main(args: Array<String>) {
    val modelDir = File(System.getenv("WHISPER_MODEL_DIR") ?: error("set WHISPER_MODEL_DIR to a Hugging Face Whisper snapshot directory"))
    val outDir = File(System.getenv("WHISPER_OUT_DIR") ?: args.getOrNull(0) ?: "build/mlir/whisper")
    fun env(name: String): Int? = System.getenv(name)?.toInt()
    val defaults = WhisperExportOptions()
    val audioCtx = env("WHISPER_AUDIO_CTX") ?: defaults.audioCtx
    val configFile = File(modelDir, "config.json")
    require(configFile.isFile) { "no config.json in $modelDir" }
    val cfg = WhisperExportHarness.configFromHf(configFile.readText(), audioCtx)
    val language = System.getenv("WHISPER_LANGUAGE") ?: defaults.language
    val options = WhisperExportOptions(
        audioCtx = audioCtx,
        maxPositions = env("WHISPER_MAX_POSITIONS") ?: defaults.maxPositions,
        promptLength = if (cfg.isMultilingual) 4 else 2,
        language = language,
    )
    println(
        "[whisper-export] ${cfg.encoderLayers}+${cfg.decoderLayers} layers, dim ${cfg.dim}, ${cfg.nHeads} heads, " +
            "vocab ${cfg.vocabSize}${if (cfg.isMultilingual) " (multilingual, prompt language $language)" else " (English-only)"}, " +
            "audioCtx ${cfg.audioCtx} (${cfg.melFrames} mel frames), maxPositions ${options.maxPositions}",
    )
    val result = WhisperExportHarness(cfg, options) { println("[whisper-export] $it") }
        .export(File(modelDir, "model.safetensors"), outDir)
    println("[whisper-export] ${result.manifest.absolutePath}")
}
