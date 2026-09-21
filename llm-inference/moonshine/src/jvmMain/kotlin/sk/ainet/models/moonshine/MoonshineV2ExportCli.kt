package sk.ainet.models.moonshine

import java.io.File

/**
 * CLI entry for the Moonshine v2 streaming export: a Hugging Face `moonshine_streaming` snapshot in, the five
 * StableHLO graphs + `dec_embed.bin` + `vocab.bin` + `manifest.json` out.
 *
 *   MOONSHINE_MODEL_DIR=<snapshot dir> MOONSHINE_OUT_DIR=build/mlir \
 *     ./gradlew :llm-inference:moonshine:exportMoonshineV2
 *
 * Env:
 *   MOONSHINE_MODEL_DIR   (required) directory with `config.json`, `model.safetensors`, `tokenizer.json`
 *   MOONSHINE_OUT_DIR     output directory (default `build/mlir/moonshine-v2`; or the first argument)
 *   MOONSHINE_GRAPH       `all` (default) or a comma-separated subset of
 *                         `frontend,encoder,adapter,prefill,with_past,embeddings,vocab`
 *   MOONSHINE_FE_SAMPLES  frontend input length in samples (default 21760)
 *   MOONSHINE_ENC_FRAMES  encoder / adapter window in frames (default 64)
 *   MOONSHINE_MAX_MEM     padded encoder-memory length the decoder graphs are built for (default 256)
 *   MOONSHINE_STATIC_PAST fix the step graph's self-attention cache length (default: dynamic)
 *
 * Geometry, per-layer attention bands, vocabulary size and the positional-table length are read from the
 * snapshot — there is nothing to configure per language or model size. With `all`, the export fails if any
 * tensor of the checkpoint was not used.
 */
public fun main(args: Array<String>) {
    val modelDir = File(System.getenv("MOONSHINE_MODEL_DIR") ?: error("set MOONSHINE_MODEL_DIR to a Hugging Face moonshine_streaming snapshot directory"))
    val outDir = File(System.getenv("MOONSHINE_OUT_DIR") ?: args.getOrNull(0) ?: "build/mlir/moonshine-v2")
    val selection = (System.getenv("MOONSHINE_GRAPH") ?: "all").lowercase().split(',').map { it.trim() }.filter { it.isNotEmpty() }.toSet()
    val known = MoonshineV2ExportHarness.Graph.entries.associateBy { it.name.lowercase() }
    val unknown = selection - known.keys - setOf("all", "embeddings", "vocab")
    require(unknown.isEmpty()) { "MOONSHINE_GRAPH: unknown $unknown — use all or any of ${known.keys + listOf("embeddings", "vocab")}" }
    val all = "all" in selection
    fun env(name: String): Int? = System.getenv(name)?.toInt()
    val defaults = MoonshineV2ExportOptions()
    val options = MoonshineV2ExportOptions(
        frontendSamples = env("MOONSHINE_FE_SAMPLES") ?: defaults.frontendSamples,
        encoderFrames = env("MOONSHINE_ENC_FRAMES") ?: defaults.encoderFrames,
        maxMemoryFrames = env("MOONSHINE_MAX_MEM") ?: defaults.maxMemoryFrames,
        staticPast = env("MOONSHINE_STATIC_PAST"),
    )

    MoonshineV2Checkpoint(modelDir).use { checkpoint ->
        val cfg = checkpoint.config
        println(
            "[moonshine-export] encoder ${cfg.dim}x${cfg.encoderLayers} (ffn ${cfg.ffnDim}), decoder ${cfg.decoderDim}x${cfg.decoderLayers} " +
                "(ffn ${cfg.decoderFfnDim}), ${cfg.nHeads} heads x ${cfg.headDim}, vocab ${cfg.vocabSize}, windows ${cfg.slidingWindows}",
        )
        val harness = MoonshineV2ExportHarness(checkpoint, options) { println("[moonshine-export] $it") }
        val graphs = if (all) known.values.toSet() else selection.mapNotNull(known::get).toSet()
        val files = ArrayList(harness.export(outDir, graphs))
        if (all || "embeddings" in selection) files += harness.writeEmbeddings(outDir).also { println("[moonshine-export] ${it.name}: ${it.length() / 1024} KiB") }
        if (all || "vocab" in selection) files += harness.writeVocab(outDir).also { println("[moonshine-export] ${it.name}: ${it.length() / 1024} KiB") }
        if (all) {
            val unused = checkpoint.unconsumed()
            check(unused.isEmpty()) { "checkpoint tensors the export did not use (the weight map is incomplete): $unused" }
            println("[moonshine-export] every checkpoint tensor was used")
            println("[moonshine-export] ${harness.writeManifest(outDir, files).absolutePath}")
        }
    }
}
