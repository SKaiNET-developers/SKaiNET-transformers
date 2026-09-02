package sk.ainet.models.gemma3n

/**
 * CLI entry for [Gemma3nExportHarness] — the `exportGemma3n` gradle task
 * (SmolLM2/FunctionGemma pattern).
 *
 * Env:
 *  - `GEMMA3N_GGUF`    path to gemma-3n GGUF (required)
 *  - `GEMMA3N_OUT_DIR` output directory (default `build/gemma3n-export`)
 *  - `GEN_SEQ`         fixed sequence length of the redecode graph (default 24)
 *  - `GEMMA3N_DTYPE`   `bf16` (default) or `f32` external params
 */
public fun main() {
    val gguf = System.getenv("GEMMA3N_GGUF")
        ?: error("GEMMA3N_GGUF must point at a gemma-3n GGUF checkpoint")
    val outDir = System.getenv("GEMMA3N_OUT_DIR") ?: "build/gemma3n-export"
    val seq = System.getenv("GEN_SEQ")?.toIntOrNull() ?: 24
    val bf16 = (System.getenv("GEMMA3N_DTYPE") ?: "bf16").lowercase() != "f32"
    val layers = System.getenv("GEMMA3N_LAYERS")?.toIntOrNull()

    val r = Gemma3nExportHarness.export(gguf = gguf, outDir = outDir, seq = seq, bf16 = bf16, layers = layers)
    println("gemma3n export complete:")
    println("  mlir        = ${r.mlirPath}")
    println("  safetensors = ${r.safetensorsPath} (${r.weightMiB} MiB, ${r.externalParamCount} params)")
    println("  manifest    = ${r.manifestPath}")
    println("  seq=${r.seq} vocab=${r.vocabSize} fn=@${Gemma3nExportHarness.FN_REDECODE}")
}
