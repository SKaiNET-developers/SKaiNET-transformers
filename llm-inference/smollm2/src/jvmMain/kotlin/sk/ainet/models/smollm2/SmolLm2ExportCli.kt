package sk.ainet.models.smollm2

/**
 * CLI entry for the SmolLM2 compiled export, driven by the
 * `:llm-inference:smollm2:exportSmolLm2` gradle task (see build.gradle.kts).
 *
 * Env: SMOLLM2_GGUF (required), SMOLLM2_OUT_DIR (default build/mlir),
 * SMOLLM2_SEQ (default 24), SMOLLM2_DTYPE (bf16 default; set FP32 for
 * numeric bring-up).
 */
public fun main() {
    val gguf = System.getenv("SMOLLM2_GGUF") ?: error("set SMOLLM2_GGUF to the SmolLM2 .gguf")
    val outDir = System.getenv("SMOLLM2_OUT_DIR") ?: "build/mlir"
    val seq = System.getenv("SMOLLM2_SEQ")?.toInt() ?: 24
    val bf16 = !System.getenv("SMOLLM2_DTYPE").equals("FP32", ignoreCase = true)

    val r = SmolLm2ExportHarness.export(gguf, outDir, seq, bf16)
    println(
        "[smollm2-export] redecode: wrote ${r.mlirPath} + ${r.safetensorsPath} " +
            "(${r.externalParamCount} params, ${r.weightMiB} MiB, seq=${r.seq}, bf16=$bf16)",
    )
}
