package sk.ainet.apps.kgemma

/**
 * CLI entry for the FunctionGemma compiled export — driven by `scripts/compile-gemma.sh`.
 *
 *   GEMMA_GGUF=…Q5_K_M.gguf GEMMA_OUT_DIR=build/mlir \
 *     ./gradlew -PuseLocalSkainet=true :llm-runtime:kgemma:exportFunctionGemma
 *
 * Env: GEMMA_GGUF (required), GEMMA_OUT_DIR (default build/mlir), GEN_SEQ (24),
 * PARTIAL_ROTARY (1.0), GEMMA_DTYPE (bf16 default; set FP32 for numeric bring-up).
 */
public fun main(args: Array<String>) {
    val gguf = System.getenv("GEMMA_GGUF") ?: error("set GEMMA_GGUF to the FunctionGemma Q5_K_M .gguf")
    val outDir = System.getenv("GEMMA_OUT_DIR") ?: args.getOrNull(0) ?: "build/mlir"
    val seq = System.getenv("GEN_SEQ")?.toInt() ?: 24
    val partial = System.getenv("PARTIAL_ROTARY")?.toFloat() ?: 1.0f
    val bf16 = !System.getenv("GEMMA_DTYPE").equals("FP32", ignoreCase = true)
    val r = FunctionGemmaExport.export(gguf = gguf, outDir = outDir, seq = seq, partialRotary = partial, bf16 = bf16)
    println(
        "[functiongemma-export] wrote ${r.mlirPath} + ${r.safetensorsPath} " +
            "(${r.externalParamCount} params, ${r.weightMiB} MiB, seq=${r.seq}, dtype=${if (bf16) "bf16" else "f32"})",
    )
}
