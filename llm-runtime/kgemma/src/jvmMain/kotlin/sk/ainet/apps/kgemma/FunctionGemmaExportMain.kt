package sk.ainet.apps.kgemma

/**
 * CLI entry for the FunctionGemma compiled export — driven by `scripts/compile-gemma.sh`.
 *
 *   GEMMA_GGUF=…Q5_K_M.gguf GEMMA_OUT_DIR=build/mlir GEMMA_GRAPH=all \
 *     ./gradlew -PuseLocalSkainet=true :llm-runtime:kgemma:exportFunctionGemma
 *
 * Env: GEMMA_GGUF (required), GEMMA_OUT_DIR (default build/mlir), GEN_SEQ (24),
 * PARTIAL_ROTARY (1.0), GEMMA_DTYPE (bf16 default; set FP32 for numeric bring-up),
 * GEMMA_GRAPH selects the graph(s):
 *   redecode (default) — the fixed seq=[GEN_SEQ] re-decode graph (gemma-gen.mlir + gemma.safetensors)
 *   prefill            — the KV-cache PREFILL graph (gemma-prefill.mlir)
 *   with_past          — the KV-cache DECODE graph (gemma-with-past.mlir, dynamic `?` cache)
 *   all                — redecode + prefill + with_past (all three share the one gemma.safetensors/irpa)
 * The prefill/with_past graphs reference the SAME "model" external weights as redecode, so a single
 * `gemma.safetensors` -> `gemma-gen.irpa` (written by the redecode export) serves all three vmfbs.
 */
public fun main(args: Array<String>) {
    val gguf = System.getenv("GEMMA_GGUF") ?: error("set GEMMA_GGUF to the FunctionGemma Q5_K_M .gguf")
    val outDir = System.getenv("GEMMA_OUT_DIR") ?: args.getOrNull(0) ?: "build/mlir"
    val seq = System.getenv("GEN_SEQ")?.toInt() ?: 24
    val partial = System.getenv("PARTIAL_ROTARY")?.toFloat() ?: 1.0f
    val bf16 = !System.getenv("GEMMA_DTYPE").equals("FP32", ignoreCase = true)
    // GEMMA_QUANT=int8 quantizes the 2D matmul weights to per-row int8 in the compiled graph (Phase 5).
    val quantInt8 = System.getenv("GEMMA_QUANT").equals("int8", ignoreCase = true)
    val graph = (System.getenv("GEMMA_GRAPH") ?: "redecode").lowercase()

    if (graph == "redecode" || graph == "all") {
        val r = FunctionGemmaExport.export(gguf = gguf, outDir = outDir, seq = seq, partialRotary = partial, bf16 = bf16, quantizeInt8 = quantInt8)
        val tag = if (quantInt8) "int8" else if (bf16) "bf16" else "f32"
        println(
            "[functiongemma-export] redecode: wrote ${r.mlirPath} + ${r.safetensorsPath} " +
                "(${r.externalParamCount} params, ${r.weightMiB} MiB, seq=${r.seq}, dtype=$tag)",
        )
    }
    if (graph == "prefill" || graph == "all") {
        FunctionGemmaExport.exportPrefill(gguf = gguf, outDir = outDir, seq = seq, partialRotary = partial, bf16 = bf16)
        println("[functiongemma-export] prefill: wrote $outDir/gemma-prefill.mlir (seq=$seq)")
    }
    if (graph == "with_past" || graph == "all") {
        FunctionGemmaExport.exportWithPast(gguf = gguf, outDir = outDir, dynamicPast = true, partialRotary = partial, bf16 = bf16)
        println("[functiongemma-export] with_past: wrote $outDir/gemma-with-past.mlir (dynamic 1x1x?x256 cache)")
    }
    if (graph !in setOf("redecode", "prefill", "with_past", "all")) {
        error("GEMMA_GRAPH must be redecode|prefill|with_past|all, got '$graph'")
    }
}
