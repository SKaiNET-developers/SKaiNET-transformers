package sk.ainet.models.functiongemma

/**
 * CLI entry for the FunctionGemma compiled export — driven by `scripts/compile-gemma.sh`
 * via the `:llm-inference:functiongemma:exportFunctionGemma` gradle task (see build.gradle.kts).
 *
 * Same env contract as the legacy `:llm-runtime:kgemma:exportFunctionGemma` wrapper (D2 —
 * deprecate-don't-delete), so `compile-gemma.sh` switches by task path alone:
 *
 *   GEMMA_GGUF=…Q5_K_M.gguf GEMMA_OUT_DIR=build/mlir GEMMA_GRAPH=all \
 *     ./gradlew :llm-inference:functiongemma:exportFunctionGemma
 *
 * Env: GEMMA_GGUF (required), GEMMA_OUT_DIR (default build/mlir), GEN_SEQ (24),
 * PARTIAL_ROTARY (1.0), GEMMA_DTYPE (bf16 default; set FP32 for numeric bring-up),
 * GEMMA_GRAPH selects the graph(s):
 *   redecode (default) — the fixed seq=[GEN_SEQ] re-decode graph (gemma-gen.mlir + gemma.safetensors)
 *   prefill            — the KV-cache PREFILL graph (gemma-prefill.mlir)
 *   with_past          — the KV-cache DECODE graph (gemma-with-past.mlir, dynamic `?` cache)
 *   all                — redecode + prefill + with_past + manifest.json (the full module contract)
 * The prefill/with_past graphs reference the SAME "model" external weights as redecode, so a single
 * `gemma.safetensors` -> `gemma-gen.irpa` (written by the redecode export) serves the redecode vmfb;
 * the KV graphs each write and serve their OWN safetensors/irpa (per-trace external numbering).
 */
public fun main(args: Array<String>) {
    val gguf = System.getenv("GEMMA_GGUF") ?: error("set GEMMA_GGUF to the FunctionGemma Q5_K_M .gguf")
    val outDir = System.getenv("GEMMA_OUT_DIR") ?: args.getOrNull(0) ?: "build/mlir"
    val seq = System.getenv("GEN_SEQ")?.toInt() ?: 24
    val partial = System.getenv("PARTIAL_ROTARY")?.toFloat() ?: 1.0f
    val quant = when {
        System.getenv("GEMMA_QUANT").equals("int8", ignoreCase = true) -> FunctionGemmaQuant.INT8
        System.getenv("GEMMA_DTYPE").equals("FP32", ignoreCase = true) -> FunctionGemmaQuant.FP32
        else -> FunctionGemmaQuant.BF16
    }
    val graph = (System.getenv("GEMMA_GRAPH") ?: "redecode").lowercase()
    if (graph !in setOf("redecode", "prefill", "with_past", "redecode_at", "prefill_at", "all")) {
        error("GEMMA_GRAPH must be redecode|prefill|with_past|redecode_at|prefill_at|all, got '$graph'")
    }

    val spec = FunctionGemmaSpec(gguf = gguf, seq = seq, partialRotary = partial, quant = quant)

    if (graph == "redecode" || graph == "all") {
        val r = FunctionGemmaExportHarness.exportRedecode(spec, outDir)
        println(
            "[functiongemma-export] redecode: wrote ${r.mlirPath} + ${r.safetensorsPath} " +
                "(${r.externalParamCount} params, ${r.weightMiB} MiB, seq=${r.seq}, dtype=$quant)",
        )
    }
    if (graph == "prefill" || graph == "all") {
        FunctionGemmaExportHarness.exportPrefill(spec, outDir)
        println("[functiongemma-export] prefill: wrote $outDir/gemma-prefill.mlir (seq=$seq)")
    }
    if (graph == "redecode_at") {
        val r = FunctionGemmaExportHarness.exportRedecodeAt(spec, outDir)
        println("[functiongemma-export] redecode_at: wrote ${r.mlirPath} + ${r.safetensorsPath} (${r.externalParamCount} externals, ${r.weightMiB} MiB, seq=${r.seq})")
    }
    if (graph == "prefill_at") {
        FunctionGemmaExportHarness.exportPrefillAt(spec, outDir)
        println("[functiongemma-export] prefill_at: wrote $outDir/gemma-prefill-at.mlir + gemma-prefill-at.safetensors")
    }
    if (graph == "with_past" || graph == "all") {
        FunctionGemmaExportHarness.exportWithPast(spec, outDir)
        println("[functiongemma-export] with_past: wrote $outDir/gemma-with-past.mlir (dynamic 1x1x?x256 cache)")
    }
    if (graph == "all") {
        val manifest = FunctionGemmaExportHarness.writeManifest(spec, outDir)
        println("[functiongemma-export] manifest: wrote ${manifest.absolutePath}")
    }
}
