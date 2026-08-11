package sk.ainet.apps.kgemma

import sk.ainet.models.functiongemma.FunctionGemmaExportHarness

/**
 * DEPRECATED — moved to [sk.ainet.models.functiongemma.FunctionGemmaExportHarness] in
 * `:llm-inference:functiongemma` (the whisper/moonshine "one self-contained module" shape:
 * spec + harness + contract manifest, consumed downstream via docker
 * `skainet/iree-compiler:3.11.0` + a thin runtime).
 *
 * This object is kept as a **delegating shim** (deprecate-don't-delete) so existing callers —
 * [FunctionGemmaExportMain]'s env CLI, `scripts/compile-gemma.sh`'s
 * `:llm-runtime:kgemma:exportFunctionGemma` gradle task, and the pre-existing
 * `FunctionGemmaExportTest` / `FunctionGemmaWithPastMlirDumpTest` / `FunctionGemmaInt8QuantTest`
 * integration tests — keep compiling and passing unchanged. The move is behavior-identical:
 * the harness bodies were moved VERBATIM (golden MLIR/safetensors byte-hashes verified equal
 * on the same checkpoint before/after the move).
 *
 * New callers should use [FunctionGemmaExportHarness] (or its spec-driven `FunctionGemmaSpec` /
 * `exportAll` entry point) directly.
 */
@Deprecated(
    "Moved to :llm-inference:functiongemma. Use sk.ainet.models.functiongemma.FunctionGemmaExportHarness directly.",
    ReplaceWith(
        "FunctionGemmaExportHarness",
        "sk.ainet.models.functiongemma.FunctionGemmaExportHarness",
    ),
)
public object FunctionGemmaExport {

    @Deprecated(
        "Moved to :llm-inference:functiongemma.FunctionGemmaExportHarness.RedecodeResult.",
        ReplaceWith(
            "FunctionGemmaExportHarness.RedecodeResult",
            "sk.ainet.models.functiongemma.FunctionGemmaExportHarness",
        ),
    )
    public data class Result(
        val mlirPath: String,
        val safetensorsPath: String,
        val externalParamCount: Int,
        val weightMiB: Long,
        val seq: Int,
    )

    /** @see FunctionGemmaExportHarness.export */
    @Deprecated(
        "Moved to :llm-inference:functiongemma.FunctionGemmaExportHarness.export.",
        ReplaceWith(
            "FunctionGemmaExportHarness.export(gguf, outDir, seq, partialRotary, bf16, quantizeInt8)",
            "sk.ainet.models.functiongemma.FunctionGemmaExportHarness",
        ),
    )
    public fun export(
        gguf: String,
        outDir: String,
        seq: Int = 24,
        partialRotary: Float = 1.0f,
        bf16: Boolean = true,
        quantizeInt8: Boolean = false,
    ): Result {
        val r = FunctionGemmaExportHarness.export(gguf, outDir, seq, partialRotary, bf16, quantizeInt8)
        return Result(r.mlirPath, r.safetensorsPath, r.externalParamCount, r.weightMiB, r.seq)
    }

    /** @see FunctionGemmaExportHarness.exportWithPast */
    @Deprecated(
        "Moved to :llm-inference:functiongemma.FunctionGemmaExportHarness.exportWithPast.",
        ReplaceWith(
            "FunctionGemmaExportHarness.exportWithPast(gguf, outDir, past, dynamicPast, partialRotary, bf16)",
            "sk.ainet.models.functiongemma.FunctionGemmaExportHarness",
        ),
    )
    public fun exportWithPast(
        gguf: String,
        outDir: String,
        past: Int = 1,
        dynamicPast: Boolean = true,
        partialRotary: Float = 1.0f,
        bf16: Boolean = true,
    ): String = FunctionGemmaExportHarness.exportWithPast(gguf, outDir, past, dynamicPast, partialRotary, bf16)

    /** @see FunctionGemmaExportHarness.exportPrefill */
    @Deprecated(
        "Moved to :llm-inference:functiongemma.FunctionGemmaExportHarness.exportPrefill.",
        ReplaceWith(
            "FunctionGemmaExportHarness.exportPrefill(gguf, outDir, seq, partialRotary, bf16)",
            "sk.ainet.models.functiongemma.FunctionGemmaExportHarness",
        ),
    )
    public fun exportPrefill(
        gguf: String,
        outDir: String,
        seq: Int = 24,
        partialRotary: Float = 1.0f,
        bf16: Boolean = true,
    ): String = FunctionGemmaExportHarness.exportPrefill(gguf, outDir, seq, partialRotary, bf16)
}
