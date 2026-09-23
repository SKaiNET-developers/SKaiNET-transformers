package sk.ainet.models.qwen

/**
 * Env-driven CLI entry for [QwenExportHarness] (SKaiNET-transformers#411), the same shape as
 * `FunctionGemmaExportCli` / the module's `exportQwen` Gradle task:
 *
 *   QWEN_GGUF=…/qwen3-0.6b-q8_0.gguf QWEN_OUT_DIR=build/mlir QWEN_GRAPH=all \
 *     ./gradlew :llm-inference:qwen:exportQwen
 *
 * Env vars: `QWEN_GGUF` (required), `QWEN_OUT_DIR` (default `build/mlir`), `QWEN_GRAPH`
 * (`prefill_at` | `prefill_with_past` | `with_past` | `all`, default `all`), `QWEN_SEQ`
 * (prefill-at fixed length, default 1024 — the box's catalog prefix is 844 tokens, see
 * NLU-QWEN-TRACKING.md Q0.2), `QWEN_CHUNK` (default [QwenKvContract.DEFAULT_CHUNK]), `QWEN_DTYPE`
 * (`bf16` default | `fp32`).
 */
public fun main() {
    val gguf = System.getenv("QWEN_GGUF") ?: error("QWEN_GGUF is required (path to a Qwen2/Qwen3 .gguf)")
    val outDir = System.getenv("QWEN_OUT_DIR") ?: "build/mlir"
    val graph = System.getenv("QWEN_GRAPH") ?: "all"
    val seq = System.getenv("QWEN_SEQ")?.toIntOrNull() ?: 1024
    val chunk = System.getenv("QWEN_CHUNK")?.toIntOrNull() ?: QwenKvContract.DEFAULT_CHUNK
    val bf16 = (System.getenv("QWEN_DTYPE") ?: "bf16").lowercase() != "fp32"

    when (graph) {
        "prefill_at" -> QwenExportHarness.exportPrefillAt(gguf, outDir, seq, bf16)
        "prefill_with_past" -> QwenExportHarness.exportPrefillWithPast(gguf, outDir, chunk, bf16)
        "with_past" -> QwenExportHarness.exportWithPast(gguf, outDir, bf16)
        "all" -> QwenExportHarness.exportAll(gguf, outDir, seq, chunk, bf16)
        else -> error("QWEN_GRAPH must be one of prefill_at|prefill_with_past|with_past|all, got '$graph'")
    }
    println("[QwenExportCli] wrote $graph to $outDir")
}
