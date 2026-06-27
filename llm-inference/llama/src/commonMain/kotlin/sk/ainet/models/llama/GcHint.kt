package sk.ainet.models.llama

/**
 * Hint the platform to reclaim unreachable memory *now*.
 *
 * On **Kotlin/Native** the GC is allocation-triggered and lags during a tight load/convert loop, so
 * the per-tensor transient copies (`pread` ByteArray → `copyOf` → `extractRawBytes` → relayout
 * ByteArray) pile up uncollected and inflate peak RSS — which OOM-kills the 2 GB board even though the
 * resident model is only ~0.9 GB. Calling this at per-tensor boundaries (after the source tensor has
 * been dropped) bounds the high-water to roughly one tensor in flight. Mirrors the established board
 * pattern in `GemmaDecoder` (`kotlin.native.runtime.GC.collect()`).
 *
 * **No-op on the JVM** — the JVM GC reclaims under allocation pressure on its own.
 */
internal expect fun gcCollectHint()
