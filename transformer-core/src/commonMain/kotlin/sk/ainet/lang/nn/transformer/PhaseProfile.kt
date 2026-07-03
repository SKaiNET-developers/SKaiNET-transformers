package sk.ainet.lang.nn.transformer

import kotlin.time.TimeSource

/**
 * Lightweight, always-on accumulating profiler for the eager decode phases.
 * Diagnostic only — the sibling of the core backend's `KernelProfile` (which
 * times the matmul dispatch paths); this one attributes the NON-matmul decode
 * tail: attention plumbing, RMSNorm, RoPE, KV-cache, sampling, detok.
 *
 * Named-bucket design: `time("attn.rope") { ... }` accumulates wall time and a
 * call count per bucket. The map lookup + lambda per call is negligible next to
 * millisecond-scale phases (a few hundred calls per generated token). The
 * decode path is single-threaded, so plain mutable state is safe.
 *
 * Read [report] after a run; [reset] between phases (e.g. after model load so
 * the breakdown covers only prefill+decode).
 */
public object PhaseProfile {
    private val clock = TimeSource.Monotonic
    private val nanos = LinkedHashMap<String, Long>()
    private val calls = LinkedHashMap<String, Long>()

    public fun <R> time(phase: String, body: () -> R): R {
        val mark = clock.markNow()
        val r = body()
        val ns = mark.elapsedNow().inWholeNanoseconds
        nanos[phase] = (nanos[phase] ?: 0L) + ns
        calls[phase] = (calls[phase] ?: 0L) + 1L
        return r
    }

    public fun reset() {
        nanos.clear()
        calls.clear()
    }

    /** Buckets sorted by accumulated time, descending, with % of the bucket total. */
    public fun report(): String {
        val total = nanos.values.sum()
        fun ms(ns: Long) = ns / 1_000_000.0
        fun pct(ns: Long) = if (total > 0) 100.0 * ns / total else 0.0
        val width = (nanos.keys.maxOfOrNull { it.length } ?: 8).coerceAtLeast(8)
        return buildString {
            appendLine("[PhaseProfile] decode phase breakdown (buckets overlap matmul time; see KernelProfile):")
            for ((phase, ns) in nanos.entries.sortedByDescending { it.value }) {
                appendLine(
                    "  ${phase.padEnd(width)} : ${ms(ns)} ms over ${calls[phase]} calls (${pct(ns)}%)"
                )
            }
            append("  ${"phase total".padEnd(width)} : ${ms(total)} ms")
        }
    }
}
