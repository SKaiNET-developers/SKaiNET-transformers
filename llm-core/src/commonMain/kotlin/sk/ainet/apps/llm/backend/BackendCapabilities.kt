package sk.ainet.apps.llm.backend

/**
 * Static description of a backend's abilities so callers can pick a compatible profile —
 * complements [BackendProvider.isAvailable]/[BackendProvider.priority] (which answer "is this
 * backend usable and how good is it right now") with "what can this backend actually do".
 *
 * Ported from the ASR cartridge ecosystem's `backend-spi.BackendCapabilities` (originally
 * vendored from skainet-whisper-kmp@cbbbd9b) when its `ExecutionContextFactory`/`BackendRegistry`
 * seam was unified into this existing [BackendProvider]/[BackendRegistry] pair rather than
 * shipping two parallel backend-selection registries.
 */
public data class BackendCapabilities(
    val supportedDTypes: Set<String>, // e.g. {"FP32"}
    val supportsCompile: Boolean = false,
    val usesNpu: Boolean = false,
    val maxSeqLen: Int = Int.MAX_VALUE,
)

/** Backend construction knobs (kept minimal; backends extend via [extras]). */
public data class BackendOptions(
    val threads: Int = 0, // 0 = backend default
    val extras: Map<String, String> = emptyMap(),
)
