package sk.ainet.asr.domain

/** Why decoding stopped — part of every result, printed by the CLI (honest-RTF discipline). */
public enum class StopReason {
    END_OF_TEXT,
    MAX_TOKENS,
    REPEAT_GUARD,
    STREAM_END,
    ERROR,
}

/** Per-stage wall-clock timings in milliseconds; audioMs is the denominator for RTF. */
public data class TranscriptionTimings(
    val audioMs: Long,
    val frontendMs: Long,
    val encodeMs: Long,
    val decodeMs: Long,
    val decodeSteps: Int = 0,
) {
    val totalMs: Long get() = frontendMs + encodeMs + decodeMs

    /** Real-time factor: processing time / audio duration. < 1.0 means faster than real time. */
    val rtf: Double get() = if (audioMs > 0) totalMs.toDouble() / audioMs else Double.NaN
}

public data class Transcription(
    val text: String,
    val tokens: List<Int>,
    val timings: TranscriptionTimings,
    val stopReason: StopReason,
)

/** Streaming output: cumulative, prefix-stable partials followed by exactly one final. */
public sealed interface AsrEvent {
    public data class Partial(val text: String) : AsrEvent

    public data class Final(val transcription: Transcription) : AsrEvent
}
