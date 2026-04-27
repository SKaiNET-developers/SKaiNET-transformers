package sk.ainet.llm.providers

/**
 * Returns the index in [buffer] at which the earliest match of any [stopSequences]
 * begins, or `null` if none of the sequences appears.
 *
 * The buffer is checked in its current entirety on every call — fine for token-level
 * streaming where each tick decodes only a few bytes.
 */
internal fun findStopSequenceStart(buffer: CharSequence, stopSequences: List<String>): Int? {
    if (stopSequences.isEmpty()) return null
    var earliest = -1
    for (s in stopSequences) {
        if (s.isEmpty()) continue
        val idx = buffer.indexOf(s)
        if (idx >= 0 && (earliest < 0 || idx < earliest)) earliest = idx
    }
    return if (earliest < 0) null else earliest
}
