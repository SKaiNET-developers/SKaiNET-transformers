package sk.ainet.decisions

/** REQ-DEC-007. */
public enum class CalibrationStatus { UNVERIFIED, CHECKPOINT_PROVIDED, LOCALLY_VALIDATED }

/** Per-question packing facts an adapter reports; counts only, never content (REQ-PRV-006). */
public data class QuestionTrace(
    public val questionId: String,
    public val packedTokens: Int,
    public val stateTokens: Int,
    public val stateTokensDropped: Int,
) {
    public val truncated: Boolean get() = stateTokensDropped > 0
}

/** What produced a result; identifiers, versions, counts and timings only (REQ-PRV-006). */
public data class Provenance(
    public val modelId: String,
    public val backend: String,
    public val precision: String,
    public val preprocessingVersion: String,
    public val calibration: CalibrationStatus,
    /** Digest of the calibration artifact, or `none`. */
    public val calibrationIdentity: String,
    public val policyId: String?,
    public val policyVersion: String?,
    public val traces: List<QuestionTrace>,
    public val durationMillis: Long,
    public val notes: List<String> = emptyList(),
)
