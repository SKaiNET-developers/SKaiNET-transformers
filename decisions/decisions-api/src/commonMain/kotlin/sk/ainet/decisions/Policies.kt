package sk.ainet.decisions

/** A policy shape the specification forbids (REQ-POL-003); raised when the policy is constructed. */
public class InvalidPolicyException(message: String) : IllegalArgumentException(message)

/** Choice acceptance: both thresholds inclusive, both optional (REQ-POL-002). */
public data class ChoiceAcceptance(
    public val minConfidence: Double? = null,
    /** Minimum gap between the top two probabilities. */
    public val minMargin: Double? = null,
) {
    init {
        minConfidence?.let { if (it !in 0.0..1.0) throw InvalidPolicyException("minConfidence must be within [0, 1], got $it") }
        minMargin?.let { if (it !in 0.0..1.0) throw InvalidPolicyException("minMargin must be within [0, 1], got $it") }
    }
}

/** Score acceptance over the level distribution, never over the expected value (REQ-POL-004). */
public sealed class ScoreAcceptance {
    public data class MinTopProbability(public val min: Double) : ScoreAcceptance() {
        init { if (min !in 0.0..1.0) throw InvalidPolicyException("min top probability must be within [0, 1], got $min") }
    }

    /** Accept while the normalised entropy of the distribution stays at or below [max]. */
    public data class MaxNormalizedEntropy(public val max: Double) : ScoreAcceptance() {
        init { if (max !in 0.0..1.0) throw InvalidPolicyException("max normalised entropy must be within [0, 1], got $max") }
    }
}

/** Boolean acceptance: `P(true) <= lower` accepts false, `>= upper` accepts true, in between abstains (REQ-POL-003). */
public data class BooleanAcceptance(public val lower: Double, public val upper: Double) {
    init {
        if (lower !in 0.0..1.0 || upper !in 0.0..1.0) throw InvalidPolicyException("boolean thresholds must be within [0, 1], got $lower / $upper")
        if (lower >= upper) throw InvalidPolicyException("boolean lower threshold $lower must be strictly below the upper threshold $upper")
    }
}

/**
 * An application-owned, versioned acceptance policy (REQ-POL-001). A primitive without a policy
 * entry abstains with [AbstentionReason.NO_POLICY]; the library ships no accepting defaults (REQ-POL-005).
 */
public data class AcceptancePolicy(
    public val id: String,
    public val version: String,
    public val choice: ChoiceAcceptance? = null,
    public val score: ScoreAcceptance? = null,
    public val boolean: BooleanAcceptance? = null,
) {
    init {
        require(id.isNotBlank() && version.isNotBlank()) { "a policy needs an id and a version" }
    }
}
