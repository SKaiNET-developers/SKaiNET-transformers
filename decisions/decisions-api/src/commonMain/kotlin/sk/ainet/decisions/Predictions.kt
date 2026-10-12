package sk.ainet.decisions

/** The named definition a [Confidence] value was computed with (REQ-DEC-006). */
public enum class ConfidenceDefinition {
    /** `1 − H(p) / log k` over the option distribution (choice, score). */
    NORMALIZED_ENTROPY,
    /** `max(P(true), 1 − P(true))` (boolean probability). */
    TWO_WAY_MAX,
}

/** Not a correctness guarantee: how concentrated the distribution is, under the stated definition. */
public data class Confidence(public val value: Double, public val definition: ConfidenceDefinition) {
    init { require(value in 0.0..1.0) { "confidence must be within [0, 1], got $value" } }
}

public data class OptionProbability(public val optionId: String, public val probability: Double)

/** A candidate prediction as returned by an adapter, before any acceptance policy (REQ-API-011). */
public sealed class Prediction {
    public abstract val confidence: Confidence

    /** Probabilities keyed by the caller's option ids, in the caller's order; selected = argmax. */
    public data class Choice(
        public val selectedOptionId: String,
        public val probabilities: List<OptionProbability>,
        override val confidence: Confidence,
    ) : Prediction() {
        public fun probabilityOf(optionId: String): Double = probabilities.first { it.optionId == optionId }.probability
        public val margin: Double get() {
            val sorted = probabilities.map { it.probability }.sortedDescending()
            return if (sorted.size < 2) 1.0 else sorted[0] - sorted[1]
        }
    }

    /** [expected] = Σ i·P(level i); [selectedLevel] is the argmax projection (REQ-DEC-004). */
    public data class Score(
        public val expected: Double,
        public val selectedLevel: Int,
        public val distribution: List<Double>,
        override val confidence: Confidence,
    ) : Prediction() {
        public val normalizedEntropy: Double get() = 1.0 - confidence.value
    }

    /** `probabilityFalse = 1 − probabilityTrue`; orientation independent of any label text (REQ-DEC-005). */
    public data class BooleanProbability(
        public val probabilityTrue: Double,
        override val confidence: Confidence,
    ) : Prediction() {
        public val probabilityFalse: Double get() = 1.0 - probabilityTrue
    }
}
