package sk.ainet.decisions

/** An accepted decision; the candidate it was derived from travels with it. */
public sealed class Decision {
    public abstract val prediction: Prediction

    public data class Choice(public val optionId: String, override val prediction: Prediction.Choice) : Decision()
    public data class Score(public val level: Int, override val prediction: Prediction.Score) : Decision()
    public data class Boolean(public val value: kotlin.Boolean, override val prediction: Prediction.BooleanProbability) : Decision()
}

/** Why a candidate was not accepted (REQ-POL-006). Low confidence is never a failure. */
public enum class AbstentionReason {
    BELOW_CONFIDENCE, BELOW_MARGIN, BETWEEN_THRESHOLDS, NO_POLICY,
    UNSUPPORTED_LANGUAGE, UNCERTAIN_MODEL_SELECTION, MISSING_BUNDLE, BUDGET_EXHAUSTED,
}

/** Typed failure causes (REQ-API-009). Messages never carry user state (REQ-PRV-007). */
public sealed class DecisionError {
    public abstract val message: String

    public data class InvalidSchema(override val message: String) : DecisionError()
    public data class UnsupportedCapability(override val message: String) : DecisionError()
    public data class ContextOverflow(public val requiredTokens: Int, public val limitTokens: Int) : DecisionError() {
        override val message: String get() = "the packed request needs $requiredTokens tokens, the context holds $limitTokens"
    }
    public data class ModelCorruption(override val message: String) : DecisionError()
    public data class ResourceExhausted(override val message: String) : DecisionError()
    public data object Cancelled : DecisionError() { override val message: String get() = "cancelled" }
    public data class NumericalFailure(override val message: String) : DecisionError()
}

/** The outcome of one question (REQ-API-008). */
public sealed class QuestionOutcome {
    public abstract val questionId: String

    public data class Accepted(override val questionId: String, public val decision: Decision) : QuestionOutcome()

    /** [candidate] is a diagnostic value only; it is not a decision (REQ-API-011). */
    public data class Abstained(override val questionId: String, public val reason: AbstentionReason, public val candidate: Prediction?) : QuestionOutcome()

    public data class Failed(override val questionId: String, public val error: DecisionError) : QuestionOutcome()
}

/** Outcomes in the request's question order plus provenance (REQ-API-010). */
public data class DecisionResult(
    public val requestId: String,
    public val outcomes: List<QuestionOutcome>,
    public val provenance: Provenance,
) {
    public fun question(id: String): QuestionOutcome = outcomes.firstOrNull { it.questionId == id }
        ?: throw NoSuchElementException("no question '$id' in request '$requestId'")
}

/** A request-level failure; no question-level predictions are fabricated (REQ-API-008). */
public class DecisionRequestException(public val error: DecisionError) : RuntimeException(error.message)
