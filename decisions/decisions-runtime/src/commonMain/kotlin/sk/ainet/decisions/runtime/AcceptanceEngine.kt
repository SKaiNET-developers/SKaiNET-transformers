package sk.ainet.decisions.runtime

import sk.ainet.decisions.AbstentionReason
import sk.ainet.decisions.AcceptancePolicy
import sk.ainet.decisions.Decision
import sk.ainet.decisions.DecisionError
import sk.ainet.decisions.DecisionQuestion
import sk.ainet.decisions.Prediction
import sk.ainet.decisions.QuestionOutcome
import sk.ainet.decisions.ScoreAcceptance

/**
 * Turns a candidate into an outcome under the application's policy (REQ-POL). Thresholds are
 * inclusive; a primitive without a policy entry abstains with [AbstentionReason.NO_POLICY];
 * a non-finite candidate is a [DecisionError.NumericalFailure], never an acceptance.
 */
public object AcceptanceEngine {

    public fun apply(question: DecisionQuestion, prediction: Prediction, policy: AcceptancePolicy?): QuestionOutcome {
        if (!isFinite(prediction)) return QuestionOutcome.Failed(question.id, DecisionError.NumericalFailure("candidate holds a non-finite probability"))
        if (policy == null) return QuestionOutcome.Abstained(question.id, AbstentionReason.NO_POLICY, prediction)
        return when (prediction) {
            is Prediction.Choice -> {
                val rule = policy.choice ?: return abstain(question, AbstentionReason.NO_POLICY, prediction)
                rule.minConfidence?.let { if (prediction.confidence.value < it) return abstain(question, AbstentionReason.BELOW_CONFIDENCE, prediction) }
                rule.minMargin?.let { if (prediction.margin < it) return abstain(question, AbstentionReason.BELOW_MARGIN, prediction) }
                QuestionOutcome.Accepted(question.id, Decision.Choice(prediction.selectedOptionId, prediction))
            }
            is Prediction.Score -> {
                val rule = policy.score ?: return abstain(question, AbstentionReason.NO_POLICY, prediction)
                val ok = when (rule) {
                    is ScoreAcceptance.MinTopProbability -> prediction.distribution.max() >= rule.min
                    is ScoreAcceptance.MaxNormalizedEntropy -> prediction.normalizedEntropy <= rule.max
                }
                if (ok) QuestionOutcome.Accepted(question.id, Decision.Score(prediction.selectedLevel, prediction))
                else abstain(question, AbstentionReason.BELOW_CONFIDENCE, prediction)
            }
            is Prediction.BooleanProbability -> {
                val rule = policy.boolean ?: return abstain(question, AbstentionReason.NO_POLICY, prediction)
                val p = prediction.probabilityTrue
                when {
                    p <= rule.lower -> QuestionOutcome.Accepted(question.id, Decision.Boolean(false, prediction))
                    p >= rule.upper -> QuestionOutcome.Accepted(question.id, Decision.Boolean(true, prediction))
                    else -> abstain(question, AbstentionReason.BETWEEN_THRESHOLDS, prediction)
                }
            }
        }
    }

    private fun abstain(question: DecisionQuestion, reason: AbstentionReason, candidate: Prediction) =
        QuestionOutcome.Abstained(question.id, reason, candidate)

    private fun isFinite(p: Prediction): Boolean = when (p) {
        is Prediction.Choice -> p.probabilities.all { it.probability.isFinite() } && p.confidence.value.isFinite()
        is Prediction.Score -> p.distribution.all { it.isFinite() } && p.expected.isFinite()
        is Prediction.BooleanProbability -> p.probabilityTrue.isFinite()
    }
}
