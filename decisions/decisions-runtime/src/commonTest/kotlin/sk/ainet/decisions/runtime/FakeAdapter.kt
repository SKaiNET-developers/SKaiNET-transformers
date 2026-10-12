package sk.ainet.decisions.runtime

import sk.ainet.decisions.AdapterCapabilities
import sk.ainet.decisions.AdapterOutcome
import sk.ainet.decisions.BooleanProbabilityQuestion
import sk.ainet.decisions.CalibrationStatus
import sk.ainet.decisions.ChoiceQuestion
import sk.ainet.decisions.Confidence
import sk.ainet.decisions.ConfidenceDefinition
import sk.ainet.decisions.DecisionAdapter
import sk.ainet.decisions.DecisionError
import sk.ainet.decisions.DecisionQuestion
import sk.ainet.decisions.DecisionState
import sk.ainet.decisions.InferencePolicy
import sk.ainet.decisions.LanguageSupport
import sk.ainet.decisions.OptionProbability
import sk.ainet.decisions.Prediction
import sk.ainet.decisions.Primitive
import sk.ainet.decisions.QuestionTrace
import sk.ainet.decisions.ScoreQuestion
import kotlin.math.ln

/**
 * Deterministic adapter for the conformance tests: the candidate for a question is scripted per
 * question id; unscripted questions get a uniform distribution. Never touches the state.
 */
internal class FakeAdapter(
    modelId: String = "fake",
    languages: LanguageSupport = LanguageSupport.Unrestricted,
    maxOptions: Int = 8,
    private val scripted: Map<String, Prediction> = emptyMap(),
    private val failing: Map<String, DecisionError> = emptyMap(),
) : DecisionAdapter {
    var closed = false
    var calls = 0

    override val capabilities = AdapterCapabilities(
        modelId = modelId, primitives = setOf(Primitive.CHOICE, Primitive.SCORE, Primitive.BOOLEAN_PROBABILITY), languages = languages,
        maxContextTokens = 64, maxOptionTokens = 16, maxOptions = maxOptions, maxQuestionsPerRequest = 4, batchSupport = false,
        backend = "fake", precision = "fp64", calibration = CalibrationStatus.UNVERIFIED, calibrationIdentity = "none", preprocessingVersion = "fake/1",
    )

    override fun predict(state: DecisionState, questions: List<DecisionQuestion>, inference: InferencePolicy): List<AdapterOutcome> {
        calls++
        return questions.map { q ->
            failing[q.id]?.let { return@map AdapterOutcome.Failure(q.id, it) }
            val p = scripted[q.id] ?: uniform(q)
            AdapterOutcome.Candidate(q.id, p, QuestionTrace(q.id, packedTokens = 10 + q.optionCount, stateTokens = 4, stateTokensDropped = 0))
        }
    }

    private fun uniform(q: DecisionQuestion): Prediction = when (q) {
        is ChoiceQuestion -> Prediction.Choice(q.options.first().id, q.options.map { OptionProbability(it.id, 1.0 / q.options.size) }, Confidence(0.0, ConfidenceDefinition.NORMALIZED_ENTROPY))
        is ScoreQuestion -> Prediction.Score((q.levels.size - 1) / 2.0, 0, List(q.levels.size) { 1.0 / q.levels.size }, Confidence(0.0, ConfidenceDefinition.NORMALIZED_ENTROPY))
        is BooleanProbabilityQuestion -> Prediction.BooleanProbability(0.5, Confidence(0.5, ConfidenceDefinition.TWO_WAY_MAX))
    }

    override fun close() { closed = true }

    companion object {
        fun choice(selected: String, vararg probs: Pair<String, Double>): Prediction.Choice {
            val p = probs.map { it.second }
            val entropy = -p.sumOf { if (it > 0) it * ln(it) else 0.0 } / ln(p.size.toDouble())
            return Prediction.Choice(selected, probs.map { OptionProbability(it.first, it.second) }, Confidence((1 - entropy).coerceIn(0.0, 1.0), ConfidenceDefinition.NORMALIZED_ENTROPY))
        }
        fun score(vararg p: Double): Prediction.Score {
            val entropy = -p.sumOf { if (it > 0) it * ln(it) else 0.0 } / ln(p.size.toDouble())
            return Prediction.Score(p.indices.sumOf { it * p[it] }, p.indices.maxBy { p[it] }, p.toList(), Confidence((1 - entropy).coerceIn(0.0, 1.0), ConfidenceDefinition.NORMALIZED_ENTROPY))
        }
        fun bool(pTrue: Double) = Prediction.BooleanProbability(pTrue, Confidence(maxOf(pTrue, 1 - pTrue), ConfidenceDefinition.TWO_WAY_MAX))
    }
}
