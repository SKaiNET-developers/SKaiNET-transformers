package sk.ainet.models.laya

import kotlinx.serialization.json.JsonPrimitive
import sk.ainet.decisions.AdapterOutcome
import sk.ainet.decisions.BooleanProbabilityQuestion
import sk.ainet.decisions.ChoiceQuestion
import sk.ainet.decisions.Confidence
import sk.ainet.decisions.ConfidenceDefinition
import sk.ainet.decisions.DecisionError
import sk.ainet.decisions.DecisionQuestion
import sk.ainet.decisions.DecisionState
import sk.ainet.decisions.InferencePolicy
import sk.ainet.decisions.OptionProbability
import sk.ainet.decisions.Prediction
import sk.ainet.decisions.QuestionTrace
import sk.ainet.decisions.ScoreQuestion
import sk.ainet.decisions.TruncationPolicy

/**
 * The typed-decision plumbing both Laya adapters share: typed question → reference question, state
 * packing with the truncation policy, logits → candidate with a packing trace.
 */
internal object LayaAdapterSupport {

    internal sealed interface Packed {
        class Failed(val outcome: AdapterOutcome.Failure) : Packed
        class Ready(val question: DecisionQuestion, val laya: LayaQuestion, val sequence: LayaSequence) : Packed
    }

    fun toLayaState(state: DecisionState): LayaState = when (state) {
        is DecisionState.Text -> LayaState.Text(state.text)
        is DecisionState.Structured -> LayaState.Structured(state.value)
    }

    /** Typed question → the reference question shape; option ids are the model-facing keys. */
    fun toLaya(question: DecisionQuestion): LayaQuestion = when (question) {
        is ChoiceQuestion -> LayaQuestion.Choice(question.instructions, question.options.map { it.id to JsonPrimitive(it.description) })
        is ScoreQuestion -> LayaQuestion.Score(question.instructions, question.levels.map { JsonPrimitive(it.description) })
        is BooleanProbabilityQuestion -> LayaQuestion.Noul(question.instructions, JsonPrimitive(question.criteria.whenFalse), JsonPrimitive(question.criteria.whenTrue))
    }

    fun pack(preprocessor: LayaPreprocessor, state: LayaState, stateIds: IntArray, question: DecisionQuestion, inference: InferencePolicy): Packed {
        val laya = toLaya(question)
        val maxLen = preprocessor.agentConfig.maxLen
        val sequence = try {
            preprocessor.prepare(state, laya, stateIds = stateIds)
        } catch (e: LayaSequenceException) {
            return Packed.Failed(AdapterOutcome.Failure(question.id, DecisionError.ContextOverflow(requiredTokens = maxLen + 1, limitTokens = maxLen)))
        }
        if (sequence.truncation.truncated && inference.truncation == TruncationPolicy.REJECT) {
            val required = sequence.length + sequence.truncation.stateTokensDropped
            return Packed.Failed(AdapterOutcome.Failure(question.id, DecisionError.ContextOverflow(requiredTokens = required, limitTokens = maxLen)))
        }
        return Packed.Ready(question, laya, sequence)
    }

    fun outcome(decoder: LayaDecoder, packed: Packed.Ready, logits: FloatArray): AdapterOutcome {
        val question = packed.question
        val laya = packed.laya
        val sequence = packed.sequence
        if (logits.any { !it.isFinite() }) return AdapterOutcome.Failure(question.id, DecisionError.NumericalFailure("non-finite logit"))
        val answer = decoder.decode(laya, logits, sequence.optionOrder)
        val prediction: Prediction = when (question) {
            is ChoiceQuestion -> {
                val a = answer as LayaAnswer.Choice
                Prediction.Choice(
                    selectedOptionId = a.choice,
                    probabilities = question.options.mapIndexed { i, o -> OptionProbability(o.id, a.probabilities[i]) },
                    confidence = Confidence(a.confidence, ConfidenceDefinition.NORMALIZED_ENTROPY),
                )
            }
            is ScoreQuestion -> {
                val a = answer as LayaAnswer.Score
                Prediction.Score(
                    expected = a.score,
                    selectedLevel = a.probabilities.indices.maxBy { a.probabilities[it] },
                    distribution = a.probabilities.toList(),
                    confidence = Confidence(a.confidence, ConfidenceDefinition.NORMALIZED_ENTROPY),
                )
            }
            is BooleanProbabilityQuestion -> {
                val a = answer as LayaAnswer.Noul
                Prediction.BooleanProbability(probabilityTrue = a.noul, confidence = Confidence(a.confidence, ConfidenceDefinition.TWO_WAY_MAX))
            }
        }
        val trace = QuestionTrace(question.id, sequence.length, sequence.truncation.stateTokens, sequence.truncation.stateTokensDropped)
        return AdapterOutcome.Candidate(question.id, prediction, trace)
    }
}
