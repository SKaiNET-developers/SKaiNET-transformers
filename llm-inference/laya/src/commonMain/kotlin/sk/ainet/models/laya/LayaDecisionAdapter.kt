package sk.ainet.models.laya

import kotlinx.serialization.json.JsonPrimitive
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
import sk.ainet.decisions.TruncationPolicy
import sk.ainet.lang.types.DType

/**
 * The Laya checkpoint behind the typed decision contract (REQ-API-014/016): maps the typed
 * questions onto the reference question shapes, runs the preprocessor, model and decoder, and
 * returns candidates with packing traces. Acceptance stays with the runtime.
 */
public class LayaDecisionAdapter<T : DType>(
    public val agent: LayaAgent<T>,
    modelId: String,
    languages: LanguageSupport = LanguageSupport.Unrestricted,
    backend: String = "skainet-cpu",
    precision: String = "fp32",
    private val onClose: () -> Unit = {},
) : DecisionAdapter {

    override val capabilities: AdapterCapabilities = run {
        val cfg = agent.preprocessor.agentConfig
        val provided = cfg.temperatureByOptions.isNotEmpty() || cfg.temperature.any { it != 1.0 }
        AdapterCapabilities(
            modelId = modelId,
            primitives = setOf(Primitive.CHOICE, Primitive.SCORE, Primitive.BOOLEAN_PROBABILITY),
            languages = languages,
            maxContextTokens = cfg.maxLen,
            maxOptionTokens = cfg.headMaxLen,
            // Every option costs at least the mask token plus the floor; the head budget bounds the count.
            maxOptions = maxOf(2, (cfg.headMaxLen - LayaSequenceBuilder.MIN_OPTION_BUDGET) / LayaSequenceBuilder.MIN_OPTION_TOKENS),
            maxQuestionsPerRequest = 64,
            batchSupport = false,
            backend = backend,
            precision = precision,
            calibration = if (provided) CalibrationStatus.CHECKPOINT_PROVIDED else CalibrationStatus.UNVERIFIED,
            calibrationIdentity = if (provided) "rl_agent_config.json" else "none",
            preprocessingVersion = PREPROCESSING_VERSION,
        )
    }

    override fun predict(state: DecisionState, questions: List<DecisionQuestion>, inference: InferencePolicy): List<AdapterOutcome> {
        val layaState = when (state) {
            is DecisionState.Text -> LayaState.Text(state.text)
            is DecisionState.Structured -> LayaState.Structured(state.value)
        }
        val stateIds = agent.preprocessor.builder.encodeState(layaState)
        return questions.map { question -> predictOne(layaState, stateIds, question, inference) }
    }

    private fun predictOne(state: LayaState, stateIds: IntArray, question: DecisionQuestion, inference: InferencePolicy): AdapterOutcome {
        val laya = toLaya(question)
        val sequence = try {
            agent.preprocessor.prepare(state, laya, stateIds = stateIds)
        } catch (e: LayaSequenceException) {
            return AdapterOutcome.Failure(question.id, DecisionError.ContextOverflow(requiredTokens = agent.preprocessor.agentConfig.maxLen + 1, limitTokens = agent.preprocessor.agentConfig.maxLen))
        }
        if (sequence.truncation.truncated && inference.truncation == TruncationPolicy.REJECT) {
            val required = sequence.length + sequence.truncation.stateTokensDropped
            return AdapterOutcome.Failure(question.id, DecisionError.ContextOverflow(requiredTokens = required, limitTokens = agent.preprocessor.agentConfig.maxLen))
        }
        val logits = agent.model.logits(sequence, laya.type)
        if (logits.any { !it.isFinite() }) return AdapterOutcome.Failure(question.id, DecisionError.NumericalFailure("non-finite logit"))
        val answer = agent.decoder.decode(laya, logits, sequence.optionOrder)
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

    /** Typed question → the reference question shape; option ids are the model-facing keys. */
    public fun toLaya(question: DecisionQuestion): LayaQuestion = when (question) {
        is ChoiceQuestion -> LayaQuestion.Choice(question.instructions, question.options.map { it.id to JsonPrimitive(it.description) })
        is ScoreQuestion -> LayaQuestion.Score(question.instructions, question.levels.map { JsonPrimitive(it.description) })
        is BooleanProbabilityQuestion -> LayaQuestion.Noul(question.instructions, JsonPrimitive(question.criteria.whenFalse), JsonPrimitive(question.criteria.whenTrue))
    }

    override fun close(): Unit = onClose()

    public companion object {
        /** The packing rules this adapter implements: Laya `build_sequence` as of release v0.4.2. */
        public const val PREPROCESSING_VERSION: String = "laya-build_sequence/0.4.2"
    }
}
