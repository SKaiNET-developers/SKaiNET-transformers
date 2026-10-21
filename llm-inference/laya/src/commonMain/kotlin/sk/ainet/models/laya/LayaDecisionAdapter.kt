package sk.ainet.models.laya

import sk.ainet.decisions.AdapterCapabilities
import sk.ainet.decisions.AdapterOutcome
import sk.ainet.decisions.CalibrationStatus
import sk.ainet.decisions.DecisionAdapter
import sk.ainet.decisions.DecisionQuestion
import sk.ainet.decisions.DecisionState
import sk.ainet.decisions.InferencePolicy
import sk.ainet.decisions.LanguageSupport
import sk.ainet.decisions.Primitive
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
        capabilitiesOf(
            modelId, languages, backend, precision, cfg,
            calibration = if (provided) CalibrationStatus.CHECKPOINT_PROVIDED else CalibrationStatus.UNVERIFIED,
            calibrationIdentity = if (provided) "rl_agent_config.json" else "none",
        )
    }

    override fun predict(state: DecisionState, questions: List<DecisionQuestion>, inference: InferencePolicy): List<AdapterOutcome> {
        val layaState = LayaAdapterSupport.toLayaState(state)
        val stateIds = agent.preprocessor.builder.encodeState(layaState)
        // Pack every question first; the ones that fit run as padded batches (LayaAgent.logitsFor),
        // the others fail individually. Outcomes keep the request order.
        val packed = questions.map { question -> LayaAdapterSupport.pack(agent.preprocessor, layaState, stateIds, question, inference) }
        val ready = packed.filterIsInstance<LayaAdapterSupport.Packed.Ready>()
        val logits = if (ready.isEmpty()) emptyList() else agent.logitsFor(ready.map { it.sequence }, ready.map { it.laya.type })
        var next = 0
        return packed.map { p ->
            when (p) {
                is LayaAdapterSupport.Packed.Failed -> p.outcome
                is LayaAdapterSupport.Packed.Ready -> LayaAdapterSupport.outcome(agent.decoder, p, logits[next++])
            }
        }
    }

    /** Typed question → the reference question shape; option ids are the model-facing keys. */
    public fun toLaya(question: DecisionQuestion): LayaQuestion = LayaAdapterSupport.toLaya(question)

    override fun close(): Unit = onClose()

    public companion object {
        /** The packing rules this adapter implements: Laya `build_sequence` as of release v0.4.2. */
        public const val PREPROCESSING_VERSION: String = "laya-build_sequence/0.4.2"

        internal fun capabilitiesOf(
            modelId: String,
            languages: LanguageSupport,
            backend: String,
            precision: String,
            cfg: LayaAgentConfig,
            calibration: CalibrationStatus,
            calibrationIdentity: String,
        ): AdapterCapabilities = AdapterCapabilities(
            modelId = modelId,
            primitives = setOf(Primitive.CHOICE, Primitive.SCORE, Primitive.BOOLEAN_PROBABILITY),
            languages = languages,
            maxContextTokens = cfg.maxLen,
            maxOptionTokens = cfg.headMaxLen,
            // Every option costs at least the mask token plus the floor; the head budget bounds the count.
            maxOptions = maxOf(2, (cfg.headMaxLen - LayaSequenceBuilder.MIN_OPTION_BUDGET) / LayaSequenceBuilder.MIN_OPTION_TOKENS),
            maxQuestionsPerRequest = 64,
            // The questions of one request run as padded batches (LayaAgent.maxBatchTokens per pass).
            batchSupport = true,
            backend = backend,
            precision = precision,
            calibration = calibration,
            calibrationIdentity = calibrationIdentity,
            preprocessingVersion = PREPROCESSING_VERSION,
        )
    }
}
