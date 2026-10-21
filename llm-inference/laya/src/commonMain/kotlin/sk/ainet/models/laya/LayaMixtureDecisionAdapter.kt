package sk.ainet.models.laya

import sk.ainet.decisions.AdapterCapabilities
import sk.ainet.decisions.AdapterOutcome
import sk.ainet.decisions.CalibrationStatus
import sk.ainet.decisions.DecisionAdapter
import sk.ainet.decisions.DecisionQuestion
import sk.ainet.decisions.DecisionState
import sk.ainet.decisions.InferencePolicy
import sk.ainet.decisions.LanguageSupport
import sk.ainet.lang.types.DType

/**
 * A [LayaMixtureAgent] behind the typed decision contract: every request is routed once (the base
 * head answers the router question about the state), then the chosen head answers the questions
 * with its own calibration. The contract has no field for the head that answered, so [onRouted]
 * reports each routing to the caller; acceptance stays with the runtime as for [LayaDecisionAdapter].
 */
public class LayaMixtureDecisionAdapter<T : DType>(
    public val mixture: LayaMixtureAgent<T>,
    modelId: String,
    languages: LanguageSupport = LanguageSupport.Unrestricted,
    backend: String = "skainet-cpu",
    precision: String = "fp32",
    private val onRouted: (LayaRouting) -> Unit = {},
    private val onClose: () -> Unit = {},
) : DecisionAdapter {

    override val capabilities: AdapterCapabilities = LayaDecisionAdapter.capabilitiesOf(
        modelId, languages, backend, precision, mixture.base.preprocessor.agentConfig,
        calibration = CalibrationStatus.CHECKPOINT_PROVIDED,
        calibrationIdentity = "rl_agent_config.json + expert.json(" + mixture.experts.keys.joinToString(",") + ")",
    )

    override fun predict(state: DecisionState, questions: List<DecisionQuestion>, inference: InferencePolicy): List<AdapterOutcome> {
        val base = mixture.base
        val pre = base.preprocessor
        val layaState = LayaAdapterSupport.toLayaState(state)
        val stateIds = pre.builder.encodeState(layaState)
        val packed = questions.map { question -> LayaAdapterSupport.pack(pre, layaState, stateIds, question, inference) }
        val ready = packed.filterIsInstance<LayaAdapterSupport.Packed.Ready>()
        if (ready.isEmpty()) return packed.map { (it as LayaAdapterSupport.Packed.Failed).outcome }

        // Route once per request, on the base head.
        val rq = mixture.router.layaQuestion
        val rs = pre.prepare(layaState, rq, stateIds = stateIds)
        val rAnswer = base.decoder.decode(rq, base.model.logits(rs, rq.type), rs.optionOrder) as LayaAnswer.Choice
        val routing = mixture.router.route(rAnswer.byKey, mixture.experts.keys, rAnswer.confidence)
        onRouted(routing)
        val (head, decoder) = if (routing.chosen == LayaMixtureAgent.GENERAL) {
            base.model.defaultHead to base.decoder
        } else {
            mixture.experts.getValue(routing.chosen).let { it.head to it.decoder }
        }

        val logits = base.logitsFor(ready.map { it.sequence }, ready.map { it.laya.type }, head)
        var next = 0
        return packed.map { p ->
            when (p) {
                is LayaAdapterSupport.Packed.Failed -> p.outcome
                is LayaAdapterSupport.Packed.Ready -> LayaAdapterSupport.outcome(decoder, p, logits[next++])
            }
        }
    }

    override fun close(): Unit = onClose()
}
