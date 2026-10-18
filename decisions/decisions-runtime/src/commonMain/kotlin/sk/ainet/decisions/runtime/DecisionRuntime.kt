package sk.ainet.decisions.runtime

import sk.ainet.decisions.AcceptancePolicy
import sk.ainet.decisions.AdapterOutcome
import sk.ainet.decisions.DecisionAdapter
import sk.ainet.decisions.DecisionError
import sk.ainet.decisions.DecisionRequest
import sk.ainet.decisions.DecisionRequestException
import sk.ainet.decisions.DecisionResult
import sk.ainet.decisions.LanguageSupport
import sk.ainet.decisions.ModelSelector
import sk.ainet.decisions.Provenance
import sk.ainet.decisions.QuestionOutcome
import kotlin.time.TimeSource

/**
 * Holds the installed adapters and opens sessions on them. Nothing here can reach a remote
 * service: adapters are constructed by the application from local artifacts (REQ-API-012).
 */
public class DecisionRuntime(adapters: Collection<DecisionAdapter>) : AutoCloseable {
    private val byModel: Map<String, DecisionAdapter> = adapters.associateBy { it.capabilities.modelId }

    init {
        require(byModel.size == adapters.size) { "adapters must declare distinct model ids" }
    }

    public val installedModels: Set<String> get() = byModel.keys

    public fun openSession(model: ModelSelector): DecisionSession {
        val id = (model as ModelSelector.Installed).modelId
        val adapter = byModel[id] ?: throw DecisionRequestException(DecisionError.UnsupportedCapability("no installed model '$id'"))
        return DecisionSession(adapter)
    }

    override fun close() { byModel.values.forEach { it.close() } }
}

/**
 * One request pipeline per adapter: validate against capabilities → adapter candidates →
 * acceptance policy → result with provenance. The adapter never sees the policy (REQ-API-016).
 */
public class DecisionSession internal constructor(private val adapter: DecisionAdapter) : AutoCloseable {
    private var closed = false

    public val capabilities: sk.ainet.decisions.AdapterCapabilities get() = adapter.capabilities

    /**
     * Decides every question of [request]. Request-level problems (closed session, model mismatch,
     * limits evaluable from the request alone) raise [DecisionRequestException]; question-level
     * problems become [QuestionOutcome.Failed]. Without [acceptance] every candidate abstains with
     * [sk.ainet.decisions.AbstentionReason.NO_POLICY].
     */
    public fun decide(request: DecisionRequest, acceptance: AcceptancePolicy? = null): DecisionResult {
        if (closed) throw DecisionRequestException(DecisionError.ResourceExhausted("session is closed"))
        val started = TimeSource.Monotonic.markNow()
        RequestValidator.validate(request, adapter.capabilities)
        val notes = ArrayList<String>()
        val adapterOutcomes = adapter.predict(request.state, request.questions, request.inference)
        require(adapterOutcomes.map { it.questionId } == request.questions.map { it.id }) {
            "adapter '${adapter.capabilities.modelId}' returned outcomes out of order or incomplete"
        }
        val outcomes = ArrayList<QuestionOutcome>(request.questions.size)
        val traces = ArrayList<sk.ainet.decisions.QuestionTrace>()
        for ((question, outcome) in request.questions.zip(adapterOutcomes)) {
            when (outcome) {
                is AdapterOutcome.Failure -> outcomes += QuestionOutcome.Failed(question.id, outcome.error)
                is AdapterOutcome.Candidate -> {
                    traces += outcome.trace
                    if (outcome.trace.truncated) notes += "question ${question.id}: ${outcome.trace.stateTokensDropped} state tokens dropped from the ${if (request.state is sk.ainet.decisions.DecisionState.Structured && (request.state as sk.ainet.decisions.DecisionState.Structured).value is kotlinx.serialization.json.JsonArray) "start" else "end"}"
                    outcomes += AcceptanceEngine.apply(question, outcome.prediction, acceptance)
                }
            }
        }
        val caps = adapter.capabilities
        return DecisionResult(
            requestId = request.id,
            outcomes = outcomes,
            provenance = Provenance(
                modelId = caps.modelId,
                backend = caps.backend,
                precision = caps.precision,
                preprocessingVersion = caps.preprocessingVersion,
                calibration = caps.calibration,
                calibrationIdentity = caps.calibrationIdentity,
                policyId = acceptance?.id,
                policyVersion = acceptance?.version,
                traces = traces,
                durationMillis = started.elapsedNow().inWholeMilliseconds,
                notes = notes,
            ),
        )
    }

    override fun close() { closed = true }
}

/** Request checks evaluable without inference (REQ-API-015, REQ-RES-003). */
public object RequestValidator {
    public fun validate(request: DecisionRequest, caps: sk.ainet.decisions.AdapterCapabilities) {
        val model = (request.model as ModelSelector.Installed).modelId
        if (model != caps.modelId) throw DecisionRequestException(DecisionError.UnsupportedCapability("request selects model '$model', session holds '${caps.modelId}'"))
        if (request.questions.size > caps.maxQuestionsPerRequest) {
            throw DecisionRequestException(DecisionError.ResourceExhausted("${request.questions.size} questions exceed the limit of ${caps.maxQuestionsPerRequest}"))
        }
        for (q in request.questions) {
            if (q.primitive !in caps.primitives) throw DecisionRequestException(DecisionError.UnsupportedCapability("question '${q.id}': ${q.primitive} is not supported by '${caps.modelId}'"))
            if (q.optionCount > caps.maxOptions) throw DecisionRequestException(DecisionError.InvalidSchema("question '${q.id}': ${q.optionCount} options exceed the limit of ${caps.maxOptions}"))
        }
        val language = request.inference.language
        val support = caps.languages
        if (language != null && support is LanguageSupport.Only && language.lowercase().substringBefore('-') !in support.languages.map { it.lowercase() }) {
            throw DecisionRequestException(DecisionError.UnsupportedCapability("language '$language' is not supported by '${caps.modelId}'"))
        }
    }
}
