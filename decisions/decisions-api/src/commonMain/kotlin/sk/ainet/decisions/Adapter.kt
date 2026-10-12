package sk.ainet.decisions

/** Language coverage an adapter declares (REQ-API-015). */
public sealed class LanguageSupport {
    public data object Unrestricted : LanguageSupport()
    public data class Only(public val languages: Set<String>) : LanguageSupport()
}

/** What a model can do; the runtime validates every request against it before scheduling (REQ-API-015). */
public data class AdapterCapabilities(
    public val modelId: String,
    public val primitives: Set<Primitive>,
    public val languages: LanguageSupport,
    public val maxContextTokens: Int,
    /** Head budget in tokens that instructions and options share; questions are rejected when the options alone exceed it. */
    public val maxOptionTokens: Int,
    public val maxOptions: Int,
    public val maxQuestionsPerRequest: Int,
    public val batchSupport: Boolean,
    public val backend: String,
    public val precision: String,
    public val calibration: CalibrationStatus,
    public val calibrationIdentity: String,
    public val preprocessingVersion: String,
)

/** What an adapter returns per question: a candidate with its trace, or a typed failure (REQ-API-016). */
public sealed class AdapterOutcome {
    public abstract val questionId: String
    public data class Candidate(override val questionId: String, public val prediction: Prediction, public val trace: QuestionTrace) : AdapterOutcome()
    public data class Failure(override val questionId: String, public val error: DecisionError) : AdapterOutcome()
}

/**
 * The model-neutral adapter contract (REQ-API-014; working name `DecisionModel` in the
 * specification): capability discovery, typed prediction of candidates, resource release.
 * Acceptance is not the adapter's business; the shared runtime owns it.
 */
public interface DecisionAdapter : AutoCloseable {
    public val capabilities: AdapterCapabilities

    /** Candidates for every question, in request order; one failed question must not fail the others. */
    public fun predict(state: DecisionState, questions: List<DecisionQuestion>, inference: InferencePolicy): List<AdapterOutcome>

    override fun close()
}
