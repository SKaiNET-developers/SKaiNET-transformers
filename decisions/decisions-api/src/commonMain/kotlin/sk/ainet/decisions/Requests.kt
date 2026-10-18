package sk.ainet.decisions

import kotlinx.serialization.json.JsonElement
import kotlin.jvm.JvmStatic

/** The state a request asks about: free text or a structured JSON value (REQ-API-001). */
public sealed class DecisionState {
    public data class Text(public val text: String) : DecisionState()

    /** Serialised canonically by the adapter (REQ-PRE-001); a JSON array is read as a conversation, newest turn last. */
    public data class Structured(public val value: JsonElement) : DecisionState()

    public companion object {
        @JvmStatic public fun text(text: String): DecisionState = Text(text)
        @JvmStatic public fun structured(value: JsonElement): DecisionState = Structured(value)
    }
}

/** One selectable option of a [ChoiceQuestion]; [id] is the caller's identifier, [description] is what the model reads. */
public data class ChoiceOption(public val id: String, public val description: String) {
    init {
        require(id.isNotBlank()) { "option id must not be blank" }
        require(description.isNotBlank()) { "option '$id' needs a description" }
    }
}

/** One ordered level of a [ScoreQuestion]; levels are indexed 0 until K in list order. */
public data class ScoreLevel(public val description: String) {
    init { require(description.isNotBlank()) { "a score level needs a description" } }
}

/** Explicit criteria for both outcomes of a [BooleanProbabilityQuestion] (REQ-API-005). */
public data class BooleanCriteria(public val whenFalse: String, public val whenTrue: String) {
    init {
        require(whenFalse.isNotBlank() && whenTrue.isNotBlank()) { "both boolean criteria must be given" }
    }
}

/** The three primitives (REQ-API-003..005). `noul` is only the wire alias of the boolean primitive (REQ-API-007). */
public enum class Primitive(public val wire: String) { CHOICE("choice"), SCORE("score"), BOOLEAN_PROBABILITY("noul") }

/** A question about the state; identifiers are unique within a request (REQ-API-002). */
public sealed class DecisionQuestion {
    public abstract val id: String
    public abstract val instructions: String
    public abstract val primitive: Primitive

    /** Number of option slots the model scores. */
    public abstract val optionCount: Int

    protected fun validateCommon() {
        require(id.isNotBlank()) { "question id must not be blank" }
        require(instructions.isNotBlank()) { "question '$id' needs instructions" }
    }
}

public data class ChoiceQuestion(
    override val id: String,
    override val instructions: String,
    public val options: List<ChoiceOption>,
) : DecisionQuestion() {
    init {
        validateCommon()
        require(options.size >= 2) { "choice question '$id' needs at least two options" }
        require(options.map { it.id }.toSet().size == options.size) { "choice question '$id' has duplicate option ids" }
    }
    override val primitive: Primitive get() = Primitive.CHOICE
    override val optionCount: Int get() = options.size
}

public data class ScoreQuestion(
    override val id: String,
    override val instructions: String,
    public val levels: List<ScoreLevel>,
) : DecisionQuestion() {
    init {
        validateCommon()
        require(levels.size >= 2) { "score question '$id' needs at least two levels" }
    }
    override val primitive: Primitive get() = Primitive.SCORE
    override val optionCount: Int get() = levels.size
}

public data class BooleanProbabilityQuestion(
    override val id: String,
    /** The statement whose truth is assessed. */
    override val instructions: String,
    public val criteria: BooleanCriteria,
) : DecisionQuestion() {
    init { validateCommon() }
    override val primitive: Primitive get() = Primitive.BOOLEAN_PROBABILITY
    override val optionCount: Int get() = 2
}

/** Which installed model answers; only local identifiers, never a location (REQ-API-012). */
public sealed class ModelSelector {
    public data class Installed(public val modelId: String) : ModelSelector() {
        init { require(modelId.isNotBlank()) { "model id must not be blank" } }
    }
}

/** What happens when the state does not fit the context (REQ-PRE-006). */
public enum class TruncationPolicy {
    /** Default: a typed [DecisionError.ContextOverflow]. */
    REJECT,
    /** Drop state tokens as the model's packing rule does and report the count in provenance. */
    TRUNCATE_STATE,
}

/** How inference may run; nothing here can enable a remote path. */
public data class InferencePolicy(
    public val truncation: TruncationPolicy = TruncationPolicy.REJECT,
    /** BCP 47 tag of the state language when the caller knows it; adapters may refuse unsupported ones. */
    public val language: String? = null,
) {
    public companion object {
        @JvmStatic public val DEFAULT: InferencePolicy = InferencePolicy()
    }
}

/** A complete request (REQ-API-001). Build with the constructor or [DecisionRequest.builder] from Java. */
public data class DecisionRequest(
    public val id: String,
    public val state: DecisionState,
    public val questions: List<DecisionQuestion>,
    public val model: ModelSelector,
    public val inference: InferencePolicy = InferencePolicy.DEFAULT,
) {
    init {
        require(id.isNotBlank()) { "request id must not be blank" }
        require(questions.isNotEmpty()) { "a request needs at least one question" }
        require(questions.map { it.id }.toSet().size == questions.size) { "request '$id' has duplicate question ids" }
    }

    public class Builder(private val id: String) {
        private var state: DecisionState? = null
        private val questions = ArrayList<DecisionQuestion>()
        private var model: ModelSelector? = null
        private var inference: InferencePolicy = InferencePolicy.DEFAULT

        public fun state(text: String): Builder = apply { state = DecisionState.Text(text) }
        public fun state(value: JsonElement): Builder = apply { state = DecisionState.Structured(value) }
        public fun question(question: DecisionQuestion): Builder = apply { questions += question }
        public fun model(modelId: String): Builder = apply { model = ModelSelector.Installed(modelId) }
        public fun inference(policy: InferencePolicy): Builder = apply { inference = policy }
        public fun build(): DecisionRequest = DecisionRequest(
            id,
            requireNotNull(state) { "request '$id' needs a state" },
            questions.toList(),
            requireNotNull(model) { "request '$id' needs a model" },
            inference,
        )
    }

    public companion object {
        @JvmStatic public fun builder(id: String): Builder = Builder(id)
    }
}
