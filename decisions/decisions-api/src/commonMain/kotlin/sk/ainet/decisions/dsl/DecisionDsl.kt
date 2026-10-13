package sk.ainet.decisions.dsl

import kotlinx.serialization.json.JsonElement
import sk.ainet.decisions.AcceptancePolicy
import sk.ainet.decisions.BooleanAcceptance
import sk.ainet.decisions.BooleanCriteria
import sk.ainet.decisions.BooleanProbabilityQuestion
import sk.ainet.decisions.ChoiceAcceptance
import sk.ainet.decisions.ChoiceOption
import sk.ainet.decisions.ChoiceQuestion
import sk.ainet.decisions.DecisionQuestion
import sk.ainet.decisions.DecisionRequest
import sk.ainet.decisions.DecisionState
import sk.ainet.decisions.InferencePolicy
import sk.ainet.decisions.InvalidPolicyException
import sk.ainet.decisions.ModelSelector
import sk.ainet.decisions.ScoreAcceptance
import sk.ainet.decisions.ScoreLevel
import sk.ainet.decisions.ScoreQuestion
import sk.ainet.decisions.TruncationPolicy

/**
 * Marks the builder scopes of the decision DSL so that a nested block cannot call the
 * functions of an enclosing scope by accident (an `option(...)` inside `score { }` is a
 * compile error, not a question on the wrong list).
 */
@DslMarker
@Target(AnnotationTarget.CLASS, AnnotationTarget.TYPE)
public annotation class DecisionDsl

// ---- questions ---------------------------------------------------------------

/** The options of one choice question, in the order the model and the result list them. */
@DecisionDsl
public class ChoiceOptions internal constructor() {
    internal val options = ArrayList<ChoiceOption>()

    /** One option; [description] is what the model reads and defaults to the [id] when that already reads well. */
    public fun option(id: String, description: String = id) {
        options += ChoiceOption(id, description)
    }

    /** Several options whose ids double as descriptions. */
    public fun options(vararg ids: String) {
        ids.forEach { option(it) }
    }
}

/** The levels of one score question, lowest first. */
@DecisionDsl
public class ScoreLevels internal constructor() {
    internal val levels = ArrayList<ScoreLevel>()

    public fun level(description: String) {
        levels += ScoreLevel(description)
    }

    public fun levels(vararg descriptions: String) {
        descriptions.forEach { level(it) }
    }
}

/** The two criteria of a boolean probability question; the plain `no` / `yes` pair unless set. */
@DecisionDsl
public class BooleanCriteriaScope internal constructor() {
    public var whenFalse: String = "no"
    public var whenTrue: String = "yes"

    internal fun build(): BooleanCriteria = BooleanCriteria(whenFalse, whenTrue)
}

/** The question vocabulary shared by [decisionRequest] and [decisionSchema]. */
@DecisionDsl
public abstract class QuestionsScope internal constructor() {
    protected val questions: MutableList<DecisionQuestion> = ArrayList()

    /** Adds a question built elsewhere. */
    public fun question(question: DecisionQuestion) {
        questions += question
    }

    /** One option out of a list (REQ-API-003). */
    public fun choice(id: String, instructions: String, options: ChoiceOptions.() -> Unit) {
        question(ChoiceQuestion(id, instructions, ChoiceOptions().apply(options).options))
    }

    /** One option out of [optionIds], each id doubling as its description. */
    public fun choice(id: String, instructions: String, vararg optionIds: String) {
        choice(id, instructions) { options(*optionIds) }
    }

    /** An ordinal scale (REQ-API-004). */
    public fun score(id: String, instructions: String, levels: ScoreLevels.() -> Unit) {
        question(ScoreQuestion(id, instructions, ScoreLevels().apply(levels).levels))
    }

    /** An ordinal scale given as its level descriptions, lowest first. */
    public fun score(id: String, instructions: String, vararg levelDescriptions: String) {
        score(id, instructions) { levels(*levelDescriptions) }
    }

    /** The probability that [statement] is true (REQ-API-005); the wire name of the primitive is `noul`. */
    public fun probability(id: String, statement: String, criteria: BooleanCriteriaScope.() -> Unit = {}) {
        question(BooleanProbabilityQuestion(id, statement, BooleanCriteriaScope().apply(criteria).build()))
    }
}

// ---- acceptance policy ---------------------------------------------------------

@DecisionDsl
public class ChoiceAcceptanceScope internal constructor() {
    /** Minimum entropy confidence `1 − H(p) / log k`, inclusive. */
    public var minConfidence: Double? = null

    /** Minimum gap between the top two probabilities, inclusive. */
    public var minMargin: Double? = null

    internal fun build(): ChoiceAcceptance = ChoiceAcceptance(minConfidence, minMargin)
}

@DecisionDsl
public class ScoreAcceptanceScope internal constructor() {
    private var acceptance: ScoreAcceptance? = null

    /** Accept when the mass on the top level reaches [min]. */
    public fun minTopProbability(min: Double): Unit = set(ScoreAcceptance.MinTopProbability(min))

    /** Accept while the normalised entropy of the level distribution stays at or below [max]. */
    public fun maxNormalizedEntropy(max: Double): Unit = set(ScoreAcceptance.MaxNormalizedEntropy(max))

    private fun set(value: ScoreAcceptance) {
        if (acceptance != null) throw InvalidPolicyException("a score policy takes exactly one criterion, got $acceptance and $value")
        acceptance = value
    }

    internal fun build(): ScoreAcceptance = acceptance ?: throw InvalidPolicyException("score { } needs minTopProbability(...) or maxNormalizedEntropy(...)")
}

@DecisionDsl
public class BooleanAcceptanceScope internal constructor() {
    /** `P(true) <= lower` accepts false. */
    public var lower: Double? = null

    /** `P(true) >= upper` accepts true; in between the question abstains. */
    public var upper: Double? = null

    /** Both thresholds at once: abstain while `lower < P(true) < upper`. */
    public fun abstainBetween(lower: Double, upper: Double) {
        this.lower = lower
        this.upper = upper
    }

    internal fun build(): BooleanAcceptance = BooleanAcceptance(
        lower ?: throw InvalidPolicyException("boolean { } needs a lower threshold"),
        upper ?: throw InvalidPolicyException("boolean { } needs an upper threshold"),
    )
}

/** Builds an [AcceptancePolicy]; a primitive without a block keeps abstaining with `NO_POLICY` (REQ-POL-005). */
@DecisionDsl
public class AcceptancePolicyScope internal constructor(private val id: String, private val version: String) {
    private var choice: ChoiceAcceptance? = null
    private var score: ScoreAcceptance? = null
    private var boolean: BooleanAcceptance? = null

    public fun choice(block: ChoiceAcceptanceScope.() -> Unit) {
        choice = ChoiceAcceptanceScope().apply(block).build()
    }

    public fun score(block: ScoreAcceptanceScope.() -> Unit) {
        score = ScoreAcceptanceScope().apply(block).build()
    }

    public fun boolean(block: BooleanAcceptanceScope.() -> Unit) {
        boolean = BooleanAcceptanceScope().apply(block).build()
    }

    internal fun build(): AcceptancePolicy = AcceptancePolicy(id, version, choice, score, boolean)
}

/**
 * An application-owned, versioned policy (REQ-POL-001):
 * ```
 * val policy = acceptancePolicy("triage", "1") {
 *     choice { minConfidence = 0.2; minMargin = 0.1 }
 *     score { minTopProbability(0.35) }
 *     boolean { abstainBetween(0.4, 0.6) }
 * }
 * ```
 */
public fun acceptancePolicy(id: String, version: String, block: AcceptancePolicyScope.() -> Unit): AcceptancePolicy =
    AcceptancePolicyScope(id, version).apply(block).build()

// ---- inference policy -------------------------------------------------------

/** Shared by request and schema scopes: how inference may run (REQ-PRE-006). */
@DecisionDsl
public abstract class InferenceScope internal constructor() : QuestionsScope() {
    /** Replaces the whole inference policy; the functions below adjust single fields. */
    public var inference: InferencePolicy = InferencePolicy.DEFAULT

    /** An over-long state is truncated as the model's packing rule does, instead of being rejected. */
    public fun truncateState() {
        inference = inference.copy(truncation = TruncationPolicy.TRUNCATE_STATE)
    }

    /** BCP 47 tag of the state language when the caller knows it. */
    public fun language(tag: String) {
        inference = inference.copy(language = tag)
    }
}

// ---- one request --------------------------------------------------------------

@DecisionDsl
public class DecisionRequestScope internal constructor(private val id: String) : InferenceScope() {
    private var state: DecisionState? = null
    private var model: ModelSelector? = null

    public fun state(text: String) {
        state = DecisionState.Text(text)
    }

    public fun state(value: JsonElement) {
        state = DecisionState.Structured(value)
    }

    public fun state(state: DecisionState) {
        this.state = state
    }

    /** The installed model that answers (REQ-API-012). */
    public fun model(modelId: String) {
        model = ModelSelector.Installed(modelId)
    }

    public fun model(selector: ModelSelector) {
        model = selector
    }

    internal fun build(): DecisionRequest = DecisionRequest(
        id,
        requireNotNull(state) { "request '$id' needs a state" },
        questions.toList(),
        requireNotNull(model) { "request '$id' needs a model" },
        inference,
    )
}

/**
 * One complete request:
 * ```
 * val request = decisionRequest("ticket-4711") {
 *     model("laya-multilingual")
 *     state("Customer writes: my invoice shows the old address.")
 *     choice("department", "Which department handles this request?") {
 *         option("billing", "billing and invoices")
 *         option("account", "account data and profile")
 *     }
 *     score("urgency", "How urgent is this request?", "can wait", "this week", "today", "now")
 *     probability("escalate", "Should a human look at this request?")
 * }
 * ```
 */
public fun decisionRequest(id: String, block: DecisionRequestScope.() -> Unit): DecisionRequest =
    DecisionRequestScope(id).apply(block).build()

// ---- a schema: the same questions for many states ----------------------------------

/**
 * Fixed questions, an optional policy and an optional model, applied to one state after another
 * with [request]. The natural shape of a decision model: the questions are the application's
 * design, the state is the item at hand.
 */
public class DecisionSchema(
    public val id: String,
    public val questions: List<DecisionQuestion>,
    public val policy: AcceptancePolicy? = null,
    public val model: ModelSelector? = null,
    public val inference: InferencePolicy = InferencePolicy.DEFAULT,
) {
    init {
        require(id.isNotBlank()) { "schema id must not be blank" }
        require(questions.isNotEmpty()) { "schema '$id' needs at least one question" }
        require(questions.map { it.id }.toSet().size == questions.size) { "schema '$id' has duplicate question ids" }
    }

    /** A request for [state]; [model] defaults to the schema's model and must be given when the schema has none. */
    public fun request(requestId: String, state: DecisionState, model: ModelSelector? = this.model): DecisionRequest = DecisionRequest(
        requestId,
        state,
        questions,
        requireNotNull(model) { "schema '$id' names no model; pass one to request()" },
        inference,
    )

    public fun request(requestId: String, state: String, model: ModelSelector? = this.model): DecisionRequest =
        request(requestId, DecisionState.Text(state), model)

    public fun request(requestId: String, state: JsonElement, model: ModelSelector? = this.model): DecisionRequest =
        request(requestId, DecisionState.Structured(state), model)

    /** The question with [questionId], or null. */
    public fun question(questionId: String): DecisionQuestion? = questions.firstOrNull { it.id == questionId }
}

@DecisionDsl
public class DecisionSchemaScope internal constructor(private val id: String) : InferenceScope() {
    private var policy: AcceptancePolicy? = null
    private var model: ModelSelector? = null

    /** The installed model every request of this schema goes to; can still be overridden per request. */
    public fun model(modelId: String) {
        model = ModelSelector.Installed(modelId)
    }

    public fun model(selector: ModelSelector) {
        model = selector
    }

    /** The acceptance policy; its id defaults to the schema id. */
    public fun policy(version: String, id: String = this.id, block: AcceptancePolicyScope.() -> Unit) {
        policy = acceptancePolicy(id, version, block)
    }

    public fun policy(policy: AcceptancePolicy) {
        this.policy = policy
    }

    internal fun build(): DecisionSchema = DecisionSchema(id, questions.toList(), policy, model, inference)
}

/**
 * Questions and policy declared once, states supplied later:
 * ```
 * val triage = decisionSchema("commit-triage") {
 *     model("laya-multilingual")
 *     choice("kind", "What kind of change is this commit?", "feature", "bugfix", "refactor", "docs", "test", "chore")
 *     score("risk", "How risky is this change for the team?", "trivial", "low", "moderate", "high", "critical or breaking")
 *     probability("standup", "Is this commit worth mentioning in the team's daily stand-up?")
 *     policy(version = "1") {
 *         choice { minConfidence = 0.2; minMargin = 0.1 }
 *         score { minTopProbability(0.35) }
 *         boolean { abstainBetween(0.4, 0.6) }
 *     }
 * }
 * val request = triage.request("commit-7f3a", state = commitText)
 * ```
 */
public fun decisionSchema(id: String, block: DecisionSchemaScope.() -> Unit): DecisionSchema =
    DecisionSchemaScope(id).apply(block).build()
