package sk.ainet.models.laya

import kotlinx.serialization.json.JsonPrimitive
import sk.ainet.lang.types.DType
import kotlin.time.TimeSource

/**
 * The prompted router of a mixture of expert heads: the base head answers [question] — a choice
 * over fine-grained kinds of text, each with a description — about the user's state, and every
 * kind maps to an expert through [experts]; a kind without a mapping, an expert that is not loaded,
 * or a top probability under [threshold] all route to the base head (`general`).
 *
 * The kinds are deliberately finer than the experts ("what kind of text is this?" is an easier
 * question for the base model than "which expert should answer?").
 */
public class LayaRouterSpec(
    public val question: String,
    /** Kind id → description, in option order. */
    public val options: List<Pair<String, String>>,
    /** Kind id → expert name; kinds not listed route to [LayaMixtureAgent.GENERAL]. */
    public val experts: Map<String, String>,
    public val threshold: Double = DEFAULT_THRESHOLD,
) {
    init {
        require(options.size >= 2) { "a router needs at least two kinds" }
        require(options.map { it.first }.toSet().size == options.size) { "router kinds must be distinct" }
        require(threshold in 0.0..1.0) { "threshold must be a probability, got $threshold" }
        val unknown = experts.keys.filter { kind -> options.none { it.first == kind } }
        require(unknown.isEmpty()) { "router maps kinds that are not options: $unknown" }
    }

    /** The router question as the base head answers it. */
    public val layaQuestion: LayaQuestion.Choice = LayaQuestion.Choice(question, options.map { (kind, description) -> kind to JsonPrimitive(description) })

    /**
     * The routing decision for the base head's [probabilities] over the kinds: the top kind's
     * expert when it is in [available] and the top probability reaches [threshold], else `general`;
     * [forced] overrides the pick but is recorded beside it.
     */
    public fun route(probabilities: Map<String, Double>, available: Set<String>, confidence: Double, forced: String? = null): LayaRouting {
        require(probabilities.isNotEmpty()) { "routing needs probabilities" }
        val top = probabilities.maxBy { it.value }.key
        val topProbability = probabilities.getValue(top)
        val mapped = experts[top]
        val pick = if (mapped != null && mapped in available && topProbability >= threshold) mapped else LayaMixtureAgent.GENERAL
        return LayaRouting(probabilities, top, topProbability, confidence, threshold, pick, forced ?: pick)
    }

    public companion object {
        public const val DEFAULT_THRESHOLD: Double = 0.3
    }
}

/** What the router saw and decided for one request. */
public class LayaRouting(
    /** The base head's probabilities over the router kinds, in option order. */
    public val probabilities: Map<String, Double>,
    /** The most probable kind. */
    public val top: String,
    public val topProbability: Double,
    /** The entropy confidence of the router answer (the reference reports it as `confidence`). */
    public val confidence: Double,
    public val threshold: Double,
    /** What the router picked: the mapped expert, or `general`. */
    public val routerPick: String,
    /** What answered: [routerPick] unless the caller forced a head. */
    public val chosen: String,
) {
    public val overridden: Boolean get() = chosen != routerPick
}

/** The answers of one request to a mixture, with the routing and the cost. */
public class LayaMixtureResult(
    /** The head that answered: an expert name or [LayaMixtureAgent.GENERAL]. */
    public val expert: String,
    /** `null` when routing was skipped. */
    public val routing: LayaRouting?,
    /** The answers of [expert], keyed like the questions. */
    public val answers: Map<String, LayaAnswer>,
    /** The base head's answers on the same encoder output, when a comparison was asked for and an expert answered. */
    public val generalAnswers: Map<String, LayaAnswer>?,
    /** The packed sequences, keyed like the questions. */
    public val sequences: Map<String, LayaSequence>,
    /** Encoder time: the questions' passes and the router's pass. */
    public val encoderMillis: Long,
    /** Head time: the router decode, the chosen head, the comparison head. */
    public val headsMillis: Long,
    /** Tokens the encoder read, router question included. */
    public val inputTokens: Int,
)

/**
 * One base [LayaAgent] (its encoder and `general` head), the [experts] trained on top of it, and
 * the [router] that picks one head per request. The encoder runs once per question — the sequence
 * contains the question — and the router question costs one more pass; heads are two blocks and
 * cost little, so a comparison with `general` runs on the same encoder output.
 */
public class LayaMixtureAgent<T : DType>(
    public val base: LayaAgent<T>,
    experts: Collection<LayaExpert<T>>,
    public val router: LayaRouterSpec,
) {
    public val experts: Map<String, LayaExpert<T>> = experts.associateBy { it.name }

    init {
        require(GENERAL !in this.experts) { "'$GENERAL' is the base head and cannot be an expert name" }
        require(this.experts.size == experts.size) { "expert names must be distinct" }
        for (e in experts) require(e.head.hiddenSize == base.model.hiddenSize) { "expert '${e.name}' has hidden size ${e.head.hiddenSize}, the base model ${base.model.hiddenSize}" }
    }

    /** `general` first, then the experts in registration order. */
    public val heads: List<String> = listOf(GENERAL) + this.experts.keys

    /**
     * Answers [questions] about [state] with the routed head — or with [expert] when given (the
     * routing is still computed and reported, unless [route] is `false`). With [compare], the base
     * head answers too when an expert answered.
     */
    public fun decide(state: LayaState, questions: Map<String, LayaQuestion>, expert: String? = null, compare: Boolean = false, route: Boolean = true): LayaMixtureResult {
        require(expert == null || expert == GENERAL || expert in experts) { "unknown expert '$expert'; have $heads" }
        require(questions.isNotEmpty()) { "a request needs at least one question" }
        val clock = TimeSource.Monotonic
        val pre = base.preprocessor
        val stateIds = pre.builder.encodeState(state)
        val keys = questions.keys.toList()
        val asked = keys.map { questions.getValue(it) }
        val types = asked.map { it.type }
        val sequences = asked.map { pre.prepare(state, it, stateIds = stateIds) }

        var encoderMillis = 0L
        var headsMillis = 0L
        var t = clock.markNow()
        val encoded = base.encodeFor(sequences)
        encoderMillis += t.elapsedNow().inWholeMilliseconds

        var routing: LayaRouting? = null
        var routerTokens = 0
        if (route) {
            val rq = router.layaQuestion
            val rs = pre.prepare(state, rq, stateIds = stateIds)
            routerTokens = rs.length
            t = clock.markNow()
            val rEncoded = base.model.encode(rs.ids)
            encoderMillis += t.elapsedNow().inWholeMilliseconds
            t = clock.markNow()
            val rLogits = base.model.logitsFrom(rEncoded, rs, rq.type, base.model.defaultHead)
            val answer = base.decoder.decode(rq, rLogits, rs.optionOrder) as LayaAnswer.Choice
            routing = router.route(answer.byKey, experts.keys, answer.confidence, forced = expert)
            headsMillis += t.elapsedNow().inWholeMilliseconds
        }
        val chosen = expert ?: routing?.chosen ?: GENERAL

        t = clock.markNow()
        val (head, decoder) = if (chosen == GENERAL) base.model.defaultHead to base.decoder else experts.getValue(chosen).let { it.head to it.decoder }
        val answers = base.decodeAll(keys, asked, sequences, base.logitsFrom(encoded, types, head), decoder).mapValues { it.value.answer }
        val generalAnswers = if (compare && chosen != GENERAL) {
            base.decodeAll(keys, asked, sequences, base.logitsFrom(encoded, types, base.model.defaultHead), base.decoder).mapValues { it.value.answer }
        } else {
            null
        }
        headsMillis += t.elapsedNow().inWholeMilliseconds

        return LayaMixtureResult(
            expert = chosen,
            routing = routing,
            answers = answers,
            generalAnswers = generalAnswers,
            sequences = keys.indices.associate { keys[it] to sequences[it] },
            encoderMillis = encoderMillis,
            headsMillis = headsMillis,
            inputTokens = encoded.inputTokens + routerTokens,
        )
    }

    public companion object {
        /** The base checkpoint's own head. */
        public const val GENERAL: String = "general"
    }
}
