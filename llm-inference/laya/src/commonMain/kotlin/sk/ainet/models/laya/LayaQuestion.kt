package sk.ainet.models.laya

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonPrimitive

/** The three Laya primitives, with the index the checkpoint's `type_emb` and temperatures use. */
public enum class LayaQuestionType(public val index: Int, public val wire: String) {
    CHOICE(0, "choice"),
    SCORE(1, "score"),
    NOUL(2, "noul"),
}

/**
 * A question as the reference `render_options` sees it. Options are positional: two orders are
 * two different questions, so [LayaQuestion.Choice] takes an ordered list of criteria.
 */
public sealed class LayaQuestion {
    public abstract val type: LayaQuestionType

    /** The instruction text. A structured instruction is JSON-serialised by the caller first. */
    public abstract val instructions: String

    /** Option texts in label-index order, exactly as `render_options` renders them. */
    public abstract fun renderOptions(): List<String>

    /** Number of options, hence of markers the sequence must carry. */
    public val optionCount: Int get() = renderOptions().size

    /**
     * `choice`: one option per criterion key, in order. A criterion that is `null`, absent or the
     * empty string renders as the bare key; anything else renders as `key: criterion`, where a
     * structured criterion becomes single-line JSON ([PythonJson]). `0` and `false` are
     * legitimate criteria and render.
     */
    public class Choice(
        override val instructions: String,
        public val criteria: List<Pair<String, JsonElement?>>,
    ) : LayaQuestion() {
        public constructor(instructions: String, criteria: Map<String, String?>) :
            this(instructions, criteria.entries.map { (k, v) -> k to v?.let(::JsonPrimitive) })

        init {
            require(criteria.isNotEmpty()) { "a choice question needs at least one criterion" }
            require(criteria.map { it.first }.toSet().size == criteria.size) { "choice criteria keys must be distinct" }
        }

        override val type: LayaQuestionType get() = LayaQuestionType.CHOICE

        /** The caller's option identifiers, in option order. */
        public val keys: List<String> get() = criteria.map { it.first }

        override fun renderOptions(): List<String> = criteria.map { (key, value) ->
            if (hasNoDescription(value)) key else "$key: ${renderCriterion(value!!)}"
        }
    }

    /** `score`: one option per level, rendered `level i: text`. */
    public class Score(
        override val instructions: String,
        public val levels: List<JsonElement>,
    ) : LayaQuestion() {
        public constructor(instructions: String, levels: List<String>, @Suppress("UNUSED_PARAMETER") asText: Unit = Unit) :
            this(instructions, levels.map { JsonPrimitive(it) as JsonElement })

        init {
            require(levels.isNotEmpty()) { "a score question needs at least one level" }
        }

        override val type: LayaQuestionType get() = LayaQuestionType.SCORE

        override fun renderOptions(): List<String> = levels.mapIndexed { i, level -> "level $i: ${renderCriterion(level)}" }
    }

    /**
     * `noul`: always two options, `false` then `true`, so the second probability is the answer.
     * [labels] replaces the words `false`/`true` in front of the colon; both must be distinct and
     * non-empty after trimming.
     */
    public class Noul(
        override val instructions: String,
        public val falseCriterion: JsonElement? = null,
        public val trueCriterion: JsonElement? = null,
        public val labels: Pair<String, String>? = null,
    ) : LayaQuestion() {
        public constructor(instructions: String, falseCriterion: String?, trueCriterion: String?) :
            this(instructions, falseCriterion?.let(::JsonPrimitive), trueCriterion?.let(::JsonPrimitive), null)

        private val resolvedLabels: Pair<String, String> = run {
            val (f, t) = labels ?: ("false" to "true")
            val falseLabel = f.trim()
            val trueLabel = t.trim()
            require(falseLabel.isNotEmpty() && trueLabel.isNotEmpty() && falseLabel != trueLabel) {
                "noul labels must map exactly 'false' and 'true' to distinct non-empty strings"
            }
            falseLabel to trueLabel
        }

        override val type: LayaQuestionType get() = LayaQuestionType.NOUL

        override fun renderOptions(): List<String> = listOf(
            resolvedLabels.first + ": " + (if (hasNoDescription(falseCriterion)) "no, the statement does not hold" else renderCriterion(falseCriterion!!)),
            resolvedLabels.second + ": " + (if (hasNoDescription(trueCriterion)) "yes, the statement holds" else renderCriterion(trueCriterion!!)),
        )
    }

    public companion object {
        /** `render_criterion`: strings pass through, anything structured becomes single-line JSON. */
        public fun renderCriterion(value: JsonElement): String =
            if (value is JsonPrimitive && value.isString) value.content else PythonJson.dumps(value)

        /** The reference treats only `None` and `""` as "no description" (`0` and `False` are not). */
        public fun hasNoDescription(value: JsonElement?): Boolean =
            value == null || value is JsonNull || (value is JsonPrimitive && value.isString && value.content.isEmpty())
    }
}

/**
 * The state a question is asked about. Text is used as given; a structured state is serialised
 * with [PythonJson] (`serialize_state`). A list state is a conversation, newest turn last, and
 * truncates from the left so the current turn survives (REQ-PRE-002).
 */
public sealed class LayaState {
    public abstract fun serialize(): String

    /** Whether an oversized state keeps its end rather than its beginning. */
    public open val truncatesLeft: Boolean get() = false

    public data class Text(public val text: String) : LayaState() {
        override fun serialize(): String = text
    }

    public data class Structured(public val value: JsonElement) : LayaState() {
        override fun serialize(): String = PythonJson.dumps(value)
        override val truncatesLeft: Boolean get() = value is JsonArray
    }

    public companion object {
        public fun of(text: String): LayaState = Text(text)
        public fun of(value: JsonElement): LayaState = if (value is JsonPrimitive && value.isString) Text(value.content) else Structured(value)
    }
}
