package sk.ainet.decisions

// Typed readback of a DecisionResult. The rule everywhere: null means "no accepted decision"
// (abstained or failed), an unknown question id throws, a question of another primitive throws.

/** The outcome of [questionId], or null when the result has no such question. */
public fun DecisionResult.outcomeOrNull(questionId: String): QuestionOutcome? =
    outcomes.firstOrNull { it.questionId == questionId }

/** The adapter's candidate: the accepted decision's prediction, an abstention's diagnostic candidate, null for a failure. */
public val QuestionOutcome.candidate: Prediction?
    get() = when (this) {
        is QuestionOutcome.Accepted -> decision.prediction
        is QuestionOutcome.Abstained -> candidate
        is QuestionOutcome.Failed -> null
    }

/** The accepted decision of [questionId], or null when the question abstained or failed. */
public fun DecisionResult.accepted(questionId: String): Decision? =
    (question(questionId) as? QuestionOutcome.Accepted)?.decision

/** The accepted option id of a choice question, or null. */
public fun DecisionResult.choice(questionId: String): String? = when (val d = accepted(questionId)) {
    null -> null
    is Decision.Choice -> d.optionId
    else -> throw IllegalArgumentException("question '$questionId' is not a choice question")
}

/** The accepted level (0-based, in the question's level order) of a score question, or null. */
public fun DecisionResult.score(questionId: String): Int? = when (val d = accepted(questionId)) {
    null -> null
    is Decision.Score -> d.level
    else -> throw IllegalArgumentException("question '$questionId' is not a score question")
}

/** The accepted value of a boolean probability question, or null. */
public fun DecisionResult.boolean(questionId: String): Boolean? = when (val d = accepted(questionId)) {
    null -> null
    is Decision.Boolean -> d.value
    else -> throw IllegalArgumentException("question '$questionId' is not a boolean probability question")
}

/** P(true) of a boolean probability question from the accepted decision or the abstained candidate; null after a failure. */
public fun DecisionResult.probabilityTrue(questionId: String): Double? = when (val c = question(questionId).candidate) {
    null -> null
    is Prediction.BooleanProbability -> c.probabilityTrue
    else -> throw IllegalArgumentException("question '$questionId' is not a boolean probability question")
}

/** True when every question was accepted. */
public val DecisionResult.isComplete: Boolean
    get() = outcomes.all { it is QuestionOutcome.Accepted }

/** `kind=bugfix risk=1 standup=true adr=?`: one token per question, `?` for an abstention, `!` for a failure. */
public fun DecisionResult.summary(): String = outcomes.joinToString(" ") { outcome ->
    val value = when (outcome) {
        is QuestionOutcome.Accepted -> when (val d = outcome.decision) {
            is Decision.Choice -> d.optionId
            is Decision.Score -> d.level.toString()
            is Decision.Boolean -> d.value.toString()
        }
        is QuestionOutcome.Abstained -> "?"
        is QuestionOutcome.Failed -> "!"
    }
    "${outcome.questionId}=$value"
}
