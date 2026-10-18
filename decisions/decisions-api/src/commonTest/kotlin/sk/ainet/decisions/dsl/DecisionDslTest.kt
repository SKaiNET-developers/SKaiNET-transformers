package sk.ainet.decisions.dsl

import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import sk.ainet.decisions.AcceptancePolicy
import sk.ainet.decisions.BooleanAcceptance
import sk.ainet.decisions.BooleanCriteria
import sk.ainet.decisions.BooleanProbabilityQuestion
import sk.ainet.decisions.CalibrationStatus
import sk.ainet.decisions.ChoiceAcceptance
import sk.ainet.decisions.ChoiceOption
import sk.ainet.decisions.ChoiceQuestion
import sk.ainet.decisions.Confidence
import sk.ainet.decisions.ConfidenceDefinition
import sk.ainet.decisions.Decision
import sk.ainet.decisions.DecisionError
import sk.ainet.decisions.DecisionResult
import sk.ainet.decisions.DecisionState
import sk.ainet.decisions.InvalidPolicyException
import sk.ainet.decisions.ModelSelector
import sk.ainet.decisions.OptionProbability
import sk.ainet.decisions.Prediction
import sk.ainet.decisions.Provenance
import sk.ainet.decisions.QuestionOutcome
import sk.ainet.decisions.ScoreAcceptance
import sk.ainet.decisions.ScoreLevel
import sk.ainet.decisions.ScoreQuestion
import sk.ainet.decisions.TruncationPolicy
import sk.ainet.decisions.accepted
import sk.ainet.decisions.boolean
import sk.ainet.decisions.candidate
import sk.ainet.decisions.choice
import sk.ainet.decisions.isComplete
import sk.ainet.decisions.outcomeOrNull
import sk.ainet.decisions.probabilityTrue
import sk.ainet.decisions.score
import sk.ainet.decisions.summary
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class DecisionDslTest {

    @Test
    fun request_dsl_builds_the_same_request_as_the_constructors() {
        val expected = sk.ainet.decisions.DecisionRequest(
            "ticket-4711",
            DecisionState.Text("my invoice shows the old address"),
            listOf(
                ChoiceQuestion("department", "Which department handles this request?", listOf(ChoiceOption("billing", "billing and invoices"), ChoiceOption("account", "account data and profile"), ChoiceOption("other", "other"))),
                ScoreQuestion("urgency", "How urgent is this request?", listOf(ScoreLevel("can wait"), ScoreLevel("this week"), ScoreLevel("today"), ScoreLevel("now"))),
                BooleanProbabilityQuestion("escalate", "Should a human look at this request?", BooleanCriteria("no", "yes")),
                BooleanProbabilityQuestion("duplicate", "The customer reports a duplicate invoice.", BooleanCriteria("not mentioned", "a duplicate is mentioned")),
            ),
            ModelSelector.Installed("laya-multilingual"),
        )
        val actual = decisionRequest("ticket-4711") {
            model("laya-multilingual")
            state("my invoice shows the old address")
            choice("department", "Which department handles this request?") {
                option("billing", "billing and invoices")
                option("account", "account data and profile")
                option("other")
            }
            score("urgency", "How urgent is this request?", "can wait", "this week", "today", "now")
            probability("escalate", "Should a human look at this request?")
            probability("duplicate", "The customer reports a duplicate invoice.") {
                whenFalse = "not mentioned"
                whenTrue = "a duplicate is mentioned"
            }
        }
        assertEquals(expected, actual)
    }

    @Test
    fun request_dsl_requires_state_model_and_a_question_and_carries_the_inference_policy() {
        assertFailsWith<IllegalArgumentException> { decisionRequest("r") { model("m"); choice("q", "i", "a", "b") } }
        assertFailsWith<IllegalArgumentException> { decisionRequest("r") { state("s"); choice("q", "i", "a", "b") } }
        assertFailsWith<IllegalArgumentException> { decisionRequest("r") { state("s"); model("m") } }
        assertFailsWith<IllegalArgumentException> { decisionRequest("r") { state("s"); model("m"); choice("q", "i", "a") } }
        val structured = decisionRequest("r") {
            state(buildJsonObject { put("text", "x") })
            model("m")
            truncateState()
            language("de")
            score("s", "how much?") { level("none"); level("some"); level("all") }
        }
        assertTrue(structured.state is DecisionState.Structured)
        assertEquals(TruncationPolicy.TRUNCATE_STATE, structured.inference.truncation)
        assertEquals("de", structured.inference.language)
        assertEquals(3, structured.questions.single().optionCount)
    }

    @Test
    fun policy_dsl_builds_the_same_policy_as_the_constructors_and_rejects_half_policies() {
        val expected = AcceptancePolicy("triage", "1", ChoiceAcceptance(0.2, 0.1), ScoreAcceptance.MinTopProbability(0.35), BooleanAcceptance(0.4, 0.6))
        val actual = acceptancePolicy("triage", "1") {
            choice { minConfidence = 0.2; minMargin = 0.1 }
            score { minTopProbability(0.35) }
            boolean { abstainBetween(0.4, 0.6) }
        }
        assertEquals(expected, actual)
        val partial = acceptancePolicy("p", "1") { choice { } }
        assertEquals(ChoiceAcceptance(), partial.choice)
        assertNull(partial.score)
        assertNull(partial.boolean)
        assertFailsWith<InvalidPolicyException> { acceptancePolicy("p", "1") { score { } } }
        assertFailsWith<InvalidPolicyException> { acceptancePolicy("p", "1") { score { minTopProbability(0.3); maxNormalizedEntropy(0.5) } } }
        assertFailsWith<InvalidPolicyException> { acceptancePolicy("p", "1") { boolean { lower = 0.4 } } }
        assertFailsWith<InvalidPolicyException> { acceptancePolicy("p", "1") { boolean { abstainBetween(0.6, 0.4) } } }
    }

    @Test
    fun schema_fixes_questions_policy_and_model_and_stamps_out_requests() {
        val triage = decisionSchema("commit-triage") {
            model("laya-multilingual")
            truncateState()
            choice("kind", "What kind of change is this commit?", "feature", "bugfix", "refactor")
            probability("standup", "Is this commit worth mentioning in the daily stand-up?")
            policy(version = "1") {
                choice { minConfidence = 0.2 }
                boolean { abstainBetween(0.4, 0.6) }
            }
        }
        assertEquals("commit-triage", triage.policy?.id)
        assertEquals("1", triage.policy?.version)
        assertEquals(listOf("kind", "standup"), triage.questions.map { it.id })
        assertTrue(triage.question("kind") is ChoiceQuestion)
        assertNull(triage.question("risk"))

        val a = triage.request("commit-1", "fix: off-by-one in the date filter")
        val b = triage.request("commit-2", buildJsonObject { put("message", "docs: typo") })
        assertEquals(triage.questions, a.questions)
        assertEquals(ModelSelector.Installed("laya-multilingual"), a.model)
        assertEquals(TruncationPolicy.TRUNCATE_STATE, a.inference.truncation)
        assertTrue(b.state is DecisionState.Structured)
        assertEquals(ModelSelector.Installed("other"), triage.request("commit-3", "x", ModelSelector.Installed("other")).model)

        val noModel = decisionSchema("s") { probability("p", "statement") }
        assertNull(noModel.model)
        assertNull(noModel.policy)
        assertFailsWith<IllegalArgumentException> { noModel.request("r", "state") }
        assertEquals("m", (noModel.request("r", "state", ModelSelector.Installed("m")).model as ModelSelector.Installed).modelId)

        assertFailsWith<IllegalArgumentException> { decisionSchema("s") { } }
        assertFailsWith<IllegalArgumentException> { decisionSchema("s") { probability("p", "a"); probability("p", "b") } }
        assertFailsWith<IllegalArgumentException> { decisionSchema(" ") { probability("p", "a") } }
    }

    @Test
    fun typed_readback_returns_null_for_abstentions_and_failures_and_refuses_the_wrong_primitive() {
        val choicePrediction = Prediction.Choice("bugfix", listOf(OptionProbability("feature", 0.1), OptionProbability("bugfix", 0.8), OptionProbability("docs", 0.1)), Confidence(0.6, ConfidenceDefinition.NORMALIZED_ENTROPY))
        val scorePrediction = Prediction.Score(1.2, 1, listOf(0.2, 0.5, 0.3), Confidence(0.3, ConfidenceDefinition.NORMALIZED_ENTROPY))
        val unsure = Prediction.BooleanProbability(0.55, Confidence(0.55, ConfidenceDefinition.TWO_WAY_MAX))
        val result = DecisionResult(
            "r",
            listOf(
                QuestionOutcome.Accepted("kind", Decision.Choice("bugfix", choicePrediction)),
                QuestionOutcome.Accepted("risk", Decision.Score(1, scorePrediction)),
                QuestionOutcome.Abstained("standup", sk.ainet.decisions.AbstentionReason.BETWEEN_THRESHOLDS, unsure),
                QuestionOutcome.Failed("adr", DecisionError.NumericalFailure("nan")),
                QuestionOutcome.Accepted("breaking", Decision.Boolean(false, Prediction.BooleanProbability(0.1, Confidence(0.9, ConfidenceDefinition.TWO_WAY_MAX)))),
            ),
            Provenance("m", "fake", "fp64", "fake/1", CalibrationStatus.UNVERIFIED, "none", "p", "1", emptyList(), 1),
        )
        assertEquals("bugfix", result.choice("kind"))
        assertEquals(1, result.score("risk"))
        assertNull(result.boolean("standup"))
        assertEquals(0.55, result.probabilityTrue("standup"))
        assertNull(result.boolean("adr"))
        assertNull(result.probabilityTrue("adr"))
        assertEquals(false, result.boolean("breaking"))
        assertEquals(0.1, result.probabilityTrue("breaking"))
        assertEquals(choicePrediction, result.question("kind").candidate)
        assertEquals(unsure, result.question("standup").candidate)
        assertNull(result.question("adr").candidate)
        assertTrue(result.accepted("kind") is Decision.Choice)
        assertNull(result.accepted("standup"))
        assertNull(result.outcomeOrNull("missing"))
        assertFalse(result.isComplete)
        assertEquals("kind=bugfix risk=1 standup=? adr=! breaking=false", result.summary())
        assertFailsWith<IllegalArgumentException> { result.score("kind") }
        assertFailsWith<IllegalArgumentException> { result.choice("risk") }
        assertFailsWith<IllegalArgumentException> { result.boolean("kind") }
        assertFailsWith<IllegalArgumentException> { result.probabilityTrue("kind") }
        assertFailsWith<NoSuchElementException> { result.choice("missing") }
    }
}
