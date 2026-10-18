package sk.ainet.decisions.runtime

import sk.ainet.decisions.AbstentionReason
import sk.ainet.decisions.AcceptancePolicy
import sk.ainet.decisions.BooleanAcceptance
import sk.ainet.decisions.BooleanCriteria
import sk.ainet.decisions.BooleanProbabilityQuestion
import sk.ainet.decisions.ChoiceAcceptance
import sk.ainet.decisions.ChoiceOption
import sk.ainet.decisions.ChoiceQuestion
import sk.ainet.decisions.Confidence
import sk.ainet.decisions.ConfidenceDefinition
import sk.ainet.decisions.Decision
import sk.ainet.decisions.DecisionError
import sk.ainet.decisions.DecisionRequest
import sk.ainet.decisions.DecisionRequestException
import sk.ainet.decisions.DecisionState
import sk.ainet.decisions.InferencePolicy
import sk.ainet.decisions.LanguageSupport
import sk.ainet.decisions.ModelSelector
import sk.ainet.decisions.OptionProbability
import sk.ainet.decisions.Prediction
import sk.ainet.decisions.QuestionOutcome
import sk.ainet.decisions.ScoreAcceptance
import sk.ainet.decisions.ScoreLevel
import sk.ainet.decisions.ScoreQuestion
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/** REQ-API / REQ-POL acceptance on a deterministic adapter: thresholds, reasons, order, failures. */
class DecisionRuntimeConformanceTest {
    private val dept = ChoiceQuestion("dept", "Which department?", listOf(ChoiceOption("billing", "Invoices"), ChoiceOption("tech", "Faults"), ChoiceOption("other", "Else")))
    private val urgency = ScoreQuestion("urgency", "How urgent?", listOf(ScoreLevel("low"), ScoreLevel("mid"), ScoreLevel("high")))
    private val dup = BooleanProbabilityQuestion("dup", "A duplicate invoice is reported.", BooleanCriteria("not mentioned", "mentioned"))
    // Thresholds chosen so that the equality cases below are exactly representable (0.75 - 0.25 == 0.5).
    private val policy = AcceptancePolicy("triage", "1", choice = ChoiceAcceptance(minConfidence = 0.3, minMargin = 0.5), score = ScoreAcceptance.MinTopProbability(0.6), boolean = BooleanAcceptance(0.2, 0.8))

    private fun request(vararg qs: sk.ainet.decisions.DecisionQuestion, model: String = "fake", inference: InferencePolicy = InferencePolicy.DEFAULT) =
        DecisionRequest("r1", DecisionState.Text("state"), qs.toList(), ModelSelector.Installed(model), inference)

    private fun session(adapter: FakeAdapter) = DecisionRuntime(listOf(adapter)).openSession(ModelSelector.Installed(adapter.capabilities.modelId))

    @Test
    fun thresholds_are_inclusive_at_exact_equality() {
        val adapter = FakeAdapter(scripted = mapOf(
            "dept" to FakeAdapter.choice("billing", "billing" to 0.75, "tech" to 0.25, "other" to 0.0).copy(confidence = Confidence(0.3, ConfidenceDefinition.NORMALIZED_ENTROPY)),
            "urgency" to FakeAdapter.score(0.2, 0.6, 0.2),
            "dup" to FakeAdapter.bool(0.8),
        ))
        val result = session(adapter).decide(request(dept, urgency, dup), policy)
        assertEquals(listOf("dept", "urgency", "dup"), result.outcomes.map { it.questionId })
        val d = assertIs<QuestionOutcome.Accepted>(result.question("dept")).decision as Decision.Choice
        assertEquals("billing", d.optionId)
        assertEquals(1, (assertIs<QuestionOutcome.Accepted>(result.question("urgency")).decision as Decision.Score).level)
        assertEquals(true, (assertIs<QuestionOutcome.Accepted>(result.question("dup")).decision as Decision.Boolean).value)
        assertEquals("triage", result.provenance.policyId)
        assertEquals(3, result.provenance.traces.size)
    }

    @Test
    fun every_abstention_reason_is_reachable_and_keeps_the_candidate_as_diagnostic() {
        val adapter = FakeAdapter(scripted = mapOf(
            "dept" to FakeAdapter.choice("billing", "billing" to 0.5, "tech" to 0.4, "other" to 0.1).copy(confidence = Confidence(0.9, ConfidenceDefinition.NORMALIZED_ENTROPY)), // margin 0.1 < 0.5
            "urgency" to FakeAdapter.score(0.5, 0.3, 0.2),                                              // top 0.5 < 0.6
            "dup" to FakeAdapter.bool(0.5),                                                             // between
        ))
        val result = session(adapter).decide(request(dept, urgency, dup), policy)
        val deptOutcome = assertIs<QuestionOutcome.Abstained>(result.question("dept"))
        assertEquals(AbstentionReason.BELOW_MARGIN, deptOutcome.reason)
        assertNotNull(deptOutcome.candidate)
        assertEquals(AbstentionReason.BELOW_CONFIDENCE, assertIs<QuestionOutcome.Abstained>(result.question("urgency")).reason)
        assertEquals(AbstentionReason.BETWEEN_THRESHOLDS, assertIs<QuestionOutcome.Abstained>(result.question("dup")).reason)

        val lowConf = FakeAdapter(scripted = mapOf("dept" to FakeAdapter.choice("billing", "billing" to 0.4, "tech" to 0.35, "other" to 0.25).copy(confidence = Confidence(0.1, ConfidenceDefinition.NORMALIZED_ENTROPY))))
        assertEquals(AbstentionReason.BELOW_CONFIDENCE, assertIs<QuestionOutcome.Abstained>(session(lowConf).decide(request(dept), AcceptancePolicy("p", "1", choice = ChoiceAcceptance(minConfidence = 0.3))).question("dept")).reason)
    }

    @Test
    fun no_policy_never_accepts() {
        val adapter = FakeAdapter(scripted = mapOf("dup" to FakeAdapter.bool(0.99)))
        val none = session(adapter).decide(request(dup))
        assertEquals(AbstentionReason.NO_POLICY, assertIs<QuestionOutcome.Abstained>(none.question("dup")).reason)
        val partial = session(adapter).decide(request(dup, dept), AcceptancePolicy("p", "1", choice = ChoiceAcceptance()))
        assertEquals(AbstentionReason.NO_POLICY, assertIs<QuestionOutcome.Abstained>(partial.question("dup")).reason)
        assertIs<QuestionOutcome.Accepted>(partial.question("dept"), "an empty choice rule accepts every candidate; the policy is the application's")
        assertEquals(null, none.provenance.policyId)
    }

    @Test
    fun boolean_orientation_and_false_acceptance() {
        val adapter = FakeAdapter(scripted = mapOf("dup" to FakeAdapter.bool(0.2)))
        val accepted = assertIs<QuestionOutcome.Accepted>(session(adapter).decide(request(dup), policy).question("dup"))
        val decision = accepted.decision as Decision.Boolean
        assertEquals(false, decision.value)
        assertEquals(0.8, decision.prediction.probabilityFalse, 1e-12)
    }

    @Test
    fun question_failures_do_not_fail_the_request_and_non_finite_candidates_never_accept() {
        val adapter = FakeAdapter(
            scripted = mapOf("dup" to Prediction.BooleanProbability(Double.NaN, Confidence(1.0, ConfidenceDefinition.TWO_WAY_MAX))),
            failing = mapOf("urgency" to DecisionError.ContextOverflow(90, 64)),
        )
        val result = session(adapter).decide(request(dept, urgency, dup), policy)
        assertIs<QuestionOutcome.Abstained>(result.question("dept"))
        assertIs<DecisionError.ContextOverflow>(assertIs<QuestionOutcome.Failed>(result.question("urgency")).error)
        assertIs<DecisionError.NumericalFailure>(assertIs<QuestionOutcome.Failed>(result.question("dup")).error)
        assertEquals(2, result.provenance.traces.size, "every packed question carries a trace, the failed one did not pack")
    }

    @Test
    fun request_level_problems_are_typed_and_evaluated_before_inference() {
        val adapter = FakeAdapter(languages = LanguageSupport.Only(setOf("de", "en")), maxOptions = 2)
        val s = session(adapter)
        assertIs<DecisionError.UnsupportedCapability>(assertFailsWith<DecisionRequestException> { s.decide(request(dup, model = "other")) }.error)
        assertIs<DecisionError.UnsupportedCapability>(assertFailsWith<DecisionRequestException> { s.decide(request(dup, inference = InferencePolicy(language = "fr-FR"))) }.error)
        assertIs<DecisionError.InvalidSchema>(assertFailsWith<DecisionRequestException> { s.decide(request(dept)) }.error, "3 options exceed the limit of 2")
        assertIs<DecisionError.ResourceExhausted>(assertFailsWith<DecisionRequestException> { s.decide(request(dup, dup.copy(id = "b"), dup.copy(id = "c"), dup.copy(id = "d"), dup.copy(id = "e"))) }.error)
        assertEquals(0, adapter.calls, "nothing reached the adapter")
        s.decide(request(dup, inference = InferencePolicy(language = "de-CH")))
        assertEquals(1, adapter.calls)
        assertFailsWith<DecisionRequestException> { DecisionRuntime(listOf(adapter)).openSession(ModelSelector.Installed("missing")) }
    }

    @Test
    fun closing_releases_and_refuses_further_work() {
        val adapter = FakeAdapter()
        val runtime = DecisionRuntime(listOf(adapter))
        val s = runtime.openSession(ModelSelector.Installed("fake"))
        s.close()
        assertIs<DecisionError.ResourceExhausted>(assertFailsWith<DecisionRequestException> { s.decide(request(dup)) }.error)
        runtime.close()
        assertTrue(adapter.closed)
    }

    @Test
    fun distributions_from_the_fake_sum_to_one_and_choice_ids_belong_to_the_question() {
        val result = session(FakeAdapter()).decide(request(dept, urgency), AcceptancePolicy("p", "1", choice = ChoiceAcceptance(), score = ScoreAcceptance.MaxNormalizedEntropy(1.0)))
        val c = (assertIs<QuestionOutcome.Accepted>(result.question("dept")).decision as Decision.Choice)
        assertTrue(c.optionId in dept.options.map { it.id })
        assertEquals(1.0, c.prediction.probabilities.sumOf { it.probability }, 1e-5)
        val sc = (assertIs<QuestionOutcome.Accepted>(result.question("urgency")).decision as Decision.Score).prediction
        assertEquals(1.0, sc.distribution.sum(), 1e-5)
        assertTrue(sc.expected in 0.0..2.0)
        assertEquals(listOf(OptionProbability("billing", 1.0 / 3), OptionProbability("tech", 1.0 / 3), OptionProbability("other", 1.0 / 3)), c.prediction.probabilities)
    }
}
