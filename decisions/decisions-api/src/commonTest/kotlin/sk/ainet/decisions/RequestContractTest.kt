package sk.ainet.decisions

import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class RequestContractTest {
    private val choice = ChoiceQuestion("dept", "Which department?", listOf(ChoiceOption("billing", "Invoices and refunds"), ChoiceOption("tech", "Software faults")))

    @Test
    fun invalid_option_sets_fail_before_anything_else() {
        assertFailsWith<IllegalArgumentException> { ChoiceQuestion("q", "i", listOf(ChoiceOption("a", "x"))) }
        assertFailsWith<IllegalArgumentException> { ChoiceQuestion("q", "i", listOf(ChoiceOption("a", "x"), ChoiceOption("a", "y"))) }
        assertFailsWith<IllegalArgumentException> { ChoiceOption("a", " ") }
        assertFailsWith<IllegalArgumentException> { ScoreQuestion("q", "i", listOf(ScoreLevel("only one"))) }
        assertFailsWith<IllegalArgumentException> { BooleanProbabilityQuestion("q", "", BooleanCriteria("no", "yes")) }
        assertFailsWith<IllegalArgumentException> { BooleanCriteria("", "yes") }
    }

    @Test
    fun request_requires_unique_question_ids_and_a_model() {
        assertFailsWith<IllegalArgumentException> { DecisionRequest("r", DecisionState.Text("s"), listOf(choice, choice.copy(instructions = "again")), ModelSelector.Installed("m")) }
        assertFailsWith<IllegalArgumentException> { DecisionRequest("r", DecisionState.Text("s"), emptyList(), ModelSelector.Installed("m")) }
        assertFailsWith<IllegalArgumentException> { ModelSelector.Installed(" ") }
        assertFailsWith<IllegalArgumentException> { DecisionRequest.builder("r").state("s").question(choice).build() }
    }

    @Test
    fun builder_preserves_question_order_and_defaults() {
        val second = BooleanProbabilityQuestion("dup", "The customer reports a duplicate invoice.", BooleanCriteria("no duplicate", "a duplicate is mentioned"))
        val r = DecisionRequest.builder("r1").state(buildJsonObject { put("text", "x") }).question(choice).question(second).model("laya-multilingual").build()
        assertEquals(listOf("dept", "dup"), r.questions.map { it.id })
        assertEquals(TruncationPolicy.REJECT, r.inference.truncation)
        assertTrue(r.state is DecisionState.Structured)
        assertEquals(Primitive.BOOLEAN_PROBABILITY, second.primitive)
        assertEquals("noul", Primitive.BOOLEAN_PROBABILITY.wire)
    }

    @Test
    fun invalid_policy_shapes_are_typed_errors() {
        assertFailsWith<InvalidPolicyException> { BooleanAcceptance(0.6, 0.6) }
        assertFailsWith<InvalidPolicyException> { BooleanAcceptance(0.7, 0.3) }
        assertFailsWith<InvalidPolicyException> { BooleanAcceptance(-0.1, 0.5) }
        assertFailsWith<InvalidPolicyException> { ChoiceAcceptance(minConfidence = 1.5) }
        assertFailsWith<InvalidPolicyException> { ScoreAcceptance.MinTopProbability(2.0) }
        assertFailsWith<IllegalArgumentException> { AcceptancePolicy("", "1") }
        val ok = AcceptancePolicy("triage", "2026-10", choice = ChoiceAcceptance(minConfidence = 0.5, minMargin = 0.2), boolean = BooleanAcceptance(0.2, 0.8))
        assertEquals(null, ok.score)
    }

    @Test
    fun predictions_expose_derived_values() {
        val p = Prediction.Choice("a", listOf(OptionProbability("a", 0.7), OptionProbability("b", 0.2), OptionProbability("c", 0.1)), Confidence(0.4, ConfidenceDefinition.NORMALIZED_ENTROPY))
        assertEquals(0.5, p.margin, 1e-12)
        assertEquals(0.2, p.probabilityOf("b"))
        val b = Prediction.BooleanProbability(0.3, Confidence(0.7, ConfidenceDefinition.TWO_WAY_MAX))
        assertEquals(0.7, b.probabilityFalse, 1e-12)
        assertFailsWith<IllegalArgumentException> { Confidence(1.2, ConfidenceDefinition.TWO_WAY_MAX) }
    }
}
