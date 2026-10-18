package sk.ainet.decisions.runtime

import sk.ainet.decisions.AbstentionReason
import sk.ainet.decisions.QuestionOutcome
import sk.ainet.decisions.boolean
import sk.ainet.decisions.choice
import sk.ainet.decisions.dsl.decisionSchema
import sk.ainet.decisions.isComplete
import sk.ainet.decisions.score
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

class SchemaSessionTest {
    private val triage = decisionSchema("triage") {
        model("fake")
        choice("dept", "Which department?") { option("billing", "Invoices"); option("tech", "Faults"); option("other", "Else") }
        score("urgency", "How urgent?", "low", "mid", "high")
        probability("dup", "A duplicate invoice is reported.") { whenFalse = "not mentioned"; whenTrue = "mentioned" }
        policy(version = "1") {
            choice { minConfidence = 0.3; minMargin = 0.5 }
            score { minTopProbability(0.6) }
            boolean { abstainBetween(0.2, 0.8) }
        }
    }

    @Test
    fun a_schema_session_decides_one_state_after_another_under_the_schema_policy() {
        val adapter = FakeAdapter(scripted = mapOf(
            "dept" to FakeAdapter.choice("billing", "billing" to 0.75, "tech" to 0.25, "other" to 0.0),
            "urgency" to FakeAdapter.score(0.1, 0.3, 0.6),
            "dup" to FakeAdapter.bool(0.5),
        ))
        DecisionRuntime(listOf(adapter)).openSession(triage).use { session ->
            val first = session.decide(triage, "t-1", "first ticket")
            val second = session.decide(triage, "t-2", "second ticket")
            assertEquals("t-1", first.requestId)
            assertEquals("t-2", second.requestId)
            assertEquals(2, adapter.calls)
            assertEquals("billing", first.choice("dept"))
            assertEquals(2, first.score("urgency"))
            assertNull(first.boolean("dup"))
            assertEquals(AbstentionReason.BETWEEN_THRESHOLDS, (first.question("dup") as QuestionOutcome.Abstained).reason)
            assertEquals("triage", first.provenance.policyId)
            assertEquals("1", first.provenance.policyVersion)
            assertTrue(!first.isComplete)
        }
    }

    @Test
    fun a_schema_without_a_model_cannot_open_a_session_and_an_explicit_policy_overrides_the_schema() {
        val anonymous = decisionSchema("s") { probability("dup", "A duplicate invoice is reported.") }
        val runtime = DecisionRuntime(listOf(FakeAdapter(scripted = mapOf("dup" to FakeAdapter.bool(0.9)))))
        assertFailsWith<IllegalArgumentException> { runtime.openSession(anonymous) }
        runtime.openSession(triage).use { session ->
            val noPolicy = session.decide(triage, "t", "state", acceptance = null)
            assertEquals(AbstentionReason.NO_POLICY, (noPolicy.question("dup") as QuestionOutcome.Abstained).reason)
            assertEquals(true, session.decide(triage, "t", "state").boolean("dup"))
        }
    }
}
