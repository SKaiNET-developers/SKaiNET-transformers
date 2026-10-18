package sk.ainet.models.laya

import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.double
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.decisions.AbstentionReason
import sk.ainet.decisions.AcceptancePolicy
import sk.ainet.decisions.BooleanAcceptance
import sk.ainet.decisions.BooleanCriteria
import sk.ainet.decisions.BooleanProbabilityQuestion
import sk.ainet.decisions.CalibrationStatus
import sk.ainet.decisions.ChoiceAcceptance
import sk.ainet.decisions.ChoiceOption
import sk.ainet.decisions.ChoiceQuestion
import sk.ainet.decisions.Decision
import sk.ainet.decisions.DecisionError
import sk.ainet.decisions.DecisionRequest
import sk.ainet.decisions.DecisionState
import sk.ainet.decisions.InferencePolicy
import sk.ainet.decisions.ModelSelector
import sk.ainet.decisions.QuestionOutcome
import sk.ainet.decisions.ScoreAcceptance
import sk.ainet.decisions.ScoreLevel
import sk.ainet.decisions.ScoreQuestion
import sk.ainet.decisions.TruncationPolicy
import sk.ainet.decisions.runtime.DecisionRuntime
import java.nio.file.Files
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

/** The typed contract end to end on the real checkpoint: the `billing-mixed` reference case and the overflow path. */
class LayaDecisionAdapterTest {

    @Test
    fun `billing-mixed through the typed API matches the reference and the policy decides`() {
        val dir = LayaFixtures.checkpointDir() ?: return
        if (!Files.exists(dir.resolve("model.safetensors"))) { println("[skip] no weights"); return }
        val agent = LayaModelLoader.load(dir, DirectCpuExecutionContext())
        val adapter = LayaDecisionAdapter(agent, modelId = "laya-multilingual")
        assertEquals(CalibrationStatus.UNVERIFIED, adapter.capabilities.calibration, "multilingual ships no fitted temperatures")
        assertEquals(1024, adapter.capabilities.maxContextTokens)

        val fixture = LayaFixtures.resource("predict.json")["single"]!!.jsonObject["billing-mixed"]!!.jsonObject
        val request = DecisionRequest.builder("billing-mixed")
            .state(fixture["state"]!!.jsonPrimitive.content)
            .question(ChoiceQuestion("intent", "What does the customer want?", listOf(
                ChoiceOption("refund", "money back for a duplicate charge"), ChoiceOption("replace", "a replacement unit"), ChoiceOption("info", "an explanation only"))))
            .question(ScoreQuestion("urgency", "How urgent is this?", listOf("not urgent", "somewhat", "urgent", "critical").map(::ScoreLevel)))
            .question(BooleanProbabilityQuestion("duplicate", "The customer was charged more than once.", BooleanCriteria("no, the statement does not hold", "yes, the statement holds")))
            .model("laya-multilingual")
            .build()
        val policy = AcceptancePolicy("test-triage", "1", choice = ChoiceAcceptance(minConfidence = 0.5, minMargin = 0.3), score = ScoreAcceptance.MinTopProbability(0.7), boolean = BooleanAcceptance(0.1, 0.9))

        DecisionRuntime(listOf(adapter)).use { runtime ->
            val result = runtime.openSession(ModelSelector.Installed("laya-multilingual")).decide(request, policy)
            val expected = fixture["answers"]!!.jsonObject
            val intent = assertIs<QuestionOutcome.Accepted>(result.question("intent")).decision as Decision.Choice
            assertEquals(expected["intent"]!!.jsonObject["choice"]!!.jsonPrimitive.content, intent.optionId)
            for ((key, p) in expected["intent"]!!.jsonObject["probabilities"]!!.jsonObject) assertEquals(p.jsonPrimitive.double, intent.prediction.probabilityOf(key), 1.5e-4, "p[$key]")
            // urgency: top level 0.6075 < 0.7 → abstains, candidate still inspectable
            val urgency = assertIs<QuestionOutcome.Abstained>(result.question("urgency"))
            assertEquals(AbstentionReason.BELOW_CONFIDENCE, urgency.reason)
            val candidate = assertIs<sk.ainet.decisions.Prediction.Score>(urgency.candidate)
            assertEquals(expected["urgency"]!!.jsonObject["score"]!!.jsonPrimitive.double, candidate.expected, 6e-4)
            assertEquals(1, candidate.selectedLevel)
            val duplicate = assertIs<QuestionOutcome.Accepted>(result.question("duplicate")).decision as Decision.Boolean
            assertEquals(true, duplicate.value)
            assertEquals(expected["duplicate"]!!.jsonObject["noul"]!!.jsonPrimitive.double, duplicate.prediction.probabilityTrue, 1.5e-4)
            assertEquals(fixture["usage"]!!.jsonObject["input_tokens"]!!.jsonPrimitive.content.toInt(), result.provenance.traces.sumOf { it.packedTokens })
            assertEquals("test-triage", result.provenance.policyId)
            assertEquals(LayaDecisionAdapter.PREPROCESSING_VERSION, result.provenance.preprocessingVersion)
            assertTrue(result.provenance.notes.isEmpty())
            println("[laya] typed API billing-mixed: ${result.outcomes.map { it::class.simpleName }} in ${result.provenance.durationMillis} ms")

            // Overflow: a state far beyond the context is a typed failure by default and a traced truncation when allowed.
            val long = DecisionRequest("long", DecisionState.Text("evidence sentence number one. ".repeat(400)), listOf(request.questions.last()), ModelSelector.Installed("laya-multilingual"))
            val rejected = runtime.openSession(ModelSelector.Installed("laya-multilingual")).decide(long, policy)
            val overflow = assertIs<DecisionError.ContextOverflow>(assertIs<QuestionOutcome.Failed>(rejected.question("duplicate")).error)
            assertEquals(1024, overflow.limitTokens)
            assertTrue(overflow.requiredTokens > 1024)
            val truncated = runtime.openSession(ModelSelector.Installed("laya-multilingual")).decide(long.copy(inference = InferencePolicy(truncation = TruncationPolicy.TRUNCATE_STATE)), policy)
            assertTrue(truncated.question("duplicate") !is QuestionOutcome.Failed)
            assertTrue(truncated.provenance.traces.single().stateTokensDropped > 0)
            assertEquals(1, truncated.provenance.notes.size)
        }
    }
}
