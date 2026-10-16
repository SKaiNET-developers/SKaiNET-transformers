package sk.ainet.models.laya

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import sk.ainet.context.DirectCpuExecutionContext
import java.nio.file.Files
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.fail

/**
 * REQ-EXE / REQ-DEC acceptance against the reference `predict.json` (multilingual, fused ONNX graph
 * exported from the CPU FP32 agent): probabilities, scores, noul values, confidences and the token
 * usage of the six single-state cases. Needs `LAYA_MULTILINGUAL_PATH` with `model.safetensors`.
 */
class LayaPredictFixtureTest {

    private val probabilityTolerance = System.getenv("LAYA_PARITY_TOL")?.toDoubleOrNull() ?: 1.5e-4

    @Test
    fun `multilingual predict fixtures reproduce`() {
        val dir = LayaFixtures.checkpointDir() ?: return
        if (!Files.exists(dir.resolve("model.safetensors"))) { println("[skip] $dir has no model.safetensors"); return }
        val exactGelu = System.getenv("LAYA_EXACT_GELU")?.let { it != "false" } ?: true
        val t0 = System.nanoTime()
        val agent = LayaModelLoader.load(dir, DirectCpuExecutionContext(), exactGelu = exactGelu)
        println("[laya] model loaded in ${(System.nanoTime() - t0) / 1_000_000} ms, exactGelu=$exactGelu")

        val fixture = LayaFixtures.resource("predict.json")
        val failures = mutableListOf<String>()
        var compared = 0
        var maxDiff = 0.0
        for ((name, element) in fixture["single"]!!.jsonObject) {
            val case = element.jsonObject
            val state = LayaState.of(case["state"]!!)
            val questions = case["questions"]!!.jsonObject.mapValues { (_, q) -> LayaFixtures.questionOf(q.jsonObject) }
            val t1 = System.nanoTime()
            val results = agent.decideAll(state, questions)
            println("[laya] $name: ${questions.size} questions in ${(System.nanoTime() - t1) / 1_000_000} ms")
            val expectedAnswers = case["answers"]!!.jsonObject
            for ((qid, result) in results) {
                val expected = expectedAnswers[qid]!!.jsonObject
                val got = result.answer
                fun check(what: String, g: Double, e: Double, tol: Double = probabilityTolerance) {
                    compared++
                    val diff = abs(g - e)
                    if (diff > maxDiff) maxDiff = diff
                    if (diff > tol) failures += "$name/$qid $what: got ${"%.6f".format(g)} expected $e (diff ${"%.2e".format(diff)})"
                }
                when (got) {
                    is LayaAnswer.Choice -> {
                        if (got.choice != expected["choice"]!!.jsonPrimitive.content) failures += "$name/$qid choice: got ${got.choice} expected ${expected["choice"]}"
                        for ((key, p) in expected["probabilities"]!!.jsonObject) check("p[$key]", got.byKey.getValue(key), p.jsonPrimitive.double)
                    }
                    is LayaAnswer.Score -> {
                        check("score", got.score, expected["score"]!!.jsonPrimitive.double, tol = probabilityTolerance * 4)
                        for ((level, p) in expected["probabilities"]!!.jsonObject) check("p[$level]", got.probabilities[level.toInt()], p.jsonPrimitive.double)
                    }
                    is LayaAnswer.Noul -> check("noul", got.noul, expected["noul"]!!.jsonPrimitive.double)
                }
                check("confidence", got.confidence, expected["confidence"]!!.jsonPrimitive.double, tol = probabilityTolerance * 4)
                check("answer_confidence", got.answerConfidence, expected["answer_confidence"]!!.jsonPrimitive.double)
            }
            val usage = case["usage"]!!.jsonObject
            val inputTokens = results.values.sumOf { it.sequence.length }
            if (inputTokens != usage["input_tokens"]!!.jsonPrimitive.int) failures += "$name usage.input_tokens: got $inputTokens expected ${usage["input_tokens"]}"
            val stateTokens = results.values.first().sequence.truncation.stateTokens
            if (stateTokens != usage["state_tokens"]!!.jsonPrimitive.int) failures += "$name usage.state_tokens: got $stateTokens expected ${usage["state_tokens"]}"
        }
        println("[laya] predict.json: $compared values compared, max |diff| = ${"%.3e".format(maxDiff)}, ${failures.size} outside tolerance $probabilityTolerance")
        if (failures.isNotEmpty()) fail(failures.joinToString("\n"))
    }

    /**
     * REQ-EXE-005: the questions of one request as padded batches (`decideAll`) equal the unpadded
     * single passes (`decide`) within the parity tolerance, and the time per question of a 6- and a
     * 12-question request is reported for both. Needs the checkpoint like the test above.
     */
    @Test
    fun `multilingual batched requests match single passes and report timing`() {
        val dir = LayaFixtures.checkpointDir() ?: return
        if (!Files.exists(dir.resolve("model.safetensors"))) { println("[skip] $dir has no model.safetensors"); return }
        val agent = LayaModelLoader.load(dir, DirectCpuExecutionContext())
        val fixture = LayaFixtures.resource("predict.json")
        val cases = fixture["single"]!!.jsonObject.entries.map { (name, element) ->
            val case = element.jsonObject
            Triple(name, LayaState.of(case["state"]!!), case["questions"]!!.jsonObject.mapValues { (_, q) -> LayaFixtures.questionOf(q.jsonObject) })
        }
        // Warm-up: the first pass pays for lazy allocations.
        cases.first().let { (_, state, questions) -> agent.decideAll(state, questions) }

        var maxDiff = 0.0
        var compared = 0
        for ((name, state, questions) in cases) {
            val t0 = System.nanoTime()
            val batched = agent.decideAll(state, questions)
            val batchedMs = (System.nanoTime() - t0) / 1_000_000
            val t1 = System.nanoTime()
            val stateIds = agent.preprocessor.builder.encodeState(state)
            val single = questions.mapValues { (_, q) -> agent.decide(state, q, stateIds = stateIds) }
            val singleMs = (System.nanoTime() - t1) / 1_000_000
            for ((qid, s) in single) {
                val b = batched.getValue(qid)
                assertEquals(s.logits.size, b.logits.size)
                for (i in s.logits.indices) {
                    compared++
                    maxDiff = maxOf(maxDiff, abs(s.logits[i] - b.logits[i]).toDouble())
                }
            }
            val n = questions.size
            println("[laya] $name: $n questions — batched ${batchedMs} ms (${batchedMs / n} ms/question), single $singleMs ms (${singleMs / n} ms/question)")
        }
        // One request with every fixture question about the first case's state (the long-state case
        // is left out: its single sequence runs alone either way).
        val (_, stateMany, _) = cases.first()
        val many = cases.filter { it.first != "long-state" }.flatMap { it.third.entries.map { (k, q) -> "${it.first}/$k" to q } }.toMap()
        run {
            val n = many.size
            val t0 = System.nanoTime()
            val batched = agent.decideAll(stateMany, many)
            val batchedMs = (System.nanoTime() - t0) / 1_000_000
            val t1 = System.nanoTime()
            val stateIds = agent.preprocessor.builder.encodeState(stateMany)
            val single = many.mapValues { (_, q) -> agent.decide(stateMany, q, stateIds = stateIds) }
            val singleMs = (System.nanoTime() - t1) / 1_000_000
            for ((qid, s) in single) for (i in s.logits.indices) {
                compared++
                maxDiff = maxOf(maxDiff, abs(s.logits[i] - batched.getValue(qid).logits[i]).toDouble())
            }
            println("[laya] $n questions, one request — batched $batchedMs ms (${batchedMs / n} ms/question), single $singleMs ms (${singleMs / n} ms/question)")
        }
        println("[laya] batched vs single logits: $compared compared, max |diff| = ${"%.3e".format(maxDiff)}")
        if (maxDiff > 1e-4) fail("batched vs single logits diverge: max |diff| = $maxDiff")
    }
}
