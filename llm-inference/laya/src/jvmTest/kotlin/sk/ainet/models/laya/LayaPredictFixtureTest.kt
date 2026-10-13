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
            val questions = case["questions"]!!.jsonObject.mapValues { (_, q) -> questionOf(q.jsonObject) }
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

    /** The public request shape (`type`, `instructions`, `criteria`, `labels`), normalised as `Agent._to_internal` does. */
    private fun questionOf(q: JsonObject): LayaQuestion {
        val insElement = q["instructions"]!!
        val ins = if (insElement is JsonPrimitive && insElement.isString) insElement.content else PythonJson.dumps(insElement)
        val crit = q["criteria"]
        return when (q["type"]!!.jsonPrimitive.content) {
            "choice" -> when (crit) {
                is JsonArray -> LayaQuestion.Choice(ins, crit.map { it.jsonPrimitive.content to null })
                is JsonObject -> LayaQuestion.Choice(ins, crit.entries.map { (k, v) -> k to v.takeIf { it !is JsonNull } })
                else -> error("choice without criteria")
            }
            "score" -> LayaQuestion.Score(ins, (crit as JsonArray).toList())
            "noul" -> {
                val c = crit as? JsonObject
                val labels = (q["labels"] as? JsonObject)?.let { it["false"]!!.jsonPrimitive.content to it["true"]!!.jsonPrimitive.content }
                fun get(key: String): JsonElement? = c?.entries?.firstOrNull { it.key.lowercase() == key }?.value?.takeIf { it !is JsonNull }
                LayaQuestion.Noul(ins, get("false"), get("true"), labels)
            }
            else -> error("unknown type ${q["type"]}")
        }
    }
}
