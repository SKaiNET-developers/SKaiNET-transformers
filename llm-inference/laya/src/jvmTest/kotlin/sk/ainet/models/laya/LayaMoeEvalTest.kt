package sk.ainet.models.laya

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import kotlinx.coroutines.runBlocking
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * The expert heads against the published layaMOE evaluation (`laya-moe/moe_eval.json`, Apache-2.0,
 * see `laya-moe/NOTICE.md`): for each replayed case the mixture must route to the same expert, see
 * the same router kind, and the expert's and the base head's answers must reproduce the PyTorch FP32
 * values — probabilities, scores, noul values and confidences — within [tolerance].
 *
 * Needs `LAYA_TYPED_DECISIONS_PATH` (the `convaiinnovations/laya-typed-decisions` checkpoint with
 * `model.safetensors`) and `LAYA_MOE_EXPERTS_PATH` (`safety/` and `customer_ops/` from
 * `VishalMysore/layaMOE`); skipped otherwise. The first `LAYA_MOE_EVAL_CASES` cases per domain are
 * replayed (default 2, 108 in all with `all`); a full replay takes minutes on a laptop.
 */
class LayaMoeEvalTest {

    private val tolerance = System.getenv("LAYA_MOE_TOL")?.toDoubleOrNull() ?: 2e-3

    @Test
    fun `routed answers reproduce the published layaMOE evaluation`() {
        val baseDir = LayaFixtures.checkpointDir(LayaFixtures.TYPED_DECISIONS_ENV) ?: return
        if (!Files.exists(baseDir.resolve("model.safetensors"))) { println("[skip] $baseDir has no model.safetensors"); return }
        val expertsDir = System.getenv(LayaFixtures.MOE_EXPERTS_ENV)?.trim().orEmpty().takeIf { it.isNotEmpty() }?.let(Path::of)
            ?: run { println("[skip] ${LayaFixtures.MOE_EXPERTS_ENV} not set"); return }
        val perDomain = System.getenv("LAYA_MOE_EVAL_CASES")?.let { if (it == "all") Int.MAX_VALUE else it.toIntOrNull() } ?: 2

        val ctx = DirectCpuExecutionContext()
        val t0 = System.nanoTime()
        val base = LayaModelLoader.load(baseDir, ctx)
        val experts = listOf("safety", "customer_ops").map { name ->
            val dir = expertsDir.resolve(name)
            runBlocking {
                LayaExpertLoader.load(
                    metaJson = Files.readString(dir.resolve("expert.json")),
                    weights = { JvmRandomAccessSource.open(dir.resolve("expert.safetensors").toString()) },
                    base = base, ctx = ctx,
                )
            }
        }
        val mixture = LayaMixtureAgent(base, experts, LayaMoeFixtures.ROUTER)
        println("[laya-moe] base + ${experts.map { it.name }} loaded in ${(System.nanoTime() - t0) / 1_000_000} ms")

        val published = Json.parseToJsonElement(LayaFixtures.resourceText("/laya-moe/moe_eval.json")).jsonArray.map { it.jsonObject }
        val byCase = published.groupBy { it["domain"]!!.jsonPrimitive.content to it["id"]!!.jsonPrimitive.content }
        val evalFiles = HashMap<String, JsonObject>()
        fun domainFile(domain: String): JsonObject = evalFiles.getOrPut(domain) { Json.parseToJsonElement(LayaFixtures.resourceText("/laya-moe/eval/$domain.json")).jsonObject }

        val failures = mutableListOf<String>()
        var compared = 0
        var maxDiff = 0.0
        var replayed = 0
        val correct = linkedMapOf("general" to 0, "moe" to 0)
        var answers = 0
        var totalMs = 0L
        val seenPerDomain = HashMap<String, Int>()
        for ((key, entries) in byCase) {
            val (domain, id) = key
            val n = seenPerDomain.getOrDefault(domain, 0)
            if (n >= perDomain) continue
            seenPerDomain[domain] = n + 1
            val file = domainFile(domain)
            val case = file["cases"]!!.jsonArray.map { it.jsonObject }.first { it["id"]!!.jsonPrimitive.content == id }
            val questions = file["questions"]!!.jsonObject.mapValues { (_, q) -> LayaFixtures.questionOf(q.jsonObject) }
            val state = LayaState.of(case["state"]!!)
            val expected = case["expected"]!!.jsonObject

            val t1 = System.nanoTime()
            val result = mixture.decide(state, questions, compare = true)
            val ms = (System.nanoTime() - t1) / 1_000_000
            totalMs += ms
            replayed++
            val routing = result.routing!!
            println("[laya-moe] $domain/$id: ${questions.size} questions in $ms ms (encoder ${result.encoderMillis}, heads ${result.headsMillis}); routed to ${result.expert} via ${routing.top} p=${"%.4f".format(routing.topProbability)}")

            val first = entries.first()
            val publishedExpert = first["routed_to"]!!.jsonPrimitive.content
            if (publishedExpert != result.expert) failures += "$domain/$id routed to ${result.expert}, published $publishedExpert"
            val publishedRouting = first["routing"]!!.jsonObject
            if (publishedRouting["top"]!!.jsonPrimitive.content != routing.top) failures += "$domain/$id router top ${routing.top}, published ${publishedRouting["top"]}"
            fun check(what: String, got: Double, exp: Double, tol: Double = tolerance) {
                compared++
                val diff = abs(got - exp)
                if (diff > maxDiff) maxDiff = diff
                if (diff > tol) failures += "$domain/$id $what: got ${"%.5f".format(got)} published $exp (diff ${"%.2e".format(diff)})"
            }
            for ((kind, p) in publishedRouting["probabilities"]!!.jsonObject) check("router p[$kind]", routing.probabilities.getValue(kind), p.jsonPrimitive.double)

            for (entry in entries) {
                val qid = entry["question"]!!.jsonPrimitive.content
                val mine = mapOf("moe" to result.answers.getValue(qid), "general" to (result.generalAnswers ?: result.answers).getValue(qid))
                for ((lane, got) in mine) {
                    val pub = entry["answers"]!!.jsonObject[lane]!!.jsonObject
                    when (got) {
                        is LayaAnswer.Choice -> {
                            if (got.choice != pub["choice"]!!.jsonPrimitive.content) failures += "$domain/$id/$qid[$lane] choice ${got.choice}, published ${pub["choice"]}"
                            for ((k, p) in pub["probabilities"]!!.jsonObject) check("$qid[$lane] p[$k]", got.byKey.getValue(k), p.jsonPrimitive.double)
                        }
                        is LayaAnswer.Score -> {
                            check("$qid[$lane] score", got.score, pub["score"]!!.jsonPrimitive.double, tol = tolerance * 4)
                            for ((level, p) in pub["probabilities"]!!.jsonObject) check("$qid[$lane] p[$level]", got.probabilities[level.toInt()], p.jsonPrimitive.double)
                        }
                        is LayaAnswer.Noul -> check("$qid[$lane] noul", got.noul, pub["noul"]!!.jsonPrimitive.double)
                    }
                    check("$qid[$lane] confidence", got.confidence, pub["confidence"]!!.jsonPrimitive.double, tol = tolerance * 4)
                    if (lane == "moe") answers++
                    if (LayaMoeFixtures.isCorrect(got, expected[qid]!!)) correct[lane] = correct.getValue(lane) + 1
                }
            }
        }
        println("[laya-moe] replayed $replayed cases, $answers answers in $totalMs ms (${if (replayed > 0) totalMs / replayed else 0} ms/case); $compared values compared, max |diff| ${"%.2e".format(maxDiff)}")
        println("[laya-moe] accuracy on the replayed cases: general ${correct["general"]}/$answers, routed ${correct["moe"]}/$answers")
        if (failures.isNotEmpty()) fail("${failures.size} mismatches:\n" + failures.joinToString("\n"))
        assertTrue(replayed > 0, "nothing replayed")
    }
}

/** The router of layaMOE (`laya_moe/moe.py`), and the correctness rule of its evaluation. */
internal object LayaMoeFixtures {
    val ROUTER: LayaRouterSpec = LayaRouterSpec(
        question = "What kind of text is this?",
        options = listOf(
            "agent_action" to "A plan or command an AI agent wants to run on software, data or infrastructure",
            "public_comment" to "A comment posted by a reader on a blog, forum or social media",
            "support_ticket" to "A customer asking a software company for help with its product",
            "delivery_issue" to "A problem with shipping or delivering a parcel",
            "work_email" to "An email between colleagues at work",
            "it_alert" to "A monitoring alert or report about an IT system incident",
            "patient_message" to "A message from a patient to a doctor or clinic",
            "product_review" to "A customer review of a product they bought",
            "sales_inquiry" to "A potential customer contacting a sales team",
            "other" to "Something else",
        ),
        experts = mapOf(
            "agent_action" to "safety", "public_comment" to "safety",
            "support_ticket" to "customer_ops", "delivery_issue" to "customer_ops", "work_email" to "customer_ops",
        ),
        threshold = 0.3,
    )

    /** `expected`: an option key for choice, a level index for score (the most probable level must match), `true`/`false` for noul (at 0.5). */
    fun isCorrect(answer: LayaAnswer, expected: kotlinx.serialization.json.JsonElement): Boolean = when (answer) {
        is LayaAnswer.Choice -> answer.choice == expected.jsonPrimitive.content
        is LayaAnswer.Score -> answer.probabilities.indices.maxBy { answer.probabilities[it] } == expected.jsonPrimitive.content.toInt()
        is LayaAnswer.Noul -> (answer.noul >= 0.5) == (expected.jsonPrimitive.booleanOrNull ?: false)
    }
}
