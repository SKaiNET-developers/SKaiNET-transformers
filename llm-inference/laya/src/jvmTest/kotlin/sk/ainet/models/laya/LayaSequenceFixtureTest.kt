package sk.ainet.models.laya

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * REQ-PRE acceptance: every `multilingual` case of the reference `sequences.json` must reproduce
 * ids, markers, option stats and truncation stats exactly, with the engine tokenizer.
 */
class LayaSequenceFixtureTest {

    @Test
    fun `multilingual sequence fixtures reproduce exactly`() {
        val pre = LayaFixtures.preprocessor() ?: return
        check("multilingual", pre, minCases = 26)
    }

    /** The same contract for the ModernBERT-large checkpoint (`laya-typed-decisions`, the expert heads' base). */
    @Test
    fun `english sequence fixtures reproduce exactly`() {
        val pre = LayaFixtures.preprocessor(LayaFixtures.TYPED_DECISIONS_ENV) ?: return
        check("english", pre, minCases = 26)
    }

    private fun check(section: String, pre: LayaPreprocessor, minCases: Int) {
        val fixture = LayaFixtures.resource("sequences.json")[section]!!.jsonObject
        val special = fixture["special_ids"]!!.jsonObject
        assertEquals(special["cls"]!!.jsonPrimitive.int, pre.specialIds.cls, "cls id")
        assertEquals(special["sep"]!!.jsonPrimitive.int, pre.specialIds.sep, "sep id")
        assertEquals(special["mask"]!!.jsonPrimitive.int, pre.specialIds.mask, "mask id")
        assertEquals(special["pad"]!!.jsonPrimitive.int, pre.specialIds.pad, "pad id")
        assertEquals(special["mask_token"]!!.jsonPrimitive.content, pre.specialIds.maskToken)

        val failures = mutableListOf<String>()
        var passed = 0
        for ((name, element) in fixture["cases"]!!.jsonObject) {
            val case = element.jsonObject
            try {
                val state = LayaState.of(case["state"]!!)
                val question = questionOf(case["question"]!!.jsonObject)
                val order = (case["option_order"] as? JsonArray)?.map { it.jsonPrimitive.int }?.toIntArray()
                val truncateLeft = case["truncate_left"]!!.jsonPrimitive.booleanOrNull ?: false
                val builder = LayaSequenceBuilder(pre.tokenizer, pre.specialIds, case["max_len"]!!.jsonPrimitive.int, case["head_max_len"]!!.jsonPrimitive.int)
                // The fixture exercises build_sequence with an explicit flag; the agent-level rule
                // (a list state truncates left) is covered by LayaSequenceBuilderTest.
                val seq = builder.build(state, question, order, truncateLeft = truncateLeft)

                val expectedIds = LayaFixtures.ints(case["ids"])
                val expectedMarkers = LayaFixtures.ints(case["markers"])
                val stats = case["stats"]!!.jsonObject
                val trunc = case["truncation"]!!.jsonObject
                val problems = mutableListOf<String>()
                if (seq.ids.toList() != expectedIds) problems += "ids differ: got ${seq.ids.toList()} expected $expectedIds"
                if (seq.markers.toList() != expectedMarkers) problems += "markers differ: got ${seq.markers.toList()} expected $expectedMarkers"
                val expectedStats = LayaOptionStats(stats["options"]!!.jsonPrimitive.int, stats["options_distinct"]!!.jsonPrimitive.int, (stats["tokens_per_option"] as? JsonPrimitive)?.takeIf { it !is JsonNull }?.int)
                if (seq.stats != expectedStats) problems += "stats differ: got ${seq.stats} expected $expectedStats"
                val expectedTrunc = LayaTruncation(trunc["state_tokens"]!!.jsonPrimitive.int, trunc["state_tokens_used"]!!.jsonPrimitive.int, trunc["state_tokens_dropped"]!!.jsonPrimitive.int)
                if (seq.truncation != expectedTrunc) problems += "truncation differs: got ${seq.truncation} expected $expectedTrunc"
                if (problems.isEmpty()) passed++ else failures += "$name:\n  " + problems.joinToString("\n  ")
            } catch (e: Exception) {
                failures += "$name: ${e::class.simpleName}: ${e.message}"
            }
        }
        println("[laya] sequences.json $section: $passed passed, ${failures.size} failed")
        if (failures.isNotEmpty()) fail(failures.joinToString("\n"))
        assertTrue(passed >= minCases, "expected at least $minCases cases, saw $passed")
    }

    private fun questionOf(q: JsonObject): LayaQuestion {
        val ins = q["ins"]!!.jsonPrimitive.content
        val crit = q["crit"]
        return when (q["t"]!!.jsonPrimitive.content) {
            "choice" -> LayaQuestion.Choice(ins, crit!!.jsonObject.entries.map { (k, v) -> k to (v.takeIf { it !is JsonNull }) })
            "score" -> LayaQuestion.Score(ins, crit!!.jsonArray.toList())
            "noul" -> {
                val c = (crit as? JsonObject)
                val labels = (q["labels"] as? JsonObject)?.let { it["false"]!!.jsonPrimitive.content to it["true"]!!.jsonPrimitive.content }
                LayaQuestion.Noul(ins, c?.get("false")?.takeIf { it !is JsonNull }, c?.get("true")?.takeIf { it !is JsonNull }, labels)
            }
            else -> error("unknown question type ${q["t"]}")
        }
    }
}
