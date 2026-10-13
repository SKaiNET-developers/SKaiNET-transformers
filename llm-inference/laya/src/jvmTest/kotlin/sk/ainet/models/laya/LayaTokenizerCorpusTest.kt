package sk.ainet.models.laya

import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.fail

/**
 * REQ-PRE-010 acceptance on the reference corpus: 40 texts (scripts, normalisation forms, emoji,
 * zero-width and RTL marks, literal mask strings, byte fallback) plus the option-truncation pair.
 */
class LayaTokenizerCorpusTest {

    @Test
    fun `multilingual tokenizer corpus matches the reference ids`() {
        val pre = LayaFixtures.preprocessor() ?: return
        val root = LayaFixtures.resource("tokenizer_ids.json")
        val texts = root["corpus_text"]!!.jsonObject
        val expected = root["multilingual"]!!.jsonObject["corpus"]!!.jsonObject
        val failures = mutableListOf<String>()
        var passed = 0
        for ((key, idsElement) in expected) {
            val text = texts[key]?.jsonPrimitive?.content ?: run { failures += "$key: no corpus_text"; continue }
            val got = pre.tokenizer.encode(text, addSpecialTokens = false).toList()
            val want = LayaFixtures.ints(idsElement)
            if (got == want) passed++ else failures += "$key (${text.take(40).replace("\n", "\\n")}): got $got expected $want"
        }
        println("[laya] tokenizer_ids.json multilingual corpus: $passed passed, ${failures.size} failed")
        if (failures.isNotEmpty()) fail(failures.joinToString("\n"))
    }

    @Test
    fun `option truncation keeps the first 48 tokens`() {
        val pre = LayaFixtures.preprocessor() ?: return
        val node = LayaFixtures.resource("tokenizer_ids.json")["multilingual"]!!.jsonObject["option_truncation"]!!.jsonObject
        val text = node["text"]!!.jsonPrimitive.content
        val uncapped = LayaFixtures.ints(node["uncapped"])
        val capped = LayaFixtures.ints(node["capped_48"])
        val got = pre.tokenizer.encode(" $text", addSpecialTokens = false).toList()
        val gotPlain = pre.tokenizer.encode(text, addSpecialTokens = false).toList()
        val match = when {
            got == uncapped -> got
            gotPlain == uncapped -> gotPlain
            else -> fail("uncapped ids differ: with leading space $got, without $gotPlain, expected $uncapped")
        }
        assertEquals(capped, match.take(48))
    }
}
