package sk.ainet.models.laya

import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class LayaSequenceBuilderTest {

    private fun builder(maxLen: Int = 1024, headMaxLen: Int = 256) = LayaSequenceBuilder(FakeTokenizer, FakeTokenizer.specialIds, maxLen, headMaxLen)

    @Test
    fun layout_is_cls_head_sep_masked_options_sep_state_sep() {
        val q = LayaQuestion.Choice("Pick.", mapOf("a" to "x", "b" to null))
        val seq = builder().build(LayaState.of("st"), q)
        val head = "choice question: Pick."
        val expected = listOf(2) + FakeTokenizer.encode(head).toList() + listOf(1) +
            listOf(4) + FakeTokenizer.encode(" a: x").toList() + listOf(4) + FakeTokenizer.encode(" b").toList() +
            listOf(1) + FakeTokenizer.encode("st").toList() + listOf(1)
        assertEquals(expected, seq.ids.toList())
        assertContentEquals(intArrayOf(1 + head.length + 1, 1 + head.length + 1 + 1 + " a: x".length), seq.markers)
        assertEquals(LayaOptionStats(2, 2, null), seq.stats)
        assertFalse(seq.truncation.truncated)
    }

    @Test
    fun mask_literal_in_user_text_becomes_a_space() {
        val q = LayaQuestion.Choice("Is <mask> here?", mapOf("a<mask>b" to "c<mask>"))
        val head = builder().buildHead(q)
        val expectedHead = FakeTokenizer.encode("choice question: Is   here?").toList()
        assertEquals(expectedHead, head.ids.drop(1).take(expectedHead.size))
        assertEquals(FakeTokenizer.encode(" a b: c ").toList(), head.ids.drop(head.markers[0] + 1).dropLast(1))
    }

    @Test
    fun options_are_capped_at_48_tokens_then_budget_floor_applies() {
        val long = "x".repeat(100)
        val q = LayaQuestion.Choice("Pick.", mapOf("a" to long, "b" to long, "c" to long))
        val head = builder(headMaxLen = 256).buildHead(q)
        // each option: mask + 48 = 49 tokens; 3 * 49 = 147 <= 240, so no floor
        assertEquals(null, head.stats.tokensPerOption)
        assertEquals(49, head.markers[1] - head.markers[0])

        val tiny = builder(maxLen = 24, headMaxLen = 16).buildHead(q)
        // budget 16 - 147 < 16 -> per = max(4, (16 - 16) / 3) = 4
        assertEquals(4, tiny.stats.tokensPerOption)
        assertEquals(4, tiny.markers[1] - tiny.markers[0])
        // instruction keeps max(8, 16 - 12) = 8 tokens
        assertEquals(8, tiny.markers[0] - 2)
    }

    @Test
    fun colliding_options_are_counted() {
        val q = LayaQuestion.Choice("Pick.", mapOf("alpha" to "same prefix one", "alpha2" to "same prefix two"))
        val head = builder(headMaxLen = 16).buildHead(q)
        assertEquals(2, head.stats.options)
        assertEquals(1, head.stats.optionsDistinct, "both cut to ' alp' + mask")
    }

    @Test
    fun state_truncates_right_by_default_and_left_for_lists() {
        val text = LayaState.of("0123456789")
        val q = LayaQuestion.Choice("P", mapOf("a" to null))
        val b = builder(maxLen = 2 + "choice question: P".length + 1 + 2 + 1 + 4 + 1, headMaxLen = 64)
        val right = b.build(text, q)
        assertEquals(FakeTokenizer.encode("0123").toList(), right.ids.drop(right.ids.size - 5).dropLast(1))
        assertEquals(LayaTruncation(10, 4, 6), right.truncation)
        assertTrue(right.truncation.truncated)

        val left = b.build(text, q, truncateLeft = true)
        assertEquals(FakeTokenizer.encode("6789").toList(), left.ids.drop(left.ids.size - 5).dropLast(1))

        val list = LayaState.of(buildJsonArray { add(JsonPrimitive("a")); add(JsonPrimitive("b")) })
        assertTrue(list.truncatesLeft)
        assertFalse(LayaState.of(buildJsonObject { put("a", 1) }).truncatesLeft)
    }

    @Test
    fun head_larger_than_max_len_is_clamped_and_markers_filtered() {
        val q = LayaQuestion.Choice("Pick.", mapOf("a" to "x", "b" to "y"))
        val seq = builder(maxLen = 12, headMaxLen = 192).build(LayaState.of("state"), q)
        assertEquals(12, seq.ids.size)
        assertTrue(seq.markers.all { it < 12 })
        assertEquals(LayaTruncation(5, 0, 5), seq.truncation)
    }

    @Test
    fun option_order_permutes_slots_and_is_validated() {
        val q = LayaQuestion.Choice("Pick.", mapOf("a" to "x", "b" to "y", "c" to "z"))
        val seq = builder().build(LayaState.of("s"), q, optionOrder = intArrayOf(2, 0, 1))
        val firstSlot = seq.ids.drop(seq.markers[0] + 1).take(" c: z".length)
        assertEquals(FakeTokenizer.encode(" c: z").toList(), firstSlot)
        assertContentEquals(intArrayOf(2, 0, 1), seq.optionOrder)
        assertFailsWith<LayaSequenceException> { builder().build(LayaState.of("s"), q, optionOrder = intArrayOf(0, 0, 1)) }
        assertFailsWith<LayaSequenceException> { builder().build(LayaState.of("s"), q, optionOrder = intArrayOf(0, 1)) }
    }

    @Test
    fun render_options_follows_the_reference() {
        assertEquals(listOf("zero: 0", "false: false", "none"), LayaQuestion.Choice("i", listOf("zero" to JsonPrimitive(0), "false" to JsonPrimitive(false), "none" to null)).renderOptions())
        assertEquals(listOf("a: {\"desc\": \"nested\", \"n\": 1}", "b: [1, 2, \"x\"]", "c: 2.5"), LayaQuestion.Choice("i", listOf(
            "a" to buildJsonObject { put("desc", "nested"); put("n", 1) },
            "b" to buildJsonArray { add(JsonPrimitive(1)); add(JsonPrimitive(2)); add(JsonPrimitive("x")) },
            "c" to JsonPrimitive(2.5),
        )).renderOptions())
        assertEquals(listOf("level 0: {\"d\": 1}", "level 1: ok", "level 2: 3"), LayaQuestion.Score("i", listOf(buildJsonObject { put("d", 1) }, JsonPrimitive("ok"), JsonPrimitive(3))).renderOptions())
        assertEquals(listOf("false: no, the statement does not hold", "true: yes, the statement holds"), LayaQuestion.Noul("i").renderOptions())
        assertEquals(listOf("nope: no, the statement does not hold", "yep: yes, the statement holds"), LayaQuestion.Noul("i", labels = "nope" to "yep").renderOptions())
        assertFailsWith<IllegalArgumentException> { LayaQuestion.Noul("i", labels = "same" to " same ") }
    }

    @Test
    fun agent_config_rejects_the_parallel_layout_and_reads_budgets() {
        val cfg = LayaAgentConfig.parse("""{"encoder": "jhu-clsp/mmBERT-base", "head_layers": 2, "max_len": 1024, "head_max_len": 256, "act_costs": {"escalate": 0.5}, "temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}}""")
        assertEquals(1024, cfg.maxLen)
        assertEquals(2, cfg.actionCount)
        cfg.requireSupported()
        assertFailsWith<UnsupportedLayaCheckpointException> { cfg.copy(optionLayoutName = "parallel").requireSupported() }
        assertFailsWith<UnsupportedLayaCheckpointException> { cfg.copy(optionLayoutName = "spiral").optionLayout }
    }

    @Test
    fun encoder_config_derives_layer_kinds_and_rope_theta() {
        val cfg = LayaEncoderConfig.parse("""{"model_type": "modernbert", "hidden_size": 768, "num_attention_heads": 12, "num_hidden_layers": 6, "intermediate_size": 1152, "vocab_size": 256000, "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention"], "rope_parameters": {"full_attention": {"rope_theta": 160000, "rope_type": "default"}, "sliding_attention": {"rope_theta": 160000, "rope_type": "default"}}}""").requireSupported()
        assertEquals(listOf(LayaLayerKind.FULL, LayaLayerKind.SLIDING, LayaLayerKind.SLIDING, LayaLayerKind.FULL, LayaLayerKind.SLIDING, LayaLayerKind.SLIDING), cfg.layerKinds)
        assertEquals(160000.0, cfg.ropeTheta(LayaLayerKind.SLIDING))
        assertEquals(64, cfg.headDim)
        assertEquals(64, cfg.localHalfWindow)
        val legacy = LayaEncoderConfig.parse("""{"model_type": "modernbert", "hidden_size": 768, "num_attention_heads": 12, "num_hidden_layers": 4, "intermediate_size": 1152, "vocab_size": 10, "global_rope_theta": 160000.0, "local_rope_theta": 10000.0}""")
        assertEquals(listOf(LayaLayerKind.FULL, LayaLayerKind.SLIDING, LayaLayerKind.SLIDING, LayaLayerKind.FULL), legacy.layerKinds)
        assertEquals(10000.0, legacy.ropeTheta(LayaLayerKind.SLIDING))
        assertFailsWith<UnsupportedLayaCheckpointException> { cfg.copy(modelType = "bert").requireSupported() }
    }
}
