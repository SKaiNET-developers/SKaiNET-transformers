package sk.ainet.models.laya

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** The routing rules, the expert metadata and the per-expert calibration — no model needed. */
class LayaMixtureSpecTest {

    private val router = LayaRouterSpec(
        question = "What kind of text is this?",
        options = listOf(
            "agent_action" to "A plan or command an AI agent wants to run",
            "support_ticket" to "A customer asking for help",
            "it_alert" to "A monitoring alert",
            "other" to "Something else",
        ),
        experts = mapOf("agent_action" to "safety", "support_ticket" to "customer_ops"),
        threshold = 0.3,
    )

    private fun probs(vararg entries: Pair<String, Double>): Map<String, Double> = linkedMapOf(*entries)

    @Test
    fun the_top_kind_routes_to_its_expert_when_loaded_and_confident_enough() {
        val r = router.route(probs("agent_action" to 0.7, "support_ticket" to 0.2, "it_alert" to 0.05, "other" to 0.05), setOf("safety", "customer_ops"), confidence = 0.6)
        assertEquals("agent_action", r.top)
        assertEquals(0.7, r.topProbability)
        assertEquals("safety", r.routerPick)
        assertEquals("safety", r.chosen)
        assertFalse(r.overridden)
        assertEquals(0.6, r.confidence)
        assertEquals(0.3, r.threshold)
    }

    @Test
    fun below_the_threshold_unmapped_or_unloaded_kinds_fall_back_to_general() {
        val weak = router.route(probs("agent_action" to 0.29, "support_ticket" to 0.28, "it_alert" to 0.23, "other" to 0.2), setOf("safety", "customer_ops"), confidence = 0.1)
        assertEquals(LayaMixtureAgent.GENERAL, weak.routerPick, "0.29 < 0.3")
        val unmapped = router.route(probs("it_alert" to 0.9, "other" to 0.1), setOf("safety", "customer_ops"), confidence = 0.8)
        assertEquals(LayaMixtureAgent.GENERAL, unmapped.routerPick, "it_alert has no expert")
        val unloaded = router.route(probs("support_ticket" to 0.9, "other" to 0.1), setOf("safety"), confidence = 0.8)
        assertEquals(LayaMixtureAgent.GENERAL, unloaded.routerPick, "customer_ops is not loaded")
        val exact = router.route(probs("support_ticket" to 0.3, "other" to 0.7 - 0.7), setOf("customer_ops"), confidence = 0.0)
        assertEquals("customer_ops", exact.routerPick, "the threshold is inclusive")
    }

    @Test
    fun a_forced_head_is_recorded_as_an_override_beside_the_router_pick() {
        val r = router.route(probs("agent_action" to 0.9, "other" to 0.1), setOf("safety"), confidence = 0.9, forced = LayaMixtureAgent.GENERAL)
        assertEquals("safety", r.routerPick)
        assertEquals(LayaMixtureAgent.GENERAL, r.chosen)
        assertTrue(r.overridden)
    }

    @Test
    fun the_router_question_is_a_choice_over_the_kinds_in_order() {
        val q = router.layaQuestion
        assertEquals(LayaQuestionType.CHOICE, q.type)
        assertEquals(listOf("agent_action", "support_ticket", "it_alert", "other"), q.keys)
        assertEquals("agent_action: A plan or command an AI agent wants to run", q.renderOptions()[0])
        assertEquals(LayaRouterSpec.DEFAULT_THRESHOLD, LayaRouterSpec("q", router.options, emptyMap()).threshold)
    }

    @Test
    fun router_specs_reject_duplicates_bad_thresholds_and_unknown_kinds() {
        assertFailsWith<IllegalArgumentException> { LayaRouterSpec("q", listOf("a" to "x", "a" to "y"), emptyMap()) }
        assertFailsWith<IllegalArgumentException> { LayaRouterSpec("q", listOf("a" to "x", "b" to "y"), emptyMap(), threshold = 1.5) }
        assertFailsWith<IllegalArgumentException> { LayaRouterSpec("q", listOf("a" to "x", "b" to "y"), mapOf("c" to "e")) }
        assertFailsWith<IllegalArgumentException> { LayaRouterSpec("q", listOf("a" to "x"), emptyMap()) }
    }

    @Test
    fun expert_metadata_parses_as_published_and_carries_its_own_temperatures() {
        val meta = LayaExpertMeta.parse(
            """
            {"expert": "safety", "base_model": "convaiinnovations/laya-typed-decisions",
             "domains": ["agent-guardrails", "content-moderation"],
             "temperature": [1.45, 1.35, 0.9],
             "temperature_by_options": {"noul:2": 0.9, "score:3-5": 1.35, "choice:3-5": 1.45},
             "train": {"items": 3120, "val_items": 380, "epochs_run": 6, "kept_epoch": 6, "lr": 0.0006, "bs": 32,
                       "rps_weight": 1.0, "seed": 0, "val_acc": 0.7342, "val_acc_base_head": 0.5842}}
            """.trimIndent(),
        )
        assertEquals("safety", meta.expert)
        assertEquals(listOf("agent-guardrails", "content-moderation"), meta.domains)
        assertEquals("convaiinnovations/laya-typed-decisions", meta.baseModel)
        assertEquals(listOf(1.45, 1.35, 0.9), meta.temperature)
        assertEquals(3, meta.temperatureByOptions.size)
        assertEquals("3120", meta.train?.get("items").toString())

        val base = LayaAgentConfig(
            encoder = "answerdotai/ModernBERT-large", maxLen = 1024, headMaxLen = 256,
            temperature = listOf(1.0148, 1.0374, 1.0575),
            temperatureByOptions = mapOf("choice:3-5" to 1.76, "choice:6-10" to 1.00002, "score:3-5" to 1.2514, "noul:2" to 1.9834, "choice:11+" to 0.1006, "choice:2" to 1.9064),
        )
        val decoder = LayaExpertLoader.decoderFor(meta, base)
        assertEquals(1024, decoder.config.maxLen, "budgets stay the base's")
        assertEquals(256, decoder.config.headMaxLen)
        assertEquals(1.45, decoder.temperature(LayaQuestionType.CHOICE, 3), "the expert's bucket")
        assertEquals(0.9, decoder.temperature(LayaQuestionType.NOUL, 2))
        assertEquals(1.35, decoder.temperature(LayaQuestionType.SCORE, 4))
        assertEquals(1.45, decoder.temperature(LayaQuestionType.CHOICE, 10), "no choice:6-10 bucket for the expert: its per-type temperature, not the base's")
        assertEquals(1.0, LayaDecoder(base).temperature(LayaQuestionType.CHOICE, 10).let { kotlin.math.round(it * 1000) / 1000 }, "the base keeps its own buckets")

        val minimal = LayaExpertMeta.parse("""{"expert": "x"}""")
        assertNull(minimal.baseModel)
        assertEquals(listOf(1.0, 1.0, 1.0), minimal.temperature)
        assertFailsWith<IllegalArgumentException> { LayaExpertMeta.parse("""{"expert": "x", "temperature": [1.0]}""") }
        assertFailsWith<IllegalArgumentException> { LayaExpertMeta.parse("""{"expert": " "}""") }
    }
}
