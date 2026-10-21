package sk.ainet.models.laya

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.decisions.AdapterOutcome
import sk.ainet.decisions.BooleanCriteria
import sk.ainet.decisions.BooleanProbabilityQuestion
import sk.ainet.decisions.CalibrationStatus
import sk.ainet.decisions.ChoiceOption
import sk.ainet.decisions.ChoiceQuestion
import sk.ainet.decisions.DecisionState
import sk.ainet.decisions.InferencePolicy
import sk.ainet.decisions.Prediction
import sk.ainet.io.weights.WeightTensor
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Expert heads on a tiny checkpoint-shaped tensor set with the one-token-per-character tokenizer:
 * a head prepared on its own reproduces the base model bit for bit ("the split is exact"), a
 * mixture routes, overrides and compares as specified, and the mixture adapter answers like the
 * plain adapter when the routed head is the base head's twin. No download.
 */
class LayaExpertSyntheticTest {
    private val ctx = DirectCpuExecutionContext()
    private val d = 8
    private val inter = 6
    private val vocab = 160 // FakeTokenizer: id = code point + 10
    private val layers = 2
    private val headLayers = 1

    private val encoderConfig = LayaEncoderConfig(
        modelType = "modernbert", hiddenSize = d, numAttentionHeads = 2, numHiddenLayers = layers, intermediateSize = inter,
        vocabSize = vocab, localAttention = 4, layerTypeNames = listOf("full_attention", "sliding_attention"),
        ropeParameters = mapOf("full_attention" to LayaRopeParameters(160000.0), "sliding_attention" to LayaRopeParameters(10000.0)),
    )
    private val agentConfig = LayaAgentConfig(
        // One token per character: the budgets must hold the longest question head plus the state untruncated.
        encoder = "test", headLayers = headLayers, maxLen = 256, headMaxLen = 96,
        temperature = listOf(1.0, 1.0, 1.0), temperatureByOptions = mapOf("noul:2" to 1.2),
    )

    private var seed = 7
    private fun t(name: String, vararg dims: Int, scale: Float = 0.1f): WeightTensor<FP32, Float> {
        val n = dims.fold(1) { a, b -> a * b }
        val data = FloatArray(n) { seed = (seed * 1103515245 + 12345) and 0x7fffffff; scale * ((seed % 2001) - 1000) / 1000f }
        return WeightTensor(name, dims.toList(), ctx.fromFloatArray(Shape(*dims), FP32::class, data))
    }
    private fun ones(name: String, size: Int) = WeightTensor(name, listOf(size), ctx.fromFloatArray<FP32, Float>(Shape(size), FP32::class, FloatArray(size) { 1f }))

    private fun encoderTensors(): List<WeightTensor<FP32, Float>> = buildList {
        add(t("encoder.embeddings.tok_embeddings.weight", vocab, d, scale = 0.5f))
        add(ones("encoder.embeddings.norm.weight", d))
        for (i in 0 until layers) {
            val p = "encoder.layers.$i"
            if (i > 0) add(ones("$p.attn_norm.weight", d))
            add(t("$p.attn.Wqkv.weight", 3 * d, d)); add(t("$p.attn.Wo.weight", d, d))
            add(ones("$p.mlp_norm.weight", d)); add(t("$p.mlp.Wi.weight", 2 * inter, d)); add(t("$p.mlp.Wo.weight", d, inter))
        }
        add(ones("encoder.final_norm.weight", d))
    }

    /** `type_emb.*`, `head.*`, `scorer.*`, `act_head.*`: what an `expert.safetensors` holds. */
    private fun headTensors(scorerScale: Float = 0.1f): List<WeightTensor<FP32, Float>> = buildList {
        for (i in 0 until headLayers) {
            val p = "head.layers.$i"
            add(ones("$p.norm1.weight", d)); add(t("$p.norm1.bias", d)); add(ones("$p.norm2.weight", d)); add(t("$p.norm2.bias", d))
            add(t("$p.self_attn.in_proj_weight", 3 * d, d)); add(t("$p.self_attn.in_proj_bias", 3 * d))
            add(t("$p.self_attn.out_proj.weight", d, d)); add(t("$p.self_attn.out_proj.bias", d))
            add(t("$p.linear1.weight", 4 * d, d)); add(t("$p.linear1.bias", 4 * d)); add(t("$p.linear2.weight", d, 4 * d)); add(t("$p.linear2.bias", d))
        }
        add(ones("scorer.0.weight", d)); add(t("scorer.0.bias", d)); add(t("scorer.1.weight", d, d)); add(t("scorer.1.bias", d))
        add(t("scorer.3.weight", 1, d, scale = scorerScale)); add(t("scorer.3.bias", 1))
        add(t("type_emb.weight", 3, d)); add(t("act_head.0.weight", 16, d + 4)); add(t("act_head.0.bias", 16))
    }

    private val preprocessor = LayaPreprocessor(FakeTokenizer, agentConfig, encoderConfig, FakeTokenizer.specialIds)

    private fun agent(headTensors: List<WeightTensor<FP32, Float>>): LayaAgent<FP32> {
        val prepared = LayaWeightPreparation.prepare(encoderTensors() + headTensors, encoderConfig, agentConfig, ctx, FP32::class)
        return LayaAgent(preprocessor, LayaDecisionModel.build(encoderConfig, agentConfig, prepared, ctx, FP32::class))
    }

    private fun expert(name: String, headTensors: List<WeightTensor<FP32, Float>>, base: LayaAgent<FP32>, temperature: List<Double> = listOf(1.0, 1.0, 1.0), byOptions: Map<String, Double> = mapOf("noul:2" to 1.2)): LayaExpert<FP32> =
        LayaExpertLoader.fromTensors(LayaExpertMeta(name, domains = listOf("$name-domain"), temperature = temperature, temperatureByOptions = byOptions), headTensors, base, ctx, FP32::class)

    private val router = LayaRouterSpec(
        question = "What kind of text is this?",
        options = listOf("alpha_kind" to "about alpha", "beta_kind" to "about beta", "other" to "something else"),
        experts = mapOf("alpha_kind" to "alpha", "beta_kind" to "beta"),
        threshold = 0.0,
    )

    private val state = LayaState.Text("the invoices page is broken and finance is blocked")
    private val questions = linkedMapOf<String, LayaQuestion>(
        "team" to LayaQuestion.Choice("Which team?", mapOf("bug" to "broken", "how_to" to "usage", "access" to "login")),
        "urgency" to LayaQuestion.Score("How urgent?", listOf("wait", "today", "now")),
        "angry" to LayaQuestion.Noul("The customer is angry"),
    )

    @Test
    fun a_head_prepared_on_its_own_reproduces_the_base_model_exactly() {
        val headOnly = headTensors()
        seed = 7
        val base = agent(headTensors())
        val prepared = LayaWeightPreparation.prepareHead(headOnly, d, headLayers, ctx, FP32::class)
        assertEquals(listOf("act_head.0.weight", "act_head.0.bias"), prepared.ignored)
        val twin = LayaDecisionModel.buildHead(prepared, d, headLayers, FP32::class)
        assertEquals(d, twin.hiddenSize)

        for ((_, q) in questions) {
            val s = preprocessor.prepare(state, q)
            val encoded = base.model.encode(s.ids)
            val expected = base.model.logits(s, q.type)
            val viaDefault = base.model.logitsFrom(encoded, s, q.type)
            val viaTwin = base.model.logitsFrom(encoded, s, q.type, twin)
            assertContentEquals(expected, viaDefault, "default head on encoded states")
            assertContentEquals(expected, viaTwin, "twin head bound from head-only tensors")
        }
        // The batched path too: one encoder pass, two heads.
        val seqs = questions.values.map { preprocessor.prepare(state, it) }
        val types = questions.values.map { it.type }
        val encoded = base.model.encodeBatch(seqs, FakeTokenizer.specialIds.pad)
        val single = base.model.logitsBatch(seqs, types, FakeTokenizer.specialIds.pad)
        val twinBatch = base.model.logitsBatchFrom(seqs, encoded, types, twin)
        for (i in seqs.indices) assertContentEquals(single[i], twinBatch[i], "batch row $i")
    }

    @Test
    fun head_only_preparation_rejects_encoder_tensors_and_missing_parts() {
        assertFailsWith<UnsupportedLayaCheckpointException> { LayaWeightPreparation.prepareHead(encoderTensors() + headTensors(), d, headLayers, ctx, FP32::class) }
        assertFailsWith<UnsupportedLayaCheckpointException> { LayaWeightPreparation.prepareHead(headTensors().filter { it.name != "scorer.3.bias" }, d, headLayers, ctx, FP32::class) }
        assertFailsWith<UnsupportedLayaCheckpointException> { LayaWeightPreparation.prepareHead(headTensors(), d + 8, headLayers, ctx, FP32::class) }
    }

    @Test
    fun a_mixture_routes_overrides_and_compares_as_specified() {
        seed = 7
        val base = agent(headTensors())
        seed = 7
        val alpha = expert("alpha", headTensors(), base) // the base head's twin: same tensors, same temperatures
        val beta = expert("beta", headTensors(scorerScale = 0.9f), base, temperature = listOf(2.5, 2.5, 2.5), byOptions = emptyMap())
        val mixture = LayaMixtureAgent(base, listOf(alpha, beta), router)
        assertEquals(listOf("general", "alpha", "beta"), mixture.heads)
        assertFailsWith<IllegalArgumentException> { LayaMixtureAgent(base, listOf(alpha, alpha), router) }
        assertFailsWith<IllegalArgumentException> { mixture.decide(state, questions, expert = "gamma") }

        val routed = mixture.decide(state, questions, compare = true)
        val routing = assertNotNull(routed.routing)
        assertEquals(router.options.map { it.first }, routing.probabilities.keys.toList(), "probabilities in kind order")
        assertEquals(1.0, routing.probabilities.values.sum(), 1e-9)
        val recomputed = router.route(routing.probabilities, mixture.experts.keys, routing.confidence)
        assertEquals(recomputed.routerPick, routing.routerPick)
        assertEquals(routing.routerPick, routed.expert)
        assertFalse(routing.overridden)
        assertEquals(questions.keys.toList(), routed.answers.keys.toList())
        assertTrue(routed.inputTokens > routed.sequences.values.sumOf { it.length }, "the router question counts")
        assertEquals(questions.keys, routed.sequences.keys)

        // General answers: the plain agent's.
        val plain = base.decideAll(state, questions)
        val general = mixture.decide(state, questions, expert = LayaMixtureAgent.GENERAL, compare = true)
        assertEquals(LayaMixtureAgent.GENERAL, general.expert)
        assertNull(general.generalAnswers, "no comparison when general answered")
        assertTrue(general.routing!!.overridden || general.routing!!.routerPick == LayaMixtureAgent.GENERAL)
        for ((k, a) in general.answers) assertContentEquals(plain.getValue(k).answer.probabilities, a.probabilities, "general/$k")

        // The twin expert answers exactly like general; the comparison shows it.
        val viaAlpha = mixture.decide(state, questions, expert = "alpha", compare = true)
        assertEquals("alpha", viaAlpha.expert)
        val cmp = assertNotNull(viaAlpha.generalAnswers)
        for ((k, a) in viaAlpha.answers) {
            assertContentEquals(plain.getValue(k).answer.probabilities, a.probabilities, "alpha/$k")
            assertContentEquals(plain.getValue(k).answer.probabilities, cmp.getValue(k).probabilities, "compare/$k")
        }
        assertEquals(viaAlpha.routing!!.routerPick != "alpha", viaAlpha.routing!!.overridden)

        // Beta has other weights and a flatter calibration: different probabilities.
        val viaBeta = mixture.decide(state, questions, expert = "beta", compare = false)
        assertNull(viaBeta.generalAnswers)
        val differ = viaBeta.answers.any { (k, a) -> a.probabilities.indices.any { abs(a.probabilities[it] - plain.getValue(k).answer.probabilities[it]) > 1e-6 } }
        assertTrue(differ, "beta must not answer like general")
        val unrouted = mixture.decide(state, questions, expert = "beta", route = false)
        assertNull(unrouted.routing)
        for ((k, a) in unrouted.answers) assertContentEquals(viaBeta.answers.getValue(k).probabilities, a.probabilities, "unrouted/$k")
    }

    @Test
    fun the_mixture_adapter_answers_like_the_plain_adapter_when_the_twin_head_is_routed() {
        seed = 7
        val base = agent(headTensors())
        seed = 7
        val alpha = expert("alpha", headTensors(), base)
        // Every kind maps to alpha, so whatever the router sees, alpha answers.
        val everything = LayaRouterSpec(router.question, router.options, router.options.associate { it.first to "alpha" }, threshold = 0.0)
        val mixture = LayaMixtureAgent(base, listOf(alpha), everything)
        var routings = 0
        val moe = LayaMixtureDecisionAdapter(mixture, modelId = "laya-moe", onRouted = { routings++ })
        val plain = LayaDecisionAdapter(base, modelId = "laya")
        assertEquals(CalibrationStatus.CHECKPOINT_PROVIDED, moe.capabilities.calibration)
        assertEquals(plain.capabilities.maxContextTokens, moe.capabilities.maxContextTokens)
        assertEquals("laya-moe", moe.capabilities.modelId)

        val typed = listOf(
            ChoiceQuestion("team", "Which team?", listOf(ChoiceOption("bug", "broken"), ChoiceOption("how_to", "usage"), ChoiceOption("access", "login"))),
            BooleanProbabilityQuestion("angry", "The customer is angry", BooleanCriteria("no, the statement does not hold", "yes, the statement holds")),
        )
        val a = plain.predict(DecisionState.Text(state.text), typed, InferencePolicy())
        val b = moe.predict(DecisionState.Text(state.text), typed, InferencePolicy())
        assertEquals(1, routings)
        assertEquals(a.size, b.size)
        for (i in a.indices) {
            val x = assertIs<AdapterOutcome.Candidate>(a[i])
            val y = assertIs<AdapterOutcome.Candidate>(b[i])
            assertEquals(x.questionId, y.questionId)
            assertEquals(x.trace, y.trace)
            when (val p = x.prediction) {
                is Prediction.Choice -> {
                    val q = assertIs<Prediction.Choice>(y.prediction)
                    assertEquals(p.selectedOptionId, q.selectedOptionId)
                    assertEquals(p.probabilities, q.probabilities)
                }
                is Prediction.BooleanProbability -> assertEquals(p.probabilityTrue, assertIs<Prediction.BooleanProbability>(y.prediction).probabilityTrue)
                else -> error("unexpected $p")
            }
        }
    }
}
