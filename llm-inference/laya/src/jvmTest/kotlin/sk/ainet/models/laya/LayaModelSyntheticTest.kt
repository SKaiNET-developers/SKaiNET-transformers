package sk.ainet.models.laya

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.weights.WeightTensor
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/** Shape, mapping and decoding checks on a tiny checkpoint-shaped tensor set; no download. */
class LayaModelSyntheticTest {
    private val ctx = DirectCpuExecutionContext()
    private val d = 8
    private val inter = 6
    private val vocab = 24
    private val layers = 3
    private val headLayers = 1

    private val encoderConfig = LayaEncoderConfig(
        modelType = "modernbert", hiddenSize = d, numAttentionHeads = 2, numHiddenLayers = layers, intermediateSize = inter,
        vocabSize = vocab, localAttention = 4, layerTypeNames = listOf("full_attention", "sliding_attention", "sliding_attention"),
        ropeParameters = mapOf("full_attention" to LayaRopeParameters(160000.0), "sliding_attention" to LayaRopeParameters(160000.0)),
    )
    private val agentConfig = LayaAgentConfig(encoder = "test", headLayers = headLayers, maxLen = 32, headMaxLen = 16, actCosts = mapOf("escalate" to 0.5))

    private var seed = 1
    private fun t(name: String, vararg dims: Int, scale: Float = 0.1f): WeightTensor<FP32, Float> {
        val n = dims.fold(1) { a, b -> a * b }
        val data = FloatArray(n) { i -> seed = (seed * 1103515245 + 12345) and 0x7fffffff; scale * ((seed % 2001) - 1000) / 1000f }
        return WeightTensor(name, dims.toList(), ctx.fromFloatArray(Shape(*dims), FP32::class, data))
    }
    private fun ones(name: String, size: Int) = WeightTensor(name, listOf(size), ctx.fromFloatArray<FP32, Float>(Shape(size), FP32::class, FloatArray(size) { 1f }))

    private fun checkpoint(): List<WeightTensor<FP32, Float>> = buildList {
        add(t("encoder.embeddings.tok_embeddings.weight", vocab, d, scale = 0.5f))
        add(ones("encoder.embeddings.norm.weight", d))
        for (i in 0 until layers) {
            val p = "encoder.layers.$i"
            if (i > 0) add(ones("$p.attn_norm.weight", d))
            add(t("$p.attn.Wqkv.weight", 3 * d, d)); add(t("$p.attn.Wo.weight", d, d))
            add(ones("$p.mlp_norm.weight", d)); add(t("$p.mlp.Wi.weight", 2 * inter, d)); add(t("$p.mlp.Wo.weight", d, inter))
        }
        add(ones("encoder.final_norm.weight", d))
        for (i in 0 until headLayers) {
            val p = "head.layers.$i"
            add(ones("$p.norm1.weight", d)); add(t("$p.norm1.bias", d)); add(ones("$p.norm2.weight", d)); add(t("$p.norm2.bias", d))
            add(t("$p.self_attn.in_proj_weight", 3 * d, d)); add(t("$p.self_attn.in_proj_bias", 3 * d))
            add(t("$p.self_attn.out_proj.weight", d, d)); add(t("$p.self_attn.out_proj.bias", d))
            add(t("$p.linear1.weight", 4 * d, d)); add(t("$p.linear1.bias", 4 * d)); add(t("$p.linear2.weight", d, 4 * d)); add(t("$p.linear2.bias", d))
        }
        add(ones("scorer.0.weight", d)); add(t("scorer.0.bias", d)); add(t("scorer.1.weight", d, d)); add(t("scorer.1.bias", d))
        add(t("scorer.3.weight", 1, d)); add(t("scorer.3.bias", 1))
        add(t("type_emb.weight", 3, d)); add(t("act_head.0.weight", 256, d + 4)); add(t("act_head.0.bias", 256)); add(t("act_head.2.weight", 2, 256)); add(t("act_head.2.bias", 2))
        add(ones("temperature", 3))
    }

    private fun model(exactGelu: Boolean = true): LayaDecisionModel<FP32> {
        val prepared = LayaWeightPreparation.prepare(checkpoint(), encoderConfig, agentConfig, ctx, FP32::class)
        assertEquals(listOf("act_head.0.weight", "act_head.0.bias", "act_head.2.weight", "act_head.2.bias"), prepared.ignored)
        return LayaDecisionModel.build(encoderConfig, agentConfig, prepared, ctx, FP32::class, exactGelu)
    }

    @Test
    fun every_parameter_is_bound_and_shapes_follow_the_reference() {
        val m = model()
        val ids = intArrayOf(2, 5, 7, 1, 4, 9, 4, 11, 1, 3, 1)
        val h = m.hidden(ids, LayaQuestionType.CHOICE)
        assertEquals(listOf(ids.size, d), h.shape.dimensions.toList())
        val logits = m.logits(ids, intArrayOf(4, 6), LayaQuestionType.CHOICE)
        assertEquals(2, logits.size)
        assertTrue(logits.all { it.isFinite() })
        // The type embedding is part of the computation: another type gives other logits.
        val other = m.logits(ids, intArrayOf(4, 6), LayaQuestionType.NOUL)
        assertTrue(logits.indices.any { abs(logits[it] - other[it]) > 1e-6f })
    }

    @Test
    fun missing_or_unknown_tensors_fail_loading() {
        val without = checkpoint().filter { it.name != "head.layers.0.linear1.bias" }
        assertFailsWith<UnsupportedLayaCheckpointException> { LayaWeightPreparation.prepare(without, encoderConfig, agentConfig, ctx, FP32::class) }
        val extra = checkpoint() + t("decoder.weight", 4, 4)
        assertFailsWith<UnsupportedLayaCheckpointException> { LayaWeightPreparation.prepare(extra, encoderConfig, agentConfig, ctx, FP32::class) }
    }

    @Test
    fun exact_gelu_matches_known_values_and_differs_from_tanh_form() {
        assertEquals(0.8413447f, LayaGelu.geluErf(1f), 1e-6f)
        assertEquals(-0.15865526f, LayaGelu.geluErf(-1f), 1e-6f)
        assertEquals(0.34573123f, LayaGelu.geluErf(0.5f), 1e-6f)
        assertEquals(0f, LayaGelu.geluErf(0f))
        val ids = intArrayOf(2, 5, 7, 1, 4, 9, 4, 11, 1)
        val a = model(exactGelu = true).logits(ids, intArrayOf(4, 6), LayaQuestionType.SCORE)
        val b = model(exactGelu = false).logits(ids, intArrayOf(4, 6), LayaQuestionType.SCORE)
        assertTrue(a.indices.any { abs(a[it] - b[it]) > 1e-7f }, "tanh and erf GELU must not coincide")
    }

    @Test
    fun a_padded_batch_reproduces_the_single_sequence_logits() {
        // Sequences of different lengths, types and marker counts through one encoder pass
        // (REQ-EXE-005): every logit equals its unpadded single pass, and the layer kinds cover the
        // full and the sliding (right-context) attention of the encoder plus the biased head.
        val m = model()
        fun seq(ids: IntArray, markers: IntArray) = LayaSequence(
            ids, markers, LayaOptionStats(markers.size, markers.size, null), LayaTruncation(0, 0, 0), IntArray(markers.size) { it },
        )
        val sequences = listOf(
            seq(intArrayOf(2, 5, 7, 1, 4, 9, 4, 11, 1, 3, 1), intArrayOf(4, 6)),
            seq(intArrayOf(2, 8, 1, 6, 1), intArrayOf(2, 4)),
            seq(intArrayOf(2, 3, 3, 5, 7, 1, 9, 1, 13, 1, 17, 1, 19, 1, 1), intArrayOf(5, 7, 9, 11)),
            seq(intArrayOf(2, 1), intArrayOf(1)),
        )
        val types = listOf(LayaQuestionType.CHOICE, LayaQuestionType.NOUL, LayaQuestionType.SCORE, LayaQuestionType.CHOICE)
        val batched = m.logitsBatch(sequences, types, padId = 0)
        var maxDiff = 0f
        for (i in sequences.indices) {
            val single = m.logits(sequences[i], types[i])
            assertEquals(single.size, batched[i].size)
            for (j in single.indices) maxDiff = maxOf(maxDiff, abs(single[j] - batched[i][j]))
        }
        assertTrue(maxDiff < 1e-5f, "batched vs single logits: max |diff| = $maxDiff")
        // Padding content must not reach the real rows: another pad id gives the same logits.
        val otherPad = m.logitsBatch(sequences, types, padId = 17)
        for (i in sequences.indices) for (j in batched[i].indices) assertEquals(batched[i][j], otherPad[i][j], 1e-6f)
    }

    @Test
    fun engine_exact_gelu_matches_the_reference_formula() {
        // The engine op LayaGelu now delegates to, checked against the double-precision reference
        // that was the host-side implementation before engine 0.59.
        val xs = FloatArray(1201) { -6f + it * 0.01f }
        val input = ctx.fromFloatArray<FP32, Float>(Shape(xs.size), FP32::class, xs)
        val out = LayaGelu<FP32, Float>(FP32::class, exact = true).forward(input, ctx).data.copyToFloatArray()
        var maxDiff = 0f
        for (i in xs.indices) maxDiff = maxOf(maxDiff, abs(out[i] - LayaGelu.geluErf(xs[i])))
        assertTrue(maxDiff <= 1e-6f, "engine gelu(approximate = false) vs reference: max |diff| = $maxDiff")
    }

    @Test
    fun resolver_builds_tensor_names_from_module_paths() {
        val r = LayaSafeTensorsNameResolver()
        assertEquals("encoder.layers.3.attn.q_proj.weight", r.resolve("encoder/layers.3/attn", "attn.q_proj.weight"))
        assertEquals("encoder.embeddings.tok_embeddings.weight", r.resolve("encoder/embeddings/EmbeddingAdapter/tok_embeddings", "tok_embeddings.weight"))
        assertEquals("head.layers.0.linear1.bias", r.resolve("head/layers.0/ffn/linear1", "linear1.bias"))
        assertEquals("scorer.3.weight", r.resolve("scorer/3", "3.weight"))
        assertEquals(null, r.resolve("other/x", "x.weight"))
    }

    @Test
    fun decoder_applies_buckets_unpermutes_and_types_answers() {
        val cfg = agentConfig.copy(temperature = listOf(2.0, 1.0, 1.0), temperatureByOptions = mapOf("choice:3-5" to 0.1, "noul:2" to 1.0))
        val dec = LayaDecoder(cfg)
        assertEquals(2.0, dec.temperature(LayaQuestionType.CHOICE, 2))
        assertEquals(0.5, dec.temperature(LayaQuestionType.CHOICE, 3), "fitted 0.1 is clamped to TEMP_MIN")
        assertEquals("choice:11+", LayaDecoder.bucket(LayaQuestionType.CHOICE, 12))
        val q = LayaQuestion.Choice("i", mapOf("a" to null, "b" to null, "c" to null))
        // slots hold options [2, 0, 1]; slot 0 has the big logit, so option c wins
        val answer = dec.decode(q, floatArrayOf(4f, 0f, 0f), intArrayOf(2, 0, 1)) as LayaAnswer.Choice
        assertEquals("c", answer.choice)
        assertTrue(answer.byKey.getValue("c") > 0.99)
        assertEquals(1.0, answer.probabilities.sum(), 1e-9)
        val score = dec.decode(LayaQuestion.Score("i", listOf("x", "y", "z"), Unit), floatArrayOf(0f, 0f, 0f), intArrayOf(0, 1, 2)) as LayaAnswer.Score
        assertEquals(1.0, score.score, 1e-9)
        assertEquals(0.0, score.confidence, 1e-9)
        val noul = dec.decode(LayaQuestion.Noul("i"), floatArrayOf(0f, 2f), intArrayOf(0, 1)) as LayaAnswer.Noul
        assertEquals(1.0 / (1.0 + Math.exp(-2.0)), noul.noul, 1e-9)
    }
}
