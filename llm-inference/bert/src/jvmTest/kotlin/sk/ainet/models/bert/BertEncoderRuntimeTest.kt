package sk.ainet.models.bert

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.weights.WeightTensor
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Synthetic-weight tests for [BertEncoderRuntime] on the DSL path — ports the
 * behavioural cases of the legacy eager-runtime test suite: output shapes,
 * L2 normalization, mask-weighted pooling, projection, input sensitivity.
 */
class BertEncoderRuntimeTest {

    private val ctx = DirectCpuExecutionContext()

    private val h = 4
    private val inter = 8
    private val vocab = 10
    private val maxPos = 16
    private val typeVocab = 2

    private fun config(projDim: Int? = null, layers: Int = 2) = BertModelConfig(
        vocabSize = vocab,
        hiddenSize = h,
        numHiddenLayers = layers,
        numAttentionHeads = 2,
        intermediateSize = inter,
        maxPositionEmbeddings = maxPos,
        typeVocabSize = typeVocab,
        layerNormEps = 1e-5,
        projectionDim = projDim,
    )

    private fun varied(rows: Int, cols: Int, scale: Float): Tensor<FP32, Float> {
        val data = FloatArray(rows * cols) { idx ->
            val r = idx / cols
            val c = idx % cols
            scale * (r + 1) * (((r + c) % cols) + 1)
        }
        return ctx.fromFloatArray(Shape(rows, cols), FP32::class, data)
    }

    private fun identityLike(rows: Int, cols: Int, scale: Float): Tensor<FP32, Float> {
        val data = FloatArray(rows * cols) { idx ->
            val r = idx / cols
            val c = idx % cols
            if (r % cols == c) scale else 0f
        }
        return ctx.fromFloatArray(Shape(rows, cols), FP32::class, data)
    }

    private fun vec(size: Int, v: Float): Tensor<FP32, Float> =
        ctx.fromFloatArray(Shape(size), FP32::class, FloatArray(size) { v })

    /** WeightTensor list shaped like a bare sentence-transformers checkpoint. */
    private fun syntheticTensors(cfg: BertModelConfig, withProjection: Boolean = false): List<WeightTensor<FP32, Float>> {
        val out = mutableListOf<WeightTensor<FP32, Float>>()
        fun add(name: String, t: Tensor<FP32, Float>) {
            out += WeightTensor(name, t.shape.dimensions.toList(), t)
        }
        add("embeddings.word_embeddings.weight", varied(vocab, h, 0.2f))
        add("embeddings.position_embeddings.weight", varied(maxPos, h, 0.1f))
        add("embeddings.token_type_embeddings.weight", varied(typeVocab, h, 0.05f))
        add("embeddings.LayerNorm.weight", vec(h, 1f))
        add("embeddings.LayerNorm.bias", vec(h, 0f))
        for (l in 0 until cfg.numHiddenLayers) {
            val p = "encoder.layer.$l"
            add("$p.attention.self.query.weight", identityLike(h, h, 0.5f))
            add("$p.attention.self.query.bias", vec(h, 0f))
            add("$p.attention.self.key.weight", identityLike(h, h, 0.5f))
            add("$p.attention.self.key.bias", vec(h, 0f))
            add("$p.attention.self.value.weight", identityLike(h, h, 0.5f))
            add("$p.attention.self.value.bias", vec(h, 0f))
            add("$p.attention.output.dense.weight", identityLike(h, h, 0.5f))
            add("$p.attention.output.dense.bias", vec(h, 0f))
            add("$p.attention.output.LayerNorm.weight", vec(h, 1f))
            add("$p.attention.output.LayerNorm.bias", vec(h, 0f))
            add("$p.intermediate.dense.weight", identityLike(inter, h, 0.5f))
            add("$p.intermediate.dense.bias", vec(inter, 0f))
            add("$p.output.dense.weight", identityLike(h, inter, 0.5f))
            add("$p.output.dense.bias", vec(h, 0f))
            add("$p.output.LayerNorm.weight", vec(h, 1f))
            add("$p.output.LayerNorm.bias", vec(h, 0f))
        }
        if (withProjection) {
            add(BertNetworkLoader.PROJECTION_WEIGHT, identityLike(cfg.projectionDim!!, h, 1f))
            add(BertNetworkLoader.PROJECTION_BIAS, vec(cfg.projectionDim!!, 0f))
        }
        return BertNetworkLoader.normalizeTensorNames(out)
    }

    private fun runtime(cfg: BertModelConfig = config(), withProjection: Boolean = false): BertEncoderRuntime<FP32> =
        createBertEncoderRuntime(cfg, syntheticTensors(cfg, withProjection), ctx)

    @Test
    fun forward_producesCorrectShape() {
        val out = runtime().forward(intArrayOf(1, 2, 3))
        assertEquals(2, out.rank)
        assertEquals(3, out.shape[0])
        assertEquals(h, out.shape[1])
    }

    @Test
    fun forward_singleToken() {
        assertEquals(Shape(1, h), runtime().forward(intArrayOf(0)).shape)
    }

    @Test
    fun encode_producesVectorOfRuntimeDimensions() {
        val rt = runtime()
        assertEquals(h, rt.dimensions)
        assertEquals(h, rt.encode(intArrayOf(1, 2, 3)).size)
    }

    @Test
    fun encode_isL2Normalized() {
        val e = runtime().encode(intArrayOf(1, 2, 3))
        val norm = sqrt(e.sumOf { (it * it).toDouble() }).toFloat()
        assertTrue(abs(norm - 1f) < 0.01f, "L2 norm should be ~1.0, got $norm")
    }

    @Test
    fun encode_withAttentionMask_normalizedAndDiffersFromUnmasked() {
        val rt = runtime()
        val masked = rt.encode(intArrayOf(1, 2, 0), attentionMask = intArrayOf(1, 1, 0))
        val unmasked = rt.encode(intArrayOf(1, 2, 0))

        val norm = sqrt(masked.sumOf { (it * it).toDouble() }).toFloat()
        assertTrue(abs(norm - 1f) < 0.01f, "L2 norm should be ~1.0 with mask, got $norm")

        val diff = masked.indices.sumOf { abs(masked[it] - unmasked[it]).toDouble() }
        assertTrue(diff > 1e-4, "mask must change pooling: diff=$diff")
    }

    @Test
    fun encode_mismatchedMaskIsRejected() {
        assertFailsWith<IllegalArgumentException> {
            runtime().encode(intArrayOf(1, 2, 3), attentionMask = intArrayOf(1, 1))
        }
    }

    @Test
    fun forward_differentInputsProduceDifferentOutputs() {
        val rt = runtime()
        val a = rt.forward(intArrayOf(1, 2, 3))
        val b = rt.forward(intArrayOf(4, 5, 6))
        var allSame = true
        outer@ for (i in 0 until a.shape[0]) {
            for (j in 0 until a.shape[1]) {
                if (a.data.get(i, j) != b.data.get(i, j)) {
                    allSame = false
                    break@outer
                }
            }
        }
        assertTrue(!allSame, "Different inputs should produce different outputs")
    }

    @Test
    fun encode_withProjection_outputsProjectionDim() {
        val projDim = 2
        val rt = runtime(config(projDim = projDim, layers = 1), withProjection = true)
        assertEquals(projDim, rt.dimensions)
        val e = rt.encode(intArrayOf(1, 2, 3))
        assertEquals(projDim, e.size)
        val norm = sqrt(e.sumOf { (it * it).toDouble() }).toFloat()
        assertTrue(abs(norm - 1f) < 0.01f, "projected embedding must stay L2-normalized, got $norm")
    }

    @Test
    fun createRuntime_missingProjectionTensorsFailsWhenConfigDeclaresProjection() {
        val cfg = config(projDim = 2, layers = 1)
        assertFailsWith<IllegalArgumentException> {
            createBertEncoderRuntime(cfg, syntheticTensors(cfg, withProjection = false), ctx)
        }
    }

    @Test
    fun encode_emptyInputIsRejected() {
        assertFailsWith<IllegalArgumentException> { runtime().encode(intArrayOf()) }
    }
}
