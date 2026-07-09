package sk.ainet.models.bert

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.assertFailsWith

/**
 * Hand-computed verification of the [BertEmbeddings] additive block and the
 * name-resolver/mapping pipeline that populates it.
 *
 * Uses a 0-encoder-layer network so the model output IS the embeddings-block
 * output (word + position + token_type row 0, LayerNorm'd).
 */
class BertEmbeddingsTest {

    private val ctx = DirectCpuExecutionContext()

    private val h = 4
    private val vocab = 10
    private val maxPos = 16
    private val typeVocab = 2
    private val eps = 1e-5

    private fun config(layers: Int = 0) = BertModelConfig(
        vocabSize = vocab,
        hiddenSize = h,
        numHiddenLayers = layers,
        numAttentionHeads = 2,
        intermediateSize = 8,
        maxPositionEmbeddings = maxPos,
        typeVocabSize = typeVocab,
        layerNormEps = eps,
        projectionDim = null,
    )

    /** Row r = [base*(r+1), base*(r+1)*2, base*(r+1)*3, base*(r+1)*4] — varied so LayerNorm keeps signal. */
    private fun table(rows: Int, base: Float): FloatArray =
        FloatArray(rows * h) { idx ->
            val r = idx / h
            val c = idx % h
            base * (r + 1) * (c + 1)
        }

    private fun tensor2d(rows: Int, data: FloatArray): Tensor<FP32, Float> =
        ctx.fromFloatArray(Shape(rows, h), FP32::class, data)

    private fun tensor1d(data: FloatArray): Tensor<FP32, Float> =
        ctx.fromFloatArray(Shape(h), FP32::class, data)

    /** Bare sentence-transformers names — exercises the bert.-prefix normalization too. */
    private fun embeddingTensors(
        word: FloatArray,
        pos: FloatArray,
        type: FloatArray,
        gamma: FloatArray = FloatArray(h) { 1f },
        beta: FloatArray = FloatArray(h) { 0f },
    ): Map<String, Tensor<FP32, Float>> = mapOf(
        "embeddings.word_embeddings.weight" to tensor2d(vocab, word),
        "embeddings.position_embeddings.weight" to tensor2d(maxPos, pos),
        "embeddings.token_type_embeddings.weight" to tensor2d(typeVocab, type),
        "embeddings.LayerNorm.weight" to tensor1d(gamma),
        "embeddings.LayerNorm.bias" to tensor1d(beta),
    )

    private fun layerNormRow(x: FloatArray, gamma: FloatArray, beta: FloatArray): FloatArray {
        val mean = x.average().toFloat()
        val variance = x.map { (it - mean) * (it - mean) }.average().toFloat()
        val denom = sqrt(variance + eps.toFloat())
        return FloatArray(x.size) { i -> (x[i] - mean) / denom * gamma[i] + beta[i] }
    }

    @Test
    fun `embeddings block matches hand-computed word + position + type row0 with LayerNorm`() {
        val word = table(vocab, 0.2f)
        val pos = table(maxPos, 0.1f)
        val type = table(typeVocab, 0.05f)
        val gamma = floatArrayOf(1f, 2f, 0.5f, 1.5f)
        val beta = floatArrayOf(0.1f, -0.1f, 0f, 0.2f)

        val model = BertNetworkLoader.fromTensorMap<FP32, Float>(
            config(layers = 0),
            embeddingTensors(word, pos, type, gamma, beta),
        )

        val tokenIds = intArrayOf(3, 0, 7)
        val input = ctx.fromFloatArray<FP32, Float>(
            Shape(tokenIds.size), FP32::class,
            FloatArray(tokenIds.size) { tokenIds[it].toFloat() },
        )
        val out = model.forward(input, ctx)
        assertEquals(listOf(tokenIds.size, h), out.shape.dimensions.toList())

        tokenIds.forEachIndexed { position, tokenId ->
            val summed = FloatArray(h) { c ->
                word[tokenId * h + c] + pos[position * h + c] + type[0 * h + c]
            }
            val expected = layerNormRow(summed, gamma, beta)
            for (c in 0 until h) {
                val actual = out.data.get(position, c)
                assertTrue(
                    abs(actual - expected[c]) < 1e-4f,
                    "pos=$position c=$c expected=${expected[c]} actual=$actual",
                )
            }
        }
    }

    @Test
    fun `position rows depend on position not token id`() {
        // Position rows must point in per-row-distinct directions: every row of
        // table() is a scalar multiple of the same (c+1) vector, which LayerNorm
        // would normalize to identical rows regardless of position.
        val rotatedPos = FloatArray(maxPos * h) { idx ->
            val r = idx / h
            val c = idx % h
            0.5f * (((r + c) % h) + 1)
        }
        val model = BertNetworkLoader.fromTensorMap<FP32, Float>(
            config(layers = 0),
            embeddingTensors(table(vocab, 0.2f), rotatedPos, table(typeVocab, 0.05f)),
        )
        val input = ctx.fromFloatArray<FP32, Float>(Shape(3), FP32::class, floatArrayOf(5f, 5f, 5f))
        val out = model.forward(input, ctx)

        // Same token at positions 0 vs 2 must differ (position table contributes).
        var diff = 0f
        for (c in 0 until h) diff += abs(out.data.get(0, c) - out.data.get(2, c))
        assertTrue(diff > 1e-3f, "identical rows for same token at different positions: position embedding lost")
    }

    @Test
    fun `full 2-layer network maps all parameters from a bare-named sentence-transformers tensor set`() {
        val cfg = config(layers = 2)
        val tensors = syntheticFullCheckpoint(cfg, bare = true)
        // Must not throw (mapped == total); pooler/projection stay unused.
        val model = BertNetworkLoader.fromTensorMap<FP32, Float>(cfg, tensors)
        val input = ctx.fromFloatArray<FP32, Float>(Shape(2), FP32::class, floatArrayOf(1f, 2f))
        val out = model.forward(input, ctx)
        assertEquals(listOf(2, h), out.shape.dimensions.toList())
    }

    @Test
    fun `full 2-layer network maps all parameters from a bert-prefixed tensor set`() {
        val cfg = config(layers = 2)
        val model = BertNetworkLoader.fromTensorMap<FP32, Float>(cfg, syntheticFullCheckpoint(cfg, bare = false))
        val input = ctx.fromFloatArray<FP32, Float>(Shape(2), FP32::class, floatArrayOf(1f, 2f))
        assertEquals(listOf(2, h), model.forward(input, ctx).shape.dimensions.toList())
    }

    @Test
    fun `missing position table fails mapping loudly`() {
        val cfg = config(layers = 0)
        val tensors = embeddingTensors(table(vocab, 0.2f), table(maxPos, 0.1f), table(typeVocab, 0.05f))
            .filterKeys { !it.contains("position_embeddings") }
        assertFailsWith<IllegalArgumentException> {
            BertNetworkLoader.fromTensorMap<FP32, Float>(cfg, tensors)
        }
    }

    @Test
    fun `sequence longer than maxPositionEmbeddings is rejected`() {
        val model = BertNetworkLoader.fromTensorMap<FP32, Float>(
            config(layers = 0),
            embeddingTensors(table(vocab, 0.2f), table(maxPos, 0.1f), table(typeVocab, 0.05f)),
        )
        val tooLong = ctx.fromFloatArray<FP32, Float>(
            Shape(maxPos + 1), FP32::class, FloatArray(maxPos + 1) { (it % vocab).toFloat() },
        )
        assertFailsWith<IllegalArgumentException> { model.forward(tooLong, ctx) }
    }

    /** Full HF-shaped tensor set for [cfg], bare or bert.-prefixed, incl. unused pooler. */
    private fun syntheticFullCheckpoint(
        cfg: BertModelConfig,
        bare: Boolean,
    ): Map<String, Tensor<FP32, Float>> {
        fun identityLike(rows: Int, cols: Int, scale: Float): Tensor<FP32, Float> {
            val data = FloatArray(rows * cols) { idx ->
                val r = idx / cols
                val c = idx % cols
                if (r % cols == c) scale else 0f
            }
            return ctx.fromFloatArray(Shape(rows, cols), FP32::class, data)
        }

        fun vec(size: Int, v: Float) = ctx.fromFloatArray<FP32, Float>(Shape(size), FP32::class, FloatArray(size) { v })

        val m = mutableMapOf<String, Tensor<FP32, Float>>()
        val p = if (bare) "" else "bert."
        m["${p}embeddings.word_embeddings.weight"] = tensor2d(vocab, table(vocab, 0.2f))
        m["${p}embeddings.position_embeddings.weight"] = tensor2d(maxPos, table(maxPos, 0.1f))
        m["${p}embeddings.token_type_embeddings.weight"] = tensor2d(typeVocab, table(typeVocab, 0.05f))
        m["${p}embeddings.LayerNorm.weight"] = vec(h, 1f)
        m["${p}embeddings.LayerNorm.bias"] = vec(h, 0f)
        for (l in 0 until cfg.numHiddenLayers) {
            val lp = "${p}encoder.layer.$l"
            m["$lp.attention.self.query.weight"] = identityLike(h, h, 0.5f)
            m["$lp.attention.self.query.bias"] = vec(h, 0f)
            m["$lp.attention.self.key.weight"] = identityLike(h, h, 0.5f)
            m["$lp.attention.self.key.bias"] = vec(h, 0f)
            m["$lp.attention.self.value.weight"] = identityLike(h, h, 0.5f)
            m["$lp.attention.self.value.bias"] = vec(h, 0f)
            m["$lp.attention.output.dense.weight"] = identityLike(h, h, 0.5f)
            m["$lp.attention.output.dense.bias"] = vec(h, 0f)
            m["$lp.attention.output.LayerNorm.weight"] = vec(h, 1f)
            m["$lp.attention.output.LayerNorm.bias"] = vec(h, 0f)
            m["$lp.intermediate.dense.weight"] = identityLike(cfg.intermediateSize, h, 0.5f)
            m["$lp.intermediate.dense.bias"] = vec(cfg.intermediateSize, 0f)
            m["$lp.output.dense.weight"] = identityLike(h, cfg.intermediateSize, 0.5f)
            m["$lp.output.dense.bias"] = vec(h, 0f)
            m["$lp.output.LayerNorm.weight"] = vec(h, 1f)
            m["$lp.output.LayerNorm.bias"] = vec(h, 0f)
        }
        // Present in real checkpoints, must land in unusedTensors without failing.
        m["${p}pooler.dense.weight"] = identityLike(h, h, 1f)
        m["${p}pooler.dense.bias"] = vec(h, 0f)
        return m
    }
}
