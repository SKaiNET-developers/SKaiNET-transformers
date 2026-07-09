package sk.ainet.models.bert

import sk.ainet.context.ExecutionContext
import sk.ainet.io.weights.WeightTensor
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

/**
 * Deterministic synthetic BERT checkpoints for tests: varied (non-degenerate
 * under LayerNorm) embeddings and non-trivial but well-conditioned projections.
 */
internal object BertSyntheticFixtures {

    fun tinyConfig(
        h: Int = 4,
        layers: Int = 2,
        projDim: Int? = null,
    ) = BertModelConfig(
        vocabSize = 10,
        hiddenSize = h,
        numHiddenLayers = layers,
        numAttentionHeads = 2,
        intermediateSize = h * 2,
        maxPositionEmbeddings = 16,
        typeVocabSize = 2,
        layerNormEps = 1e-5,
        projectionDim = projDim,
    )

    fun weightTensors(
        ctx: ExecutionContext,
        cfg: BertModelConfig,
        withProjection: Boolean = false,
    ): List<WeightTensor<FP32, Float>> {
        val h = cfg.hiddenSize

        fun varied(rows: Int, cols: Int, scale: Float): Tensor<FP32, Float> {
            val data = FloatArray(rows * cols) { idx ->
                val r = idx / cols
                val c = idx % cols
                scale * (r + 1) * (((r + c) % cols) + 1)
            }
            return ctx.fromFloatArray(Shape(rows, cols), FP32::class, data)
        }

        fun mixed(rows: Int, cols: Int, scale: Float): Tensor<FP32, Float> {
            // Diagonal-dominant with small off-diagonal texture: keeps values
            // bounded through 2 layers but exercises real mixing.
            val data = FloatArray(rows * cols) { idx ->
                val r = idx / cols
                val c = idx % cols
                val diag = if (r % cols == c) scale else 0f
                diag + 0.05f * (((r * 3 + c * 7) % 5) - 2)
            }
            return ctx.fromFloatArray(Shape(rows, cols), FP32::class, data)
        }

        fun vec(size: Int, v: Float): Tensor<FP32, Float> =
            ctx.fromFloatArray(Shape(size), FP32::class, FloatArray(size) { v })

        val out = mutableListOf<WeightTensor<FP32, Float>>()
        fun add(name: String, t: Tensor<FP32, Float>) {
            out += WeightTensor(name, t.shape.dimensions.toList(), t)
        }

        add("embeddings.word_embeddings.weight", varied(cfg.vocabSize, h, 0.2f))
        add("embeddings.position_embeddings.weight", varied(cfg.maxPositionEmbeddings, h, 0.1f))
        add("embeddings.token_type_embeddings.weight", varied(cfg.typeVocabSize, h, 0.05f))
        add("embeddings.LayerNorm.weight", vec(h, 1f))
        add("embeddings.LayerNorm.bias", vec(h, 0f))
        for (l in 0 until cfg.numHiddenLayers) {
            val p = "encoder.layer.$l"
            add("$p.attention.self.query.weight", mixed(h, h, 0.5f))
            add("$p.attention.self.query.bias", vec(h, 0.01f))
            add("$p.attention.self.key.weight", mixed(h, h, 0.5f))
            add("$p.attention.self.key.bias", vec(h, 0f))
            add("$p.attention.self.value.weight", mixed(h, h, 0.5f))
            add("$p.attention.self.value.bias", vec(h, 0f))
            add("$p.attention.output.dense.weight", mixed(h, h, 0.5f))
            add("$p.attention.output.dense.bias", vec(h, 0f))
            add("$p.attention.output.LayerNorm.weight", vec(h, 1f))
            add("$p.attention.output.LayerNorm.bias", vec(h, 0f))
            add("$p.intermediate.dense.weight", mixed(cfg.intermediateSize, h, 0.5f))
            add("$p.intermediate.dense.bias", vec(cfg.intermediateSize, 0f))
            add("$p.output.dense.weight", mixed(h, cfg.intermediateSize, 0.5f))
            add("$p.output.dense.bias", vec(h, 0f))
            add("$p.output.LayerNorm.weight", vec(h, 1f))
            add("$p.output.LayerNorm.bias", vec(h, 0f))
        }
        if (withProjection) {
            val projDim = requireNotNull(cfg.projectionDim)
            add(BertNetworkLoader.PROJECTION_WEIGHT, mixed(projDim, h, 1f))
        }
        return BertNetworkLoader.normalizeTensorNames(out)
    }
}
