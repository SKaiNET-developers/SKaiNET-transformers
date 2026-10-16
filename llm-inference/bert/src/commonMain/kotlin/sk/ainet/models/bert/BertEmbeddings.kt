package sk.ainet.models.bert

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.layers.Embedding
import sk.ainet.lang.nn.layers.EmbeddingAdapter
import sk.ainet.lang.nn.normalization.LayerNormalization
import sk.ainet.lang.nn.transformer.PaddedBatch
import sk.ainet.lang.nn.transformer.PaddedBatchAware
import sk.ainet.lang.nn.transformer.withPaddedBatch
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Complete BERT embeddings block: word + absolute position + token-type
 * embeddings, summed and layer-normalized.
 *
 * `hidden = LayerNorm(wordEmb[ids] + posEmb[0..L-1] + typeEmb[0])`
 *
 * The three lookups are expressed with graph-safe ops only:
 *  - word embeddings via [Embedding]'s `ops.gather` (the input tensor carries
 *    the token ids),
 *  - position embeddings via `ops.narrow(table, 0, 0, L)` — rows `0..L-1` of
 *    the position table *are* the absolute-position vectors, so no index
 *    tensor is needed,
 *  - token-type embeddings as row 0 of the type table reshaped to `[dim]` and
 *    broadcast-added (the same `[L, dim] + [dim]` broadcast Linear uses for
 *    bias).
 *
 * Keeping position/type lookups index-free means a traced forward has exactly
 * one non-parameter leaf (the token-id tensor), which makes compiled-mode
 * input detection unambiguous and the exported graph a clean
 * `tokens → hidden-states` encoder. A flattened padded batch ([PaddedBatch],
 * set by [withPaddedBatch]) adds the attention lengths as the only other leaf:
 * positions restart at 0 for every sequence through a `[B, S, dim] + [S, dim]`
 * broadcast of the same narrowed table, still without an index tensor.
 *
 * Single-segment only: all positions use token-type row 0, which matches every
 * sentence-embedding caller (they pass all-zero segment ids). Two-segment
 * (cross-encoder) inputs are not supported by the DSL network.
 *
 * Weight-mapping paths (resolved by `BertSafeTensorsNameResolver`):
 *  - `…/embeddings/word_embeddings` → `bert.embeddings.word_embeddings.weight`
 *  - `…/embeddings` + `position_embeddings.weight` → `bert.embeddings.position_embeddings.weight`
 *  - `…/embeddings` + `token_type_embeddings.weight` → `bert.embeddings.token_type_embeddings.weight`
 *  - `…/embeddings/LayerNorm` → `bert.embeddings.LayerNorm.{weight,bias}`
 */
public class BertEmbeddings<T : DType, V>(
    config: BertModelConfig,
    dtype: KClass<T>,
    override val name: String = "embeddings",
) : Module<T, V>(), ModuleParameters<T, V>, PaddedBatchAware {

    private val dim = config.hiddenSize

    /** The padded-batch layout of the current forward, set by [withPaddedBatch]; `null` for one sequence. */
    override var paddedBatch: PaddedBatch? = null

    private val wordEmbeddings: EmbeddingAdapter<T, V> = EmbeddingAdapter(
        Embedding(
            numEmbeddings = config.vocabSize,
            embeddingDim = dim,
            initWeight = voidTensor(dtype, config.vocabSize, dim),
            name = "word_embeddings",
        )
    )

    private val layerNorm: LayerNormalization<T, V> = LayerNormalization(
        normalizedShape = intArrayOf(dim),
        eps = config.layerNormEps,
        name = "LayerNorm",
        dtype = dtype,
    )

    override val params: List<ModuleParameter<T, V>> = listOf(
        ModuleParameter.WeightParameter(
            "position_embeddings.weight",
            voidTensor(dtype, config.maxPositionEmbeddings, dim),
        ),
        ModuleParameter.WeightParameter(
            "token_type_embeddings.weight",
            voidTensor(dtype, config.typeVocabSize, dim),
        ),
    )

    override val modules: List<Module<T, V>> = listOf(wordEmbeddings, layerNorm)

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        val word = wordEmbeddings.forward(input, ctx) // gather → [L, dim]
        val batch = paddedBatch
        val seqLen = batch?.seqLen ?: word.shape[0]
        if (batch != null) {
            require(word.shape[0] == batch.rows) {
                "BertEmbeddings($name): a padded batch of ${batch.batchSize}x${batch.seqLen} expects ${batch.rows} token ids, got ${word.shape[0]}"
            }
        }

        val positionTable = params[0].value
        require(seqLen <= positionTable.shape[0]) {
            "BertEmbeddings($name): sequence length $seqLen exceeds maxPositionEmbeddings ${positionTable.shape[0]}"
        }
        // Rows 0..L-1 of the position table are exactly the absolute-position
        // embeddings for positions 0..L-1 — same result as gather([0..L-1]).
        val pos = ctx.ops.narrow(positionTable, dim = 0, start = 0, length = seqLen) // [L, dim]

        // Segment 0 for every position: row 0 of the type table as a [dim]
        // vector, broadcast over the sequence like a Linear bias.
        val typeRow = ctx.ops.narrow(params[1].value, dim = 0, start = 0, length = 1) // [1, dim]
        val type = ctx.ops.reshape(typeRow, Shape(dim)) // [dim]

        val positioned = if (batch == null) {
            ctx.ops.add(word, pos)
        } else {
            // Every sequence of the flattened batch starts at position 0: [B, S, dim] + [S, dim].
            val rows3 = ctx.ops.reshape(word, Shape(batch.batchSize, batch.seqLen, dim))
            ctx.ops.reshape(ctx.ops.add(rows3, pos), Shape(batch.rows, dim))
        }
        val summed = ctx.ops.add(positioned, type)
        return layerNorm.forward(summed, ctx)
    }

    private companion object {
        @Suppress("UNCHECKED_CAST")
        private fun <T : DType, V> voidTensor(dtype: KClass<T>, rows: Int, cols: Int): Tensor<T, V> =
            VoidOpsTensor(
                object : TensorData<T, V> {
                    override val shape = Shape(rows, cols)
                    override fun get(vararg indices: Int): V = 0.0f as V
                    override fun set(vararg indices: Int, value: V) {}
                },
                dtype,
            )
    }
}
