package sk.ainet.models.laya

import sk.ainet.context.ExecutionContext
import sk.ainet.io.weights.MappingConfig
import sk.ainet.io.weights.WeightMapper
import sk.ainet.io.weights.WeightTensor
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.transformer.PaddedBatch
import sk.ainet.lang.nn.transformer.withPaddedBatch
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.Int32
import kotlin.reflect.KClass

/**
 * The reference `DecisionModel.forward` (REQ-EXE-006): encoder → add `type_emb[qtype]` to every
 * position → head blocks → gather the marker rows → scorer → one logit per option. [logits] runs one
 * unpadded sequence; [logitsBatch] runs several as one right-padded batch whose lengths the attention
 * op takes as its key-padding rule (REQ-EXE-005).
 */
public class LayaDecisionModel<T : DType>(
    public val encoder: Module<T, Float>,
    public val head: Module<T, Float>,
    public val scorer: Module<T, Float>,
    public val typeEmbedding: Tensor<T, Float>,
    public val hiddenSize: Int,
    private val ctx: ExecutionContext,
    private val dtype: KClass<T>,
) {
    /** Encoder output `[L, hidden]` for the token ids. */
    public fun encode(ids: IntArray): Tensor<T, Float> {
        require(ids.isNotEmpty()) { "a sequence must hold at least one token" }
        val tokens = ctx.fromFloatArray<T, Float>(Shape(ids.size), dtype, FloatArray(ids.size) { ids[it].toFloat() })
        return encoder.forward(tokens, ctx)
    }

    /** Hidden states after the type embedding and the head blocks, `[L, hidden]`. */
    public fun hidden(ids: IntArray, type: LayaQuestionType): Tensor<T, Float> {
        val encoded = encode(ids)
        val typeRow = ctx.ops.reshape(ctx.ops.narrow(typeEmbedding, dim = 0, start = type.index, length = 1), Shape(hiddenSize))
        return head.forward(ctx.ops.add(encoded, typeRow), ctx)
    }

    /** Raw (untempered) logits, one per marker, in slot order. */
    public fun logits(ids: IntArray, markers: IntArray, type: LayaQuestionType): FloatArray {
        require(markers.isNotEmpty()) { "a question needs at least one marker" }
        val h = hidden(ids, type).data.copyToFloatArray()
        val rows = FloatArray(markers.size * hiddenSize)
        for ((slot, position) in markers.withIndex()) {
            require(position in 0 until ids.size) { "marker $position outside the sequence of ${ids.size} tokens" }
            h.copyInto(rows, destinationOffset = slot * hiddenSize, startIndex = position * hiddenSize, endIndex = (position + 1) * hiddenSize)
        }
        val gathered = ctx.fromFloatArray<T, Float>(Shape(markers.size, hiddenSize), dtype, rows)
        return scorer.forward(gathered, ctx).data.copyToFloatArray()
    }

    public fun logits(sequence: LayaSequence, type: LayaQuestionType): FloatArray = logits(sequence.ids, sequence.markers, type)

    /**
     * Encoder output for a flattened padded batch, `[batchSize * seqLen, hidden]` (see [PaddedBatch]):
     * [ids] holds [PaddedBatch.rows] token ids, sequence `b` at rows `b * seqLen until (b + 1) * seqLen`.
     * Padded keys are excluded from attention through the batch's lengths; padded rows of the output
     * are finite but meaningless.
     */
    public fun encodeBatch(ids: IntArray, batch: PaddedBatch): Tensor<T, Float> {
        require(ids.size == batch.rows) { "a padded batch of ${batch.batchSize}x${batch.seqLen} needs ${batch.rows} token ids, got ${ids.size}" }
        val tokens = ctx.fromFloatArray<T, Float>(Shape(ids.size), dtype, FloatArray(ids.size) { ids[it].toFloat() })
        return withPaddedBatch(encoder, batch) { encoder.forward(tokens, ctx) }
    }

    /** [hidden] for a padded batch with one question type per sequence, `[batchSize * seqLen, hidden]`. */
    public fun hiddenBatch(ids: IntArray, batch: PaddedBatch, types: List<LayaQuestionType>): Tensor<T, Float> {
        require(types.size == batch.batchSize) { "${types.size} question types for a batch of ${batch.batchSize}" }
        val encoded = encodeBatch(ids, batch)
        // type_emb[type_b] on every row of sequence b: a gather over the type table with one index per row.
        @Suppress("UNCHECKED_CAST")
        val typeIndex = ctx.fromIntArray<Int32, Int>(Shape(batch.rows), Int32::class, IntArray(batch.rows) { types[it / batch.seqLen].index }) as Tensor<DType, *>
        val typeRows = ctx.ops.gather(typeEmbedding, typeIndex, dim = 0)
        return withPaddedBatch(head, batch) { head.forward(ctx.ops.add(encoded, typeRows), ctx) }
    }

    /**
     * [logits] for several sequences in one padded encoder pass (REQ-EXE-005): element `i` holds the
     * logits of `sequences[i]`, one per marker in slot order. Sequences are right-padded with [padId]
     * to the longest; the valid rows equal the single-sequence pass, since padded keys never receive
     * attention weight and every other module is row-wise.
     */
    public fun logitsBatch(sequences: List<LayaSequence>, types: List<LayaQuestionType>, padId: Int): List<FloatArray> {
        require(sequences.isNotEmpty()) { "a batch needs at least one sequence" }
        require(types.size == sequences.size) { "${types.size} question types for ${sequences.size} sequences" }
        for ((i, s) in sequences.withIndex()) {
            require(s.markers.isNotEmpty()) { "sequence $i needs at least one marker" }
            for (position in s.markers) require(position in 0 until s.length) { "marker $position outside sequence $i of ${s.length} tokens" }
        }
        val seqLen = sequences.maxOf { it.length }
        val b = sequences.size
        val ids = IntArray(b * seqLen) { padId }
        for ((i, s) in sequences.withIndex()) s.ids.copyInto(ids, destinationOffset = i * seqLen)
        val batch = PaddedBatch.of(ctx, seqLen, IntArray(b) { sequences[it].length })

        val h = hiddenBatch(ids, batch, types).data.copyToFloatArray()
        val total = sequences.sumOf { it.markers.size }
        val rows = FloatArray(total * hiddenSize)
        var slot = 0
        for ((i, s) in sequences.withIndex()) {
            for (position in s.markers) {
                val row = i * seqLen + position
                h.copyInto(rows, destinationOffset = slot * hiddenSize, startIndex = row * hiddenSize, endIndex = (row + 1) * hiddenSize)
                slot++
            }
        }
        val gathered = ctx.fromFloatArray<T, Float>(Shape(total, hiddenSize), dtype, rows)
        val scored = scorer.forward(gathered, ctx).data.copyToFloatArray()
        var offset = 0
        return sequences.map { s -> scored.copyOfRange(offset, offset + s.markers.size).also { offset += s.markers.size } }
    }

    public companion object {
        /**
         * Builds the module tree for the configs and binds [prepared] weights through
         * [LayaSafeTensorsNameResolver]; every parameter must be mapped and every prepared tensor used.
         */
        public fun <T : DType> build(
            encoderConfig: LayaEncoderConfig,
            agentConfig: LayaAgentConfig,
            prepared: LayaPreparedWeights<T, Float>,
            ctx: ExecutionContext,
            dtype: KClass<T>,
            exactGelu: Boolean = true,
        ): LayaDecisionModel<T> {
            val encoder = LayaNetwork.encoder<T, Float>(encoderConfig, maxSeqLen = maxOf(agentConfig.maxLen, 16), dtype, exactGelu)
            val head = LayaNetwork.head<T, Float>(encoderConfig.hiddenSize, agentConfig.headLayers, dtype)
            val scorer = LayaNetwork.scorer<T, Float>(encoderConfig.hiddenSize, dtype, exactGelu)
            bind(encoder, prepared.encoder, "encoder")
            bind(head, prepared.head, "head")
            bind(scorer, prepared.scorer, "scorer")
            return LayaDecisionModel(encoder, head, scorer, prepared.extras.typeEmbedding, encoderConfig.hiddenSize, ctx, dtype)
        }

        private fun <T : DType> bind(module: Module<T, Float>, tensors: List<WeightTensor<T, Float>>, what: String) {
            val result = WeightMapper.applyWeights(
                module, tensors,
                MappingConfig(usePathBasedMatching = false, fallbackToShapeMatching = false, nameResolver = LayaSafeTensorsNameResolver()),
            )
            if (result.mapped != result.total || result.unusedTensors.isNotEmpty()) {
                throw UnsupportedLayaCheckpointException(
                    buildString {
                        append("$what: mapped ${result.mapped}/${result.total} parameters")
                        if (result.missingParams.isNotEmpty()) append("; missing: ${result.missingParams.take(8)}")
                        if (result.unusedTensors.isNotEmpty()) append("; unused: ${result.unusedTensors.take(8)}")
                    },
                )
            }
        }
    }
}
