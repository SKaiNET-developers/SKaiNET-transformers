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
 * Everything after the encoder: the question-type embedding, the head blocks and the scorer. The
 * base checkpoint's head is [LayaDecisionModel.defaultHead]; an expert head
 * ([LayaExpert]) is another instance bound from `expert.safetensors` and run on the same encoder
 * output through [LayaDecisionModel.logitsFrom].
 */
public class LayaDecisionHead<T : DType>(
    public val head: Module<T, Float>,
    public val scorer: Module<T, Float>,
    /** `type_emb.weight`, `[3, hidden]`. */
    public val typeEmbedding: Tensor<T, Float>,
) {
    init {
        val dims = typeEmbedding.shape.dimensions
        require(dims.size == 2 && dims[0] == LayaQuestionType.entries.size) { "type embedding must be [3, hidden], got ${dims.toList()}" }
    }

    public val hiddenSize: Int get() = typeEmbedding.shape.dimensions[1]
}

/** The encoder output of one right-padded batch of sequences (see [LayaDecisionModel.encodeBatch]). */
public class LayaEncodedBatch<T : DType>(
    public val batch: PaddedBatch,
    /** Rows per sequence: the longest sequence of the batch. */
    public val seqLen: Int,
    /** `[batchSize * seqLen, hidden]`, sequence `b` at rows `b * seqLen until (b + 1) * seqLen`. */
    public val encoded: Tensor<T, Float>,
)

/**
 * The reference `DecisionModel.forward` (REQ-EXE-006): encoder → add `type_emb[qtype]` to every
 * position → head blocks → gather the marker rows → scorer → one logit per option. [logits] runs one
 * unpadded sequence; [logitsBatch] runs several as one right-padded batch whose lengths the attention
 * op takes as its key-padding rule (REQ-EXE-005).
 *
 * The encoder output does not depend on the head, so [logitsFrom] / [logitsBatchFrom] take an
 * encoder output computed once and run any [LayaDecisionHead] on it — the base head by default, an expert
 * head for a routed request, both when the two are compared.
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
    /** The checkpoint's own head (`general` in a mixture of experts). */
    public val defaultHead: LayaDecisionHead<T> = LayaDecisionHead(head, scorer, typeEmbedding)

    /** Encoder output `[L, hidden]` for the token ids. */
    public fun encode(ids: IntArray): Tensor<T, Float> {
        require(ids.isNotEmpty()) { "a sequence must hold at least one token" }
        val tokens = ctx.fromFloatArray<T, Float>(Shape(ids.size), dtype, FloatArray(ids.size) { ids[it].toFloat() })
        return encoder.forward(tokens, ctx)
    }

    /** Hidden states after the type embedding and the head blocks, `[L, hidden]`. */
    public fun hidden(ids: IntArray, type: LayaQuestionType): Tensor<T, Float> = hiddenFrom(encode(ids), type, defaultHead)

    /** [hidden] of [head] on an encoder output `[L, hidden]` from [encode]. */
    public fun hiddenFrom(encoded: Tensor<T, Float>, type: LayaQuestionType, head: LayaDecisionHead<T> = defaultHead): Tensor<T, Float> {
        checkHead(head)
        val typeRow = ctx.ops.reshape(ctx.ops.narrow(head.typeEmbedding, dim = 0, start = type.index, length = 1), Shape(hiddenSize))
        return head.head.forward(ctx.ops.add(encoded, typeRow), ctx)
    }

    /** Raw (untempered) logits, one per marker, in slot order. */
    public fun logits(ids: IntArray, markers: IntArray, type: LayaQuestionType): FloatArray =
        logitsFrom(encode(ids), ids.size, markers, type, defaultHead)

    public fun logits(sequence: LayaSequence, type: LayaQuestionType): FloatArray = logits(sequence.ids, sequence.markers, type)

    /**
     * [logits] of [head] on an encoder output from [encode] for a sequence of [length] tokens: the
     * head blocks, the marker rows gathered, the scorer.
     */
    public fun logitsFrom(encoded: Tensor<T, Float>, length: Int, markers: IntArray, type: LayaQuestionType, head: LayaDecisionHead<T> = defaultHead): FloatArray {
        require(markers.isNotEmpty()) { "a question needs at least one marker" }
        val h = hiddenFrom(encoded, type, head).data.copyToFloatArray()
        val rows = FloatArray(markers.size * hiddenSize)
        for ((slot, position) in markers.withIndex()) {
            require(position in 0 until length) { "marker $position outside the sequence of $length tokens" }
            h.copyInto(rows, destinationOffset = slot * hiddenSize, startIndex = position * hiddenSize, endIndex = (position + 1) * hiddenSize)
        }
        val gathered = ctx.fromFloatArray<T, Float>(Shape(markers.size, hiddenSize), dtype, rows)
        return head.scorer.forward(gathered, ctx).data.copyToFloatArray()
    }

    public fun logitsFrom(encoded: Tensor<T, Float>, sequence: LayaSequence, type: LayaQuestionType, head: LayaDecisionHead<T> = defaultHead): FloatArray =
        logitsFrom(encoded, sequence.length, sequence.markers, type, head)

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

    /**
     * [encodeBatch] for prepared sequences: right-pads them with [padId] to the longest and runs the
     * encoder once. The result feeds [logitsBatchFrom] for any head.
     */
    public fun encodeBatch(sequences: List<LayaSequence>, padId: Int): LayaEncodedBatch<T> {
        require(sequences.isNotEmpty()) { "a batch needs at least one sequence" }
        for ((i, s) in sequences.withIndex()) {
            require(s.markers.isNotEmpty()) { "sequence $i needs at least one marker" }
            for (position in s.markers) require(position in 0 until s.length) { "marker $position outside sequence $i of ${s.length} tokens" }
        }
        val seqLen = sequences.maxOf { it.length }
        val b = sequences.size
        val ids = IntArray(b * seqLen) { padId }
        for ((i, s) in sequences.withIndex()) s.ids.copyInto(ids, destinationOffset = i * seqLen)
        val batch = PaddedBatch.of(ctx, seqLen, IntArray(b) { sequences[it].length })
        return LayaEncodedBatch(batch, seqLen, encodeBatch(ids, batch))
    }

    /** [hidden] for a padded batch with one question type per sequence, `[batchSize * seqLen, hidden]`. */
    public fun hiddenBatch(ids: IntArray, batch: PaddedBatch, types: List<LayaQuestionType>): Tensor<T, Float> =
        hiddenBatchFrom(LayaEncodedBatch(batch, batch.seqLen, encodeBatch(ids, batch)), types, defaultHead)

    /** [hiddenBatch] of [head] on an encoded batch. */
    public fun hiddenBatchFrom(encoded: LayaEncodedBatch<T>, types: List<LayaQuestionType>, head: LayaDecisionHead<T> = defaultHead): Tensor<T, Float> {
        checkHead(head)
        val batch = encoded.batch
        require(types.size == batch.batchSize) { "${types.size} question types for a batch of ${batch.batchSize}" }
        // type_emb[type_b] on every row of sequence b: a gather over the type table with one index per row.
        @Suppress("UNCHECKED_CAST")
        val typeIndex = ctx.fromIntArray<Int32, Int>(Shape(batch.rows), Int32::class, IntArray(batch.rows) { types[it / encoded.seqLen].index }) as Tensor<DType, *>
        val typeRows = ctx.ops.gather(head.typeEmbedding, typeIndex, dim = 0)
        return withPaddedBatch(head.head, batch) { head.head.forward(ctx.ops.add(encoded.encoded, typeRows), ctx) }
    }

    /**
     * [logits] for several sequences in one padded encoder pass (REQ-EXE-005): element `i` holds the
     * logits of `sequences[i]`, one per marker in slot order. Sequences are right-padded with [padId]
     * to the longest; the valid rows equal the single-sequence pass, since padded keys never receive
     * attention weight and every other module is row-wise.
     */
    public fun logitsBatch(sequences: List<LayaSequence>, types: List<LayaQuestionType>, padId: Int): List<FloatArray> =
        logitsBatchFrom(sequences, encodeBatch(sequences, padId), types, defaultHead)

    /** [logitsBatch] of [head] on the encoded batch of the same [sequences]. */
    public fun logitsBatchFrom(sequences: List<LayaSequence>, encoded: LayaEncodedBatch<T>, types: List<LayaQuestionType>, head: LayaDecisionHead<T> = defaultHead): List<FloatArray> {
        require(sequences.size == encoded.batch.batchSize) { "${sequences.size} sequences for an encoded batch of ${encoded.batch.batchSize}" }
        require(types.size == sequences.size) { "${types.size} question types for ${sequences.size} sequences" }
        val seqLen = encoded.seqLen
        val h = hiddenBatchFrom(encoded, types, head).data.copyToFloatArray()
        val total = sequences.sumOf { it.markers.size }
        val rows = FloatArray(total * hiddenSize)
        var slot = 0
        for ((i, s) in sequences.withIndex()) {
            for (position in s.markers) {
                require(position in 0 until s.length) { "marker $position outside sequence $i of ${s.length} tokens" }
                val row = i * seqLen + position
                h.copyInto(rows, destinationOffset = slot * hiddenSize, startIndex = row * hiddenSize, endIndex = (row + 1) * hiddenSize)
                slot++
            }
        }
        val gathered = ctx.fromFloatArray<T, Float>(Shape(total, hiddenSize), dtype, rows)
        val scored = head.scorer.forward(gathered, ctx).data.copyToFloatArray()
        var offset = 0
        return sequences.map { s -> scored.copyOfRange(offset, offset + s.markers.size).also { offset += s.markers.size } }
    }

    private fun checkHead(head: LayaDecisionHead<T>) {
        require(head.hiddenSize == hiddenSize) { "head of hidden size ${head.hiddenSize} on a model of hidden size $hiddenSize" }
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

        /**
         * Builds and binds one decision head from a head-only checkpoint ([LayaWeightPreparation.prepareHead]).
         * [hiddenSize] and [headLayers] are the base model's; the result runs on that model's encoder
         * output through [logitsFrom].
         */
        public fun <T : DType> buildHead(
            prepared: LayaPreparedHead<T, Float>,
            hiddenSize: Int,
            headLayers: Int,
            dtype: KClass<T>,
            exactGelu: Boolean = true,
        ): LayaDecisionHead<T> {
            val head = LayaNetwork.head<T, Float>(hiddenSize, headLayers, dtype)
            val scorer = LayaNetwork.scorer<T, Float>(hiddenSize, dtype, exactGelu)
            bind(head, prepared.head, "head")
            bind(scorer, prepared.scorer, "scorer")
            return LayaDecisionHead(head, scorer, prepared.extras.typeEmbedding)
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
