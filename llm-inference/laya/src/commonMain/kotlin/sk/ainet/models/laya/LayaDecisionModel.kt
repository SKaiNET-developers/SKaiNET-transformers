package sk.ainet.models.laya

import sk.ainet.context.ExecutionContext
import sk.ainet.io.weights.MappingConfig
import sk.ainet.io.weights.WeightMapper
import sk.ainet.io.weights.WeightTensor
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * The reference `DecisionModel.forward` for one unpadded sequence (REQ-EXE-006):
 * encoder → add `type_emb[qtype]` to every position → head blocks → gather the marker rows →
 * scorer → one logit per option. Batched sequences with a padding mask are deferred (REQ-EXE-005).
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
