package sk.ainet.models.laya

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * The encoder output of several prepared sequences, as [LayaAgent.encodeFor] computed it: the
 * passes the batch plan ran (unpadded singles and padded batches). Hand it to [LayaAgent.logitsFrom]
 * once per head that should answer the same sequences — the encoder does not run again.
 */
public class LayaEncodedSequences<T : DType> internal constructor(
    public val sequences: List<LayaSequence>,
    internal val passes: List<Pass<T>>,
) {
    internal sealed class Pass<T : DType> {
        class Single<T : DType>(val index: Int, val encoded: Tensor<T, Float>) : Pass<T>()
        class Batch<T : DType>(val indices: IntArray, val encoded: LayaEncodedBatch<T>) : Pass<T>()
    }

    /** Tokens the encoder read, padding excluded. */
    public val inputTokens: Int get() = sequences.sumOf { it.length }
}

/**
 * Preprocessor, model and decoder of one checkpoint: ask typed questions about a state.
 *
 * @param maxBatchTokens token budget of one encoder pass in [decideAll] and [logitsFor]: consecutive
 *   sequences share a pass while `batchSize * longest <= maxBatchTokens`. Attention scores grow with
 *   `batch * heads * longest²`, so the budget bounds the transient memory of a request.
 */
public class LayaAgent<T : DType>(
    public val preprocessor: LayaPreprocessor,
    public val model: LayaDecisionModel<T>,
    public val decoder: LayaDecoder = LayaDecoder(preprocessor.agentConfig),
    public val maxBatchTokens: Int = DEFAULT_MAX_BATCH_TOKENS,
) {
    init {
        require(maxBatchTokens > 0) { "maxBatchTokens must be positive, got $maxBatchTokens" }
    }

    public class Result(public val answer: LayaAnswer, public val sequence: LayaSequence, public val logits: FloatArray)

    /** One question in one unpadded encoder pass. */
    public fun decide(state: LayaState, question: LayaQuestion, optionOrder: IntArray? = null, stateIds: IntArray? = null): Result =
        decide(state, question, optionOrder, stateIds, model.defaultHead, decoder)

    /** [decide] with another head and its decoder — an expert's ([LayaExpert]). */
    public fun decide(state: LayaState, question: LayaQuestion, optionOrder: IntArray?, stateIds: IntArray?, head: LayaDecisionHead<T>, decoder: LayaDecoder): Result {
        val sequence = preprocessor.prepare(state, question, optionOrder, stateIds)
        val logits = model.logitsFrom(model.encode(sequence.ids), sequence, question.type, head)
        return Result(decoder.decode(question, logits, sequence.optionOrder), sequence, logits)
    }

    /**
     * Every question about the same state: the state is tokenised once and the questions run as
     * padded batches under [maxBatchTokens] instead of one encoder pass each.
     */
    public fun decideAll(state: LayaState, questions: Map<String, LayaQuestion>): Map<String, Result> =
        decideAll(state, questions, model.defaultHead, decoder)

    /** [decideAll] with another head and its decoder — an expert's ([LayaExpert]). */
    public fun decideAll(state: LayaState, questions: Map<String, LayaQuestion>, head: LayaDecisionHead<T>, decoder: LayaDecoder): Map<String, Result> {
        if (questions.isEmpty()) return emptyMap()
        val stateIds = preprocessor.builder.encodeState(state)
        val keys = questions.keys.toList()
        val asked = keys.map { questions.getValue(it) }
        val sequences = asked.map { preprocessor.prepare(state, it, stateIds = stateIds) }
        val logits = logitsFor(sequences, asked.map { it.type }, head)
        return decodeAll(keys, asked, sequences, logits, decoder)
    }

    /** Decodes slot-ordered [logits] of [sequences] with [decoder], keyed like [keys]. */
    public fun decodeAll(keys: List<String>, questions: List<LayaQuestion>, sequences: List<LayaSequence>, logits: List<FloatArray>, decoder: LayaDecoder = this.decoder): Map<String, Result> {
        require(keys.size == questions.size && keys.size == sequences.size && keys.size == logits.size) { "keys, questions, sequences and logits must align" }
        val results = LinkedHashMap<String, Result>(keys.size)
        for (i in keys.indices) {
            results[keys[i]] = Result(decoder.decode(questions[i], logits[i], sequences[i].optionOrder), sequences[i], logits[i])
        }
        return results
    }

    /**
     * Logits for prepared [sequences], in order, computed in padded batches of similar length: the
     * sequences are sorted by length and a batch grows while `size * longest <= maxBatchTokens` and
     * the padding stays under [MAX_PADDING_FRACTION] of the real tokens. On a compute-bound eager
     * path padding is wasted work, so a long question never drags short ones up to its length. A
     * sequence that ends up alone takes the unpadded single pass, which keeps the fused attention
     * kernels.
     */
    public fun logitsFor(sequences: List<LayaSequence>, types: List<LayaQuestionType>): List<FloatArray> =
        logitsFor(sequences, types, model.defaultHead)

    /** [logitsFor] with another head. */
    public fun logitsFor(sequences: List<LayaSequence>, types: List<LayaQuestionType>, head: LayaDecisionHead<T>): List<FloatArray> =
        logitsFrom(encodeFor(sequences), types, head)

    /**
     * The encoder passes of [logitsFor] without a head: run once, then [logitsFrom] answers the same
     * sequences with any number of heads.
     */
    public fun encodeFor(sequences: List<LayaSequence>): LayaEncodedSequences<T> {
        val passes = ArrayList<LayaEncodedSequences.Pass<T>>()
        for (picked in planBatches(IntArray(sequences.size) { sequences[it].length }, maxBatchTokens)) {
            if (picked.size == 1) {
                val i = picked[0]
                passes += LayaEncodedSequences.Pass.Single(i, model.encode(sequences[i].ids))
            } else {
                passes += LayaEncodedSequences.Pass.Batch(picked, model.encodeBatch(picked.map { sequences[it] }, preprocessor.specialIds.pad))
            }
        }
        return LayaEncodedSequences(sequences, passes)
    }

    /** Logits of [head] for the sequences of [encoded], in their order. */
    public fun logitsFrom(encoded: LayaEncodedSequences<T>, types: List<LayaQuestionType>, head: LayaDecisionHead<T> = model.defaultHead): List<FloatArray> {
        val sequences = encoded.sequences
        require(types.size == sequences.size) { "${types.size} question types for ${sequences.size} sequences" }
        val out = arrayOfNulls<FloatArray>(sequences.size)
        for (pass in encoded.passes) {
            when (pass) {
                is LayaEncodedSequences.Pass.Single -> out[pass.index] = model.logitsFrom(pass.encoded, sequences[pass.index], types[pass.index], head)
                is LayaEncodedSequences.Pass.Batch -> {
                    val picked = pass.indices
                    val logits = model.logitsBatchFrom(picked.map { sequences[it] }, pass.encoded, picked.map { types[it] }, head)
                    for ((k, i) in picked.withIndex()) out[i] = logits[k]
                }
            }
        }
        return out.map { it ?: error("encoded sequences do not cover every index") }
    }

    public companion object {
        /** Default [maxBatchTokens]: eight sequences of 512 tokens, or sixteen of 256. */
        public const val DEFAULT_MAX_BATCH_TOKENS: Int = 4096

        /** Padded tokens a batch may add on top of its real tokens in [logitsFor]. */
        public const val MAX_PADDING_FRACTION: Double = 0.25

        /**
         * The batches [logitsFor] runs for sequences of these [lengths]: indices grouped by ascending
         * length while `size * longest <= maxBatchTokens` and the padding stays under
         * [maxPaddingFraction] of the real tokens. Every index appears in exactly one batch.
         */
        public fun planBatches(lengths: IntArray, maxBatchTokens: Int, maxPaddingFraction: Double = MAX_PADDING_FRACTION): List<IntArray> {
            val order = lengths.indices.sortedBy { lengths[it] }
            val batches = ArrayList<IntArray>()
            var start = 0
            while (start < order.size) {
                var end = start + 1
                var real = lengths[order[start]]
                while (end < order.size) {
                    val next = lengths[order[end]] // the longest so far: sorted ascending
                    val count = end - start + 1
                    if (count * next > maxBatchTokens) break
                    if (count * next - (real + next) > maxPaddingFraction * (real + next)) break
                    real += next
                    end++
                }
                batches += order.subList(start, end).toIntArray()
                start = end
            }
            return batches
        }
    }
}
