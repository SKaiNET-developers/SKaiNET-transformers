package sk.ainet.models.laya

/**
 * Preprocessor, model and decoder of one checkpoint: ask typed questions about a state.
 *
 * @param maxBatchTokens token budget of one encoder pass in [decideAll] and [logitsFor]: consecutive
 *   sequences share a pass while `batchSize * longest <= maxBatchTokens`. Attention scores grow with
 *   `batch * heads * longest²`, so the budget bounds the transient memory of a request.
 */
public class LayaAgent<T : sk.ainet.lang.types.DType>(
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
    public fun decide(state: LayaState, question: LayaQuestion, optionOrder: IntArray? = null, stateIds: IntArray? = null): Result {
        val sequence = preprocessor.prepare(state, question, optionOrder, stateIds)
        val logits = model.logits(sequence, question.type)
        return Result(decoder.decode(question, logits, sequence.optionOrder), sequence, logits)
    }

    /**
     * Every question about the same state: the state is tokenised once and the questions run as
     * padded batches under [maxBatchTokens] instead of one encoder pass each.
     */
    public fun decideAll(state: LayaState, questions: Map<String, LayaQuestion>): Map<String, Result> {
        if (questions.isEmpty()) return emptyMap()
        val stateIds = preprocessor.builder.encodeState(state)
        val keys = questions.keys.toList()
        val asked = keys.map { questions.getValue(it) }
        val sequences = asked.map { preprocessor.prepare(state, it, stateIds = stateIds) }
        val logits = logitsFor(sequences, asked.map { it.type })
        val results = LinkedHashMap<String, Result>(keys.size)
        for (i in keys.indices) {
            results[keys[i]] = Result(decoder.decode(asked[i], logits[i], sequences[i].optionOrder), sequences[i], logits[i])
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
    public fun logitsFor(sequences: List<LayaSequence>, types: List<LayaQuestionType>): List<FloatArray> {
        require(types.size == sequences.size) { "${types.size} question types for ${sequences.size} sequences" }
        val out = arrayOfNulls<FloatArray>(sequences.size)
        for (picked in planBatches(IntArray(sequences.size) { sequences[it].length }, maxBatchTokens)) {
            if (picked.size == 1) {
                val i = picked[0]
                out[i] = model.logits(sequences[i], types[i])
            } else {
                val logits = model.logitsBatch(picked.map { sequences[it] }, picked.map { types[it] }, preprocessor.specialIds.pad)
                for ((k, i) in picked.withIndex()) out[i] = logits[k]
            }
        }
        return out.map { it!! }
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
