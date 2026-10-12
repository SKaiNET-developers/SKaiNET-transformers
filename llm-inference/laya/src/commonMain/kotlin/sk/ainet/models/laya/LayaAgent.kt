package sk.ainet.models.laya

/** Preprocessor, model and decoder of one checkpoint: ask typed questions about a state. */
public class LayaAgent<T : sk.ainet.lang.types.DType>(
    public val preprocessor: LayaPreprocessor,
    public val model: LayaDecisionModel<T>,
    public val decoder: LayaDecoder = LayaDecoder(preprocessor.agentConfig),
) {
    public class Result(public val answer: LayaAnswer, public val sequence: LayaSequence, public val logits: FloatArray)

    public fun decide(state: LayaState, question: LayaQuestion, optionOrder: IntArray? = null, stateIds: IntArray? = null): Result {
        val sequence = preprocessor.prepare(state, question, optionOrder, stateIds)
        val logits = model.logits(sequence, question.type)
        return Result(decoder.decode(question, logits, sequence.optionOrder), sequence, logits)
    }

    /** Every question about the same state; the state is tokenised once. */
    public fun decideAll(state: LayaState, questions: Map<String, LayaQuestion>): Map<String, Result> {
        val stateIds = preprocessor.builder.encodeState(state)
        return questions.mapValues { (_, q) -> decide(state, q, stateIds = stateIds) }
    }
}
