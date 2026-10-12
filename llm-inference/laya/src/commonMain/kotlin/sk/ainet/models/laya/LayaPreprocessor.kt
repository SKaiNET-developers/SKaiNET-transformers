package sk.ainet.models.laya

import sk.ainet.frontend.text.Tokenizer
import sk.ainet.frontend.tokenizer.TextFrontEnds

/** The four text files of a Laya checkpoint directory, as strings; the weights are loaded separately. */
public class LayaCheckpointFiles(
    public val agentConfigJson: String,
    public val encoderConfigJson: String,
    public val tokenizerJson: String,
    public val tokenizerConfigJson: String,
) {
    public companion object
}

/**
 * Everything before the encoder: the checkpoint's tokenizer (engine text front end), its special
 * ids and the sequence builder. One instance per checkpoint; thread-safe for concurrent [prepare]
 * calls as long as the tokenizer is.
 */
public class LayaPreprocessor(
    public val tokenizer: Tokenizer,
    public val agentConfig: LayaAgentConfig,
    public val encoderConfig: LayaEncoderConfig,
    public val specialIds: LayaSpecialIds,
) {
    public val builder: LayaSequenceBuilder = LayaSequenceBuilder(tokenizer, specialIds, agentConfig)

    /**
     * Packs one question about [state]. Fewer markers than options is the reference's hard error
     * (`options do not fit in head_max_len`), raised here as [LayaSequenceException] (REQ-PRE-002).
     */
    public fun prepare(state: LayaState, question: LayaQuestion, optionOrder: IntArray? = null, stateIds: IntArray? = null): LayaSequence {
        val sequence = builder.build(state, question, optionOrder, stateIds = stateIds)
        if (sequence.markers.size != question.optionCount) {
            throw LayaSequenceException(
                "question: options do not fit in head_max_len=${agentConfig.headMaxLen} tokens " +
                    "(${sequence.markers.size} markers for ${question.optionCount} options)",
            )
        }
        return sequence
    }

    public companion object {
        /** Builds the preprocessor from the checkpoint files; rejects unsupported configs at load time. */
        public fun fromFiles(files: LayaCheckpointFiles): LayaPreprocessor {
            val agent = LayaAgentConfig.parse(files.agentConfigJson).requireSupported()
            val encoder = LayaEncoderConfig.parse(files.encoderConfigJson).requireSupported()
            val tokenizer = TextFrontEnds.fromTokenizerJson(files.tokenizerJson)
            val tokenizerConfig = LayaTokenizerConfig.parse(files.tokenizerConfigJson)
            val specialIds = LayaSpecialIds.resolve(tokenizer, tokenizerConfig)
            return LayaPreprocessor(tokenizer, agent, encoder, specialIds)
        }
    }
}
