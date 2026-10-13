package sk.ainet.models.voxtral

import sk.ainet.apps.llm.Tokenizer
import sk.ainet.frontend.text.SpecialTokens
import sk.ainet.frontend.text.Vocabulary
import sk.ainet.frontend.tokenizer.TekkenTokenizer

/**
 * Adapts the engine [TekkenTokenizer] to the [Tokenizer] interface used by the LLM runtime and
 * CLI tools.
 *
 * Deprecated (SKEEP-006): `TokenizerFactory.fromTekkenJson(json)` returns the same thing, and
 * the ids below come from the `tekken.json` file instead of the former hard-coded
 * `vocabSize = 32768`, `bos = 1`, `eos = 2`. Removed in 0.60.
 */
@Deprecated(
    "Use TokenizerFactory.fromTekkenJson(json) (SKEEP-006). Removed in 0.60.",
    ReplaceWith("TokenizerFactory.fromTekkenJson(json)", "sk.ainet.apps.llm.tokenizer.TokenizerFactory"),
)
public class TekkenTokenizerAdapter(
    private val tekken: TekkenTokenizer
) : Tokenizer {

    override fun encode(text: String): IntArray = tekken.encode(text)

    override fun decode(ids: IntArray): String = tekken.decode(ids)

    @Deprecated("Use decodeToken(id); same semantics (SKEEP-006).", ReplaceWith("decodeToken(token)"))
    override fun decode(token: Int): String = tekken.decodeToken(token)

    override val eosTokenId: Int = tekken.specialTokens.eos ?: 2
    override val bosTokenId: Int = tekken.specialTokens.bos ?: 1
    override val vocabSize: Int get() = tekken.vocabSize
    override val vocabulary: Vocabulary get() = tekken.vocabulary
    override val specialTokens: SpecialTokens get() = tekken.specialTokens

    public companion object {
        /** Parse a tekken.json string and return a [Tokenizer] instance. */
        public fun fromJson(json: String): Tokenizer {
            return TekkenTokenizerAdapter(TekkenTokenizer.fromJson(json))
        }
    }
}
