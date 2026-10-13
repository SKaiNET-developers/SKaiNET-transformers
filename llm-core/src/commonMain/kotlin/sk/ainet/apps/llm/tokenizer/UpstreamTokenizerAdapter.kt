package sk.ainet.apps.llm.tokenizer

import sk.ainet.apps.llm.Tokenizer
import sk.ainet.frontend.FrontEndSpec
import sk.ainet.frontend.text.SpecialTokens
import sk.ainet.frontend.text.Vocabulary

/**
 * Presents an engine [sk.ainet.frontend.text.Tokenizer] as this repository's [Tokenizer]: the only
 * things added are the non-null `-1` sentinels for a missing bos/eos and the `decode(Int)`
 * spelling. Everything else delegates, including the `addSpecialTokens` / `skipSpecialTokens`
 * overloads, so the checkpoint policy is preserved.
 *
 * Deleted in 0.60 together with the sub-interface (SKEEP-006).
 */
internal class UpstreamTokenizerAdapter(
    private val delegate: sk.ainet.frontend.text.Tokenizer,
) : Tokenizer {

    override val vocabulary: Vocabulary get() = delegate.vocabulary
    override val specialTokens: SpecialTokens get() = delegate.specialTokens
    override val spec: FrontEndSpec get() = delegate.spec

    override val vocabSize: Int get() = delegate.vocabSize
    override val bosTokenId: Int = delegate.specialTokens.bos ?: -1
    override val eosTokenId: Int = delegate.specialTokens.eos ?: -1

    override fun encode(text: String): IntArray = delegate.encode(text)
    override fun encode(text: String, addSpecialTokens: Boolean): IntArray = delegate.encode(text, addSpecialTokens)

    override fun decode(ids: IntArray): String = delegate.decode(ids)
    override fun decode(ids: IntArray, skipSpecialTokens: Boolean): String = delegate.decode(ids, skipSpecialTokens)

    @Deprecated("Use decodeToken(id); same semantics (SKEEP-006).", ReplaceWith("decodeToken(token)"))
    override fun decode(token: Int): String = delegate.decodeToken(token)

    override fun decodeToken(id: Int): String = delegate.decodeToken(id)
}
