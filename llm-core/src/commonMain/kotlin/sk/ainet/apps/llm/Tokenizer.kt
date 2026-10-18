package sk.ainet.apps.llm

import sk.ainet.frontend.text.SpecialTokens
import sk.ainet.frontend.text.Vocabulary

/**
 * The tokenizer surface the runtimes in this repository consume.
 *
 * Since 0.58 (SKEEP-006) this is a sub-interface of the engine's
 * [sk.ainet.frontend.text.Tokenizer]: every engine tokenizer can be passed where this type is
 * expected (through `TokenizerFactory`), and every implementation here is an engine tokenizer.
 * The differences are the 0.57 conveniences this interface keeps for two releases:
 *
 *  - [bosTokenId] / [eosTokenId] are non-null `Int`, `-1` when the checkpoint declares none
 *    (the engine's are nullable — prefer `specialTokens.bos` / `specialTokens.eos` in new code);
 *  - [decode] of a single id is spelled `decode(token)`; the engine name is [decodeToken].
 *
 * In 0.60 this interface becomes a type alias of the engine interface.
 */
interface Tokenizer : sk.ainet.frontend.text.Tokenizer {
    override fun encode(text: String): IntArray
    override fun decode(ids: IntArray): String

    /** Single-token decode for streaming (0.57 spelling). */
    @Deprecated("Use decodeToken(id); same semantics (SKEEP-006).", ReplaceWith("decodeToken(token)"))
    fun decode(token: Int): String

    @Suppress("DEPRECATION")
    override fun decodeToken(id: Int): String = decode(id)

    /** End-of-sequence token id, or `-1` when the checkpoint declares none. */
    override val eosTokenId: Int

    /** Beginning-of-sequence token id, or `-1` when the checkpoint declares none. */
    override val bosTokenId: Int

    /** Total vocabulary size. */
    override val vocabSize: Int

    /** Legacy implementations expose no tables; the engine contract still needs a size. */
    override val vocabulary: Vocabulary get() = SizeOnlyVocabulary(vocabSize)

    override val specialTokens: SpecialTokens
        get() = SpecialTokens(bos = bosTokenId.takeIf { it >= 0 }, eos = eosTokenId.takeIf { it >= 0 })
}

/** A [Vocabulary] that knows only its size — the bridge for 0.57 implementations. */
internal class SizeOnlyVocabulary(override val size: Int) : Vocabulary {
    override fun tokenToId(piece: String): Int? = null
    override fun idToToken(id: Int): String? = null
}
