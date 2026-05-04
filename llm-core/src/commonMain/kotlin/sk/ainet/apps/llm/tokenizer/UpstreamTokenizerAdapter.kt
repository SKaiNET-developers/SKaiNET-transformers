package sk.ainet.apps.llm.tokenizer

import sk.ainet.apps.llm.Tokenizer

/**
 * Bridges an upstream [sk.ainet.io.tokenizer.Tokenizer] (in
 * `skainet-io-core`) into this repo's [Tokenizer] interface.
 *
 * Upstream's `bosTokenId` / `eosTokenId` are nullable; the local interface
 * is non-null. Upstream has no single-token decode; the adapter implements
 * it via `decode(intArrayOf(token))`.
 *
 * Used by [TokenizerFactory] to route Qwen / GPT-2 / Mistral-Nemo GGUFs
 * (and tokenizer.json files) to the correct upstream byte-level BPE,
 * instead of the broken local greedy-by-score `encodeBPE`. Closes #52.
 *
 * Models without BOS / EOS report `-1` so the absence is detectable —
 * none of the GGUF chat models in this repo currently lack them, and
 * silent fallback to a "reasonable" id would mask real bugs.
 *
 * Becomes redundant once the local `Tokenizer` interface is replaced by
 * a typealias to upstream — at which point this whole class is deleted.
 */
internal class UpstreamTokenizerAdapter(
    private val delegate: sk.ainet.io.tokenizer.Tokenizer,
) : Tokenizer {

    override val vocabSize: Int get() = delegate.vocabSize
    override val bosTokenId: Int = delegate.bosTokenId ?: -1
    override val eosTokenId: Int = delegate.eosTokenId ?: -1

    override fun encode(text: String): IntArray = delegate.encode(text)

    override fun decode(tokens: IntArray): String = delegate.decode(tokens)

    override fun decode(token: Int): String = delegate.decode(intArrayOf(token))
}
