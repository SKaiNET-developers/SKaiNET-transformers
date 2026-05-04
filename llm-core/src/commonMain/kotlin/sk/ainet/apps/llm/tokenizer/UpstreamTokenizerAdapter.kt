package sk.ainet.apps.llm.tokenizer

import sk.ainet.apps.llm.Tokenizer

/**
 * Bridges an upstream [sk.ainet.io.tokenizer.Tokenizer] (in
 * `skainet-io-core`) into this repo's [Tokenizer] interface.
 *
 * Upstream's `bosTokenId` / `eosTokenId` are nullable; the local interface
 * is non-null with int defaults. Upstream has no single-token decode; the
 * adapter implements it via `decode(intArrayOf(token))`.
 *
 * Used by [TokenizerFactory.fromGGUF] to route Qwen / GPT-2 / Mistral-Nemo
 * GGUFs (`tokenizer.ggml.model == "gpt2"` / `"bpe"`) to the correct
 * byte-level BPE implementation, instead of the broken local
 * [GGUFTokenizer]'s greedy-by-score `encodeBPE`. Closes #52.
 */
internal class UpstreamTokenizerAdapter(
    private val delegate: sk.ainet.io.tokenizer.Tokenizer,
    bosTokenIdFallback: Int = 1,
    eosTokenIdFallback: Int = 2,
) : Tokenizer {

    override val vocabSize: Int get() = delegate.vocabSize
    override val bosTokenId: Int = delegate.bosTokenId ?: bosTokenIdFallback
    override val eosTokenId: Int = delegate.eosTokenId ?: eosTokenIdFallback

    override fun encode(text: String): IntArray = delegate.encode(text)

    override fun decode(tokens: IntArray): String = delegate.decode(tokens)

    override fun decode(token: Int): String = delegate.decode(intArrayOf(token))
}
