package sk.ainet.models.voxtral

import sk.ainet.apps.llm.Tokenizer
import sk.ainet.io.tokenizer.TekkenTokenizer

/**
 * Adapts [TekkenTokenizer] (from skainet-io-core) to the [Tokenizer] interface
 * used by the LLM runtime and CLI tools.
 *
 * Usage:
 * ```kotlin
 * val tekken = TekkenTokenizer.fromJson(tekkenJsonString)
 * val tokenizer: Tokenizer = TekkenTokenizerAdapter(tekken)
 * val tokens = tokenizer.encode("Hello, world!")
 * val text = tokenizer.decode(tokens)
 * ```
 */
public class TekkenTokenizerAdapter(
    private val tekken: TekkenTokenizer
) : Tokenizer {

    override fun encode(text: String): IntArray = tekken.encode(text)

    override fun decode(tokens: IntArray): String = tekken.decode(tokens)

    override fun decode(token: Int): String = tekken.decode(token)

    public companion object {
        /**
         * Parse a tekken.json string and return a [Tokenizer] instance.
         */
        public fun fromJson(json: String): Tokenizer {
            return TekkenTokenizerAdapter(TekkenTokenizer.fromJson(json))
        }
    }
}
