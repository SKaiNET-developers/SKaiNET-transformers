package sk.ainet.apps.llm

interface Tokenizer {
    fun encode(text: String): IntArray
    fun decode(tokens: IntArray): String
    fun decode(token: Int): String

    /** End-of-sequence token ID. */
    val eosTokenId: Int

    /** Beginning-of-sequence token ID. */
    val bosTokenId: Int

    /** Total vocabulary size. */
    val vocabSize: Int
}
