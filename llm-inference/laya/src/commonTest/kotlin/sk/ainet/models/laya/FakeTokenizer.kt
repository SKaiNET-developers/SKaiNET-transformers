package sk.ainet.models.laya

import sk.ainet.frontend.text.ArrayVocabulary
import sk.ainet.frontend.text.SpecialTokens
import sk.ainet.frontend.text.Tokenizer
import sk.ainet.frontend.text.Vocabulary

/**
 * One token per character, id = code point + 10, so budgets can be counted by hand. Ids 0..4 are
 * the five special tokens of the multilingual checkpoint.
 */
internal object FakeTokenizer : Tokenizer {
    override val vocabulary: Vocabulary = ArrayVocabulary(listOf("<pad>", "<eos>", "<bos>", "<unk>", "<mask>"))
    override val specialTokens: SpecialTokens = SpecialTokens(bos = 2, eos = 1, pad = 0, unk = 3)
    override fun encode(text: String): IntArray = text.map { it.code + 10 }.toIntArray()
    override fun decode(ids: IntArray): String = ids.map { (it - 10).toChar() }.joinToString("")

    val specialIds = LayaSpecialIds(cls = 2, sep = 1, mask = 4, pad = 0, maskToken = "<mask>")
}
