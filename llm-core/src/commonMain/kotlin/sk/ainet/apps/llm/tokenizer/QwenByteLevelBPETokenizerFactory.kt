package sk.ainet.apps.llm.tokenizer

/**
 * Creates a [QwenByteLevelBPETokenizer] from HuggingFace tokenizer files.
 *
 * @param tokenizerJson Content of `tokenizer.json` (vocab, merges, added_tokens)
 * @param tokenizerConfigJson Content of `tokenizer_config.json` (pretokenize_regex, special tokens, etc.)
 * @return Configured Qwen byte-level BPE tokenizer
 */
public fun createQwenBPETokenizerFromJson(
    tokenizerJson: String,
    tokenizerConfigJson: String? = null
): QwenByteLevelBPETokenizer {
    val parser = TokenizerJsonParser(tokenizerJson)

    // Parse vocab
    val vocabMap = parser.parseVocab()
    val vocabSize = vocabMap.size
    val vocab = MutableList(vocabSize) { "" }
    vocabMap.forEach { (token, id) ->
        if (id < vocabSize) {
            vocab[id] = token
        }
    }

    // Parse merges into rank map (earlier merge = lower rank = higher priority)
    val merges = parser.parseMerges()
    val mergeRanks = mutableMapOf<String, Int>()
    merges.forEachIndexed { index, (left, right) ->
        mergeRanks[left + right] = index
    }

    // Parse special tokens from added_tokens section
    val addedTokens = parser.parseAddedTokens()

    // Also collect special tokens that are marked as special in added_tokens
    val specialTokens = mutableMapOf<String, Int>()
    // Add known Qwen special tokens if found in vocab or added_tokens
    val knownSpecialPatterns = listOf(
        "<|im_start|>", "<|im_end|>", "<|endoftext|>",
        "<tool_call>", "</tool_call>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>",
        "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>"
    )

    // From added_tokens (these are definitively special)
    for ((token, id) in addedTokens) {
        if (token.startsWith("<|") || token.startsWith("<tool") || token.startsWith("</tool")) {
            specialTokens[token] = id
        }
    }

    // Also check known patterns against vocab
    for (pattern in knownSpecialPatterns) {
        if (pattern !in specialTokens) {
            val id = vocabMap[pattern]
            if (id != null) {
                specialTokens[pattern] = id
            }
        }
    }

    val eosTokenId = specialTokens["<|im_end|>"]
        ?: addedTokens["<|im_end|>"]
        ?: addedTokens["<|endoftext|>"]
        ?: vocabMap["<|endoftext|>"]
        ?: 1
    val bosTokenId = addedTokens["<|im_start|>"]
        ?: vocabMap["<|im_start|>"]
        ?: 0

    // Parse config for pretokenize_regex and flags
    var pretokenizeRegex: Regex? = null
    var addBosToken = false
    var addEosToken = false

    if (tokenizerConfigJson != null) {
        val configParser = QwenTokenizerConfigParser(tokenizerConfigJson)
        val regexStr = configParser.getString("pretokenize_regex")
        if (regexStr != null) {
            pretokenizeRegex = try {
                Regex(regexStr)
            } catch (_: Exception) {
                null
            }
        }
        addBosToken = configParser.getBoolean("add_bos_token") ?: false
        addEosToken = configParser.getBoolean("add_eos_token") ?: false
    }

    return QwenByteLevelBPETokenizer(
        vocab = vocab,
        tokenToId = vocabMap,
        mergeRanks = mergeRanks,
        specialTokens = specialTokens,
        pretokenizeRegex = pretokenizeRegex,
        eosTokenId = eosTokenId,
        addEosToken = addEosToken,
        addBosToken = addBosToken,
        bosTokenId = bosTokenId
    )
}

/**
 * Checks if a tokenizer_config.json indicates a Qwen2Tokenizer.
 */
public fun isQwen2Tokenizer(tokenizerConfigJson: String): Boolean {
    val parser = QwenTokenizerConfigParser(tokenizerConfigJson)
    val tokenizerClass = parser.getString("tokenizer_class")
    return tokenizerClass == "Qwen2Tokenizer"
}

/**
 * Parser for tokenizer_config.json that supports string and boolean fields.
 */
internal class QwenTokenizerConfigParser(private val json: String) {

    fun getBoolean(key: String): Boolean? {
        val pattern = Regex("\"$key\"\\s*:\\s*(true|false)")
        val match = pattern.find(json) ?: return null
        return match.groupValues[1] == "true"
    }

    fun getString(key: String): String? {
        val pattern = Regex("\"$key\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"")
        val match = pattern.find(json) ?: return null
        return unescapeString(match.groupValues[1])
    }

    private fun unescapeString(s: String): String {
        val sb = StringBuilder()
        var i = 0
        while (i < s.length) {
            if (s[i] == '\\' && i + 1 < s.length) {
                when (s[i + 1]) {
                    '"' -> { sb.append('"'); i += 2 }
                    '\\' -> { sb.append('\\'); i += 2 }
                    '/' -> { sb.append('/'); i += 2 }
                    'n' -> { sb.append('\n'); i += 2 }
                    'r' -> { sb.append('\r'); i += 2 }
                    't' -> { sb.append('\t'); i += 2 }
                    'u' -> {
                        if (i + 5 < s.length) {
                            val hex = s.substring(i + 2, i + 6)
                            val codePoint = hex.toIntOrNull(16)
                            if (codePoint != null) {
                                sb.append(codePoint.toChar())
                            }
                            i += 6
                        } else {
                            sb.append(s[i]); i++
                        }
                    }
                    else -> { sb.append(s[i]); i++ }
                }
            } else {
                sb.append(s[i]); i++
            }
        }
        return sb.toString()
    }
}
