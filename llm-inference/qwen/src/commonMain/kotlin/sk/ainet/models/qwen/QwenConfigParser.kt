package sk.ainet.models.qwen

import sk.ainet.models.llama.LlamaModelMetadata

/**
 * Parses HuggingFace `config.json` for Qwen2 / Qwen3 / Qwen3.5 models into [LlamaModelMetadata].
 *
 * Qwen2/3 use the same transformer architecture as LLaMA (GQA + SwiGLU FFN + RoPE),
 * so they share the same metadata structure. The config field names are identical
 * to LLaMA (`hidden_size`, `num_hidden_layers`, etc.).
 *
 * Qwen3.5 `model_type` values (`qwen3_5`, `qwen3_5_moe`) are normalized to
 * architecture string `"qwen35"` so downstream code can distinguish Qwen3.5 from
 * earlier Qwen variants.
 *
 * Qwen-specific fields like `max_window_layers` and `sliding_window` are parsed
 * but not yet exposed in the metadata (they don't affect basic inference).
 */
public object QwenConfigParser {

    /**
     * Parse a HuggingFace config.json string for Qwen2 into [LlamaModelMetadata].
     *
     * Required fields: hidden_size, num_hidden_layers, num_attention_heads,
     * num_key_value_heads, intermediate_size, vocab_size.
     * Optional: max_position_embeddings, head_dim.
     */
    public fun parse(json: String): LlamaModelMetadata {
        val map = parseJsonObject(json.trim())

        val hiddenSize = map.requireInt("hidden_size")
        val numLayers = map.requireInt("num_hidden_layers")
        val numHeads = map.requireInt("num_attention_heads")
        val numKvHeads = map.intOrNull("num_key_value_heads") ?: numHeads
        val intermediateSize = map.requireInt("intermediate_size")
        val vocabSize = map.requireInt("vocab_size")
        val contextLength = map.intOrNull("max_position_embeddings") ?: 32768
        val headDim = map.intOrNull("head_dim") ?: (hiddenSize / numHeads)
        val rawModelType = map.stringOrNull("model_type") ?: "qwen2"
        val architecture = normalizeArchitecture(rawModelType)
        val ropeTheta = map.floatOrNull("rope_theta") ?: 1_000_000f
        val rmsNormEps = map.floatOrNull("rms_norm_eps") ?: 1e-6f

        return LlamaModelMetadata(
            architecture = architecture,
            embeddingLength = hiddenSize,
            contextLength = contextLength,
            blockCount = numLayers,
            headCount = numHeads,
            kvHeadCount = numKvHeads,
            feedForwardLength = intermediateSize,
            ropeDimensionCount = headDim,
            vocabSize = vocabSize,
            ropeFreqBase = ropeTheta,
            rmsNormEps = rmsNormEps
        )
    }

    /**
     * Check if config.json indicates tied word embeddings.
     */
    public fun isTiedEmbeddings(json: String): Boolean {
        val map = parseJsonObject(json.trim())
        return map["tie_word_embeddings"] == "true"
    }

    /**
     * Normalize Qwen `model_type` values to canonical architecture strings.
     *
     * Qwen3.5 configs may use `qwen3_5`, `qwen3_5_moe`, or similar variants;
     * all are normalized to `"qwen35"`. Earlier types pass through unchanged.
     */
    private fun normalizeArchitecture(modelType: String): String = when {
        modelType.startsWith("qwen3_5") || modelType.startsWith("qwen3.5") -> "qwen35"
        else -> modelType
    }

    // ========== Lightweight JSON parsing ==========

    private fun parseJsonObject(json: String): Map<String, String> {
        if (!json.startsWith("{") || !json.endsWith("}")) {
            error("config.json: expected JSON object")
        }
        val content = json.substring(1, json.length - 1)
        val result = mutableMapOf<String, String>()

        var i = 0
        while (i < content.length) {
            while (i < content.length && content[i].isWhitespace()) i++
            if (i >= content.length) break

            if (content[i] != '"') { i++; continue }
            val keyEnd = findStringEnd(content, i)
            val key = content.substring(i + 1, keyEnd)
            i = keyEnd + 1

            while (i < content.length && (content[i].isWhitespace() || content[i] == ':')) i++

            val valueStart = i
            i = skipValue(content, i)
            var value = content.substring(valueStart, i).trim()
            if (value.startsWith("\"") && value.endsWith("\"")) {
                value = value.substring(1, value.length - 1)
            }
            result[key] = value

            while (i < content.length && (content[i].isWhitespace() || content[i] == ',')) i++
        }

        return result
    }

    private fun findStringEnd(s: String, start: Int): Int {
        var i = start + 1
        while (i < s.length) {
            when (s[i]) {
                '"' -> return i
                '\\' -> i += 2
                else -> i++
            }
        }
        return s.length
    }

    private fun skipValue(s: String, start: Int): Int {
        if (start >= s.length) return start
        return when (s[start]) {
            '"' -> findStringEnd(s, start) + 1
            '{' -> findMatching(s, start, '{', '}')
            '[' -> findMatching(s, start, '[', ']')
            else -> {
                var i = start
                while (i < s.length && s[i] != ',' && s[i] != '}' && s[i] != ']') i++
                i
            }
        }
    }

    private fun findMatching(s: String, start: Int, open: Char, close: Char): Int {
        var depth = 0
        var i = start
        var inString = false
        while (i < s.length) {
            val c = s[i]
            when {
                inString -> {
                    if (c == '"') inString = false
                    else if (c == '\\') i++
                }
                c == '"' -> inString = true
                c == open -> depth++
                c == close -> { depth--; if (depth == 0) return i + 1 }
            }
            i++
        }
        return s.length
    }

    private fun Map<String, String>.requireInt(key: String): Int =
        this[key]?.toIntOrNull() ?: error("config.json: missing or invalid '$key'")

    private fun Map<String, String>.intOrNull(key: String): Int? =
        this[key]?.toIntOrNull()

    private fun Map<String, String>.floatOrNull(key: String): Float? =
        this[key]?.toFloatOrNull()

    private fun Map<String, String>.stringOrNull(key: String): String? =
        this[key]?.takeIf { it.isNotBlank() }
}
