package sk.ainet.models.gemma

/**
 * Parser for HuggingFace Gemma 4 config.json files.
 *
 * Extracts model configuration and converts it to [GemmaModelMetadata].
 * Recognizes `model_type=gemma4` and `architectures=["Gemma4ForConditionalGeneration"]`.
 */
public object GemmaConfigParser {

    /**
     * Parse HuggingFace config.json content and extract GemmaModelMetadata.
     *
     * @param configJson The raw JSON string from config.json
     * @return Parsed model metadata
     * @throws IllegalArgumentException if required fields are missing
     */
    public fun parseFromJson(configJson: String): GemmaModelMetadata {
        val trimmed = configJson.trim()
        require(trimmed.startsWith("{") && trimmed.endsWith("}")) {
            "Invalid config.json: not a JSON object"
        }

        val content = trimmed.substring(1, trimmed.length - 1).trim()
        val topLevel = parseJsonObject(content)

        val architecture = (topLevel["model_type"] as? String) ?: "gemma4"

        // Get text_config section
        val textConfig = topLevel["text_config"]
        require(textConfig is Map<*, *>) { "Missing or invalid text_config in config.json" }
        @Suppress("UNCHECKED_CAST")
        val textConfigMap = textConfig as Map<String, Any?>

        // Required fields
        val blockCount = textConfigMap["num_hidden_layers"]?.toIntSafe()
            ?: error("Missing num_hidden_layers in text_config")
        val embeddingLength = textConfigMap["hidden_size"]?.toIntSafe()
            ?: error("Missing hidden_size in text_config")
        val headCount = textConfigMap["num_attention_heads"]?.toIntSafe()
            ?: error("Missing num_attention_heads in text_config")
        val kvHeadCount = textConfigMap["num_key_value_heads"]?.toIntSafe()
            ?: error("Missing num_key_value_heads in text_config")
        val headDim = textConfigMap["head_dim"]?.toIntSafe()
            ?: error("Missing head_dim in text_config")
        val vocabSize = textConfigMap["vocab_size"]?.toIntSafe()
            ?: error("Missing vocab_size in text_config")

        // Optional/defaultable fields
        val intermediateSize = textConfigMap["intermediate_size"]?.toIntSafe()
            ?: (embeddingLength * 4)
        val globalHeadDim = textConfigMap["global_head_dim"]?.toIntSafe()
            ?: GemmaModelMetadata.DEFAULT_GLOBAL_HEAD_DIM
        val contextLength = textConfigMap["max_position_embeddings"]?.toIntSafe()
            ?: 131072
        val slidingWindow = textConfigMap["sliding_window"]?.toIntSafe()
            ?: GemmaModelMetadata.DEFAULT_SLIDING_WINDOW
        val kvSharedLayers = textConfigMap["num_kv_shared_layers"]?.toIntSafe()
            ?: GemmaModelMetadata.DEFAULT_KV_SHARED_LAYERS
        val perLayerEmbeddingLength = textConfigMap["hidden_size_per_layer_input"]?.toIntSafe()
            ?: 0

        // Layer types (full per-layer list)
        val layerTypes = extractLayerTypes(textConfigMap, blockCount)

        // RoPE parameters (nested structure with full and sliding configs)
        val (ropeFull, ropeSliding) = extractRopeParameters(textConfigMap)

        // Token IDs from top level
        val bosTokenId = topLevel["bos_token_id"]?.toIntSafe() ?: 2
        val eosTokenId = topLevel["eos_token_id"]?.toIntSafe() ?: 1
        val padTokenId = topLevel["pad_token_id"]?.toIntSafe() ?: 0

        return GemmaModelMetadata(
            architecture = architecture,
            embeddingLength = embeddingLength,
            contextLength = contextLength,
            blockCount = blockCount,
            headCount = headCount,
            kvHeadCount = kvHeadCount,
            intermediateSize = intermediateSize,
            headDim = headDim,
            globalHeadDim = globalHeadDim,
            vocabSize = vocabSize,
            slidingWindow = slidingWindow,
            kvSharedLayers = kvSharedLayers,
            layerTypes = layerTypes,
            ropeParametersFull = ropeFull,
            ropeParametersSliding = ropeSliding,
            maxPositionEmbeddings = contextLength,
            perLayerEmbeddingLength = perLayerEmbeddingLength,
            bosTokenId = bosTokenId,
            eosTokenId = eosTokenId,
            padTokenId = padTokenId
        )
    }

    private fun extractLayerTypes(textConfig: Map<String, Any?>, blockCount: Int): List<String> {
        val layerTypes = textConfig["layer_types"]
        if (layerTypes is List<*>) {
            val mapped = layerTypes.mapNotNull { type ->
                when (type) {
                    is String -> type
                    else -> null
                }
            }
            if (mapped.isNotEmpty()) return mapped
        }
        // Default pattern: 5 sliding + 1 full repeating, last layer always full
        return buildDefaultLayerTypes(blockCount)
    }

    private fun buildDefaultLayerTypes(blockCount: Int): List<String> {
        return List(blockCount) { idx ->
            if (idx == blockCount - 1) {
                "full_attention"
            } else if ((idx + 1) % 6 == 0) {
                "full_attention"
            } else {
                "sliding_attention"
            }
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun extractRopeParameters(
        textConfig: Map<String, Any?>
    ): Pair<GemmaRopeConfig, GemmaRopeConfig> {
        val ropeParams = textConfig["rope_parameters"]

        if (ropeParams is Map<*, *>) {
            val ropeMap = ropeParams as Map<String, Any?>

            val fullConfig = extractSingleRopeConfig(ropeMap["full"], "proportional")
            val slidingConfig = extractSingleRopeConfig(ropeMap["sliding"], "default")

            return fullConfig to slidingConfig
        }

        // Fallback: use rope_theta from text_config
        val ropeTheta = textConfig["rope_theta"]?.toFloatSafe() ?: 1000000f
        val ropeLocalBase = textConfig["rope_local_base_freq"]?.toFloatSafe() ?: 10000f

        return GemmaRopeConfig(
            base = ropeTheta,
            ropeType = "proportional"
        ) to GemmaRopeConfig(
            base = ropeLocalBase,
            ropeType = "default"
        )
    }

    @Suppress("UNCHECKED_CAST")
    private fun extractSingleRopeConfig(
        config: Any?,
        defaultRopeType: String
    ): GemmaRopeConfig {
        if (config !is Map<*, *>) {
            return GemmaRopeConfig(
                base = if (defaultRopeType == "proportional") 1000000f else 10000f,
                ropeType = defaultRopeType
            )
        }

        val map = config as Map<String, Any?>
        val ropeType = (map["rope_type"] as? String) ?: defaultRopeType
        val factor = map["factor"]?.toFloatSafe() ?: 1.0f
        val originalMaxPos = map["original_max_position_embeddings"]?.toIntSafe() ?: 8192
        val partialRotaryFactor = map["partial_rotary_factor"]?.toFloatSafe() ?: 1.0f
        val base = map["base"]?.toFloatSafe()
            ?: if (ropeType == "proportional") 1000000f else 10000f

        return GemmaRopeConfig(
            base = base,
            ropeType = ropeType,
            factor = factor,
            originalMaxPositionEmbeddings = originalMaxPos,
            partialRotaryFactor = partialRotaryFactor
        )
    }

    // ========== JSON Parsing Helpers ==========

    private fun parseJsonObject(content: String): Map<String, Any?> {
        val result = mutableMapOf<String, Any?>()
        var i = 0
        val n = content.length

        while (i < n) {
            while (i < n && content[i].isWhitespace()) i++
            if (i >= n) break

            if (content[i] != '"') {
                i++
                continue
            }
            val keyEnd = findStringEnd(content, i)
            val key = unescapeString(content.substring(i + 1, keyEnd))
            i = keyEnd + 1

            while (i < n && content[i].isWhitespace()) i++
            if (i >= n || content[i] != ':') {
                i++
                continue
            }
            i++
            while (i < n && content[i].isWhitespace()) i++

            val (value, newPos) = parseJsonValue(content, i)
            result[key] = value
            i = newPos

            while (i < n && content[i].isWhitespace()) i++
            if (i < n && content[i] == ',') i++
        }

        return result
    }

    private fun parseJsonValue(s: String, start: Int): Pair<Any?, Int> {
        if (start >= s.length) return null to start

        return when (s[start]) {
            '"' -> {
                val end = findStringEnd(s, start)
                val value = unescapeString(s.substring(start + 1, end))
                value to (end + 1)
            }
            '{' -> {
                val end = findMatchingBrace(s, start, '{', '}')
                val content = s.substring(start + 1, end - 1).trim()
                val obj = parseJsonObject(content)
                obj to end
            }
            '[' -> {
                val end = findMatchingBrace(s, start, '[', ']')
                val content = s.substring(start + 1, end - 1).trim()
                val list = parseJsonArray(content)
                list to end
            }
            't' -> true to (start + 4)
            'f' -> false to (start + 5)
            'n' -> null to (start + 4)
            else -> {
                var end = start
                while (end < s.length && s[end] !in ",}]") end++
                val numStr = s.substring(start, end).trim()
                val value: Number = if (numStr.contains('.') || numStr.contains('e') || numStr.contains('E')) {
                    numStr.toDoubleOrNull() ?: 0.0
                } else {
                    numStr.toLongOrNull() ?: 0L
                }
                value to end
            }
        }
    }

    private fun parseJsonArray(content: String): List<Any?> {
        if (content.isEmpty()) return emptyList()

        val result = mutableListOf<Any?>()
        var i = 0
        val n = content.length

        while (i < n) {
            while (i < n && content[i].isWhitespace()) i++
            if (i >= n) break

            val (value, newPos) = parseJsonValue(content, i)
            result.add(value)
            i = newPos

            while (i < n && content[i].isWhitespace()) i++
            if (i < n && content[i] == ',') i++
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
        throw IllegalArgumentException("Unterminated string at position $start")
    }

    private fun findMatchingBrace(s: String, start: Int, open: Char, close: Char): Int {
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
                c == close -> {
                    depth--
                    if (depth == 0) return i + 1
                }
            }
            i++
        }
        throw IllegalArgumentException("Unmatched '$open' at position $start")
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
                    'b' -> { sb.append('\b'); i += 2 }
                    'f' -> { sb.append('\u000C'); i += 2 }
                    'n' -> { sb.append('\n'); i += 2 }
                    'r' -> { sb.append('\r'); i += 2 }
                    't' -> { sb.append('\t'); i += 2 }
                    'u' -> {
                        if (i + 5 < s.length) {
                            val hex = s.substring(i + 2, i + 6)
                            sb.append(hex.toInt(16).toChar())
                            i += 6
                        } else {
                            sb.append(s[i])
                            i++
                        }
                    }
                    else -> {
                        sb.append(s[i])
                        i++
                    }
                }
            } else {
                sb.append(s[i])
                i++
            }
        }
        return sb.toString()
    }

    private fun Any?.toIntSafe(): Int? = when (this) {
        is Int -> this
        is Long -> this.toInt()
        is Double -> this.toInt()
        is Float -> this.toInt()
        is Number -> this.toInt()
        is String -> this.toIntOrNull()
        else -> null
    }

    private fun Any?.toFloatSafe(): Float? = when (this) {
        is Float -> this
        is Double -> this.toFloat()
        is Number -> this.toFloat()
        is String -> this.toFloatOrNull()
        else -> null
    }
}
