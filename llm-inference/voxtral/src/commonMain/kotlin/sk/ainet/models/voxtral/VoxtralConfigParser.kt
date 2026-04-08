package sk.ainet.models.voxtral

import sk.ainet.models.llama.LlamaModelMetadata

/**
 * Parses Mistral's `params.json` for Voxtral TTS models into [VoxtralModelMetadata].
 *
 * Voxtral uses Mistral's `params.json` format (not HuggingFace `config.json`).
 * The file contains nested objects for the backbone, acoustic model, and codec.
 *
 * Example params.json structure:
 * ```json
 * {
 *   "dim": 3072,
 *   "n_layers": 26,
 *   "n_heads": 32,
 *   "n_kv_heads": 8,
 *   "hidden_dim": 9216,
 *   "vocab_size": 131072,
 *   "rope_theta": 1000000.0,
 *   "norm_eps": 1e-5,
 *   "acoustic_model": { "dim": 3072, "n_layers": 3, ... },
 *   "codec": { ... }
 * }
 * ```
 */
public object VoxtralConfigParser {

    /**
     * Parse a Mistral params.json string into [VoxtralModelMetadata].
     *
     * Required backbone fields: dim, n_layers, n_heads, n_kv_heads, hidden_dim, vocab_size.
     * Optional: head_dim, max_seq_len, rope_theta.
     */
    public fun parse(json: String): VoxtralModelMetadata {
        val map = parseJsonObject(json.trim())

        val backbone = parseBackbone(map)
        val acousticModel = parseAcousticModel(map)
        val codec = parseCodec(map)
        val audio = parseAudio(map)

        return VoxtralModelMetadata(
            backbone = backbone,
            acousticModel = acousticModel,
            codec = codec,
            audio = audio
        )
    }

    /**
     * Parse backbone transformer config from top-level params.json fields.
     * Mistral uses different field names than HuggingFace:
     * - `dim` instead of `hidden_size`
     * - `n_layers` instead of `num_hidden_layers`
     * - `hidden_dim` instead of `intermediate_size`
     */
    public fun parseBackbone(map: Map<String, String>): LlamaModelMetadata {
        val dim = map.requireInt("dim")
        val nLayers = map.requireInt("n_layers")
        val nHeads = map.requireInt("n_heads")
        val nKvHeads = map.intOrNull("n_kv_heads") ?: nHeads
        val hiddenDim = map.requireInt("hidden_dim")
        val vocabSize = map.requireInt("vocab_size")
        val contextLength = map.intOrNull("max_seq_len") ?: 65536
        val headDim = map.intOrNull("head_dim") ?: (dim / nHeads)

        return LlamaModelMetadata(
            architecture = "voxtral_tts",
            embeddingLength = dim,
            contextLength = contextLength,
            blockCount = nLayers,
            headCount = nHeads,
            kvHeadCount = nKvHeads,
            feedForwardLength = hiddenDim,
            ropeDimensionCount = headDim,
            vocabSize = vocabSize
        )
    }

    /**
     * Parse acoustic model config. Falls back to backbone dimensions if not specified.
     */
    private fun parseAcousticModel(map: Map<String, String>): LlamaModelMetadata {
        // Acoustic model params may be nested or prefixed with "acoustic_model_"
        val dim = map.intOrNull("acoustic_model_dim")
            ?: map.intOrNull("dim") ?: 3072
        val nLayers = map.intOrNull("acoustic_model_n_layers") ?: 3
        val nHeads = map.intOrNull("acoustic_model_n_heads")
            ?: map.intOrNull("n_heads") ?: 32
        val nKvHeads = map.intOrNull("acoustic_model_n_kv_heads")
            ?: map.intOrNull("n_kv_heads") ?: 8
        val hiddenDim = map.intOrNull("acoustic_model_hidden_dim")
            ?: map.intOrNull("hidden_dim") ?: 9216
        val vocabSize = map.intOrNull("vocab_size") ?: 131072
        val headDim = map.intOrNull("acoustic_model_head_dim")
            ?: map.intOrNull("head_dim") ?: (dim / nHeads)

        return LlamaModelMetadata(
            architecture = "voxtral_tts_acoustic",
            embeddingLength = dim,
            contextLength = 65536,
            blockCount = nLayers,
            headCount = nHeads,
            kvHeadCount = nKvHeads,
            feedForwardLength = hiddenDim,
            ropeDimensionCount = headDim,
            vocabSize = vocabSize
        )
    }

    private fun parseCodec(map: Map<String, String>): VoxtralCodecMetadata {
        return VoxtralCodecMetadata(
            samplingRate = map.intOrNull("sampling_rate") ?: 24000,
            semanticCodebookSize = map.intOrNull("semantic_codebook_size") ?: 8192,
            acousticCodebookSize = map.intOrNull("acoustic_codebook_size") ?: 21
        )
    }

    private fun parseAudio(map: Map<String, String>): VoxtralAudioConfig {
        return VoxtralAudioConfig(
            semanticCodebookSize = map.intOrNull("semantic_codebook_size") ?: 8192,
            acousticCodebookSize = map.intOrNull("acoustic_codebook_size") ?: 21,
            nAcousticCodebooks = map.intOrNull("n_acoustic_codebook") ?: 36,
            samplingRate = map.intOrNull("sampling_rate") ?: 24000
        )
    }

    /**
     * Check if config indicates tied word embeddings (Voxtral uses tied embeddings).
     */
    public fun isTiedEmbeddings(json: String): Boolean {
        val map = parseJsonObject(json.trim())
        return map["tie_word_embeddings"] != "false"
    }

    // ========== Lightweight JSON parsing ==========

    private fun parseJsonObject(json: String): Map<String, String> {
        if (!json.startsWith("{") || !json.endsWith("}")) {
            error("params.json: expected JSON object")
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
        this[key]?.toIntOrNull() ?: error("params.json: missing or invalid '$key'")

    private fun Map<String, String>.intOrNull(key: String): Int? =
        this[key]?.toIntOrNull()
}
