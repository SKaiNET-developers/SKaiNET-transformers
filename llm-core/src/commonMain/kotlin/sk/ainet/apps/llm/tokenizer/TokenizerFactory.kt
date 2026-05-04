package sk.ainet.apps.llm.tokenizer

import sk.ainet.apps.llm.Tokenizer
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.gguf.StreamingGGUFReader

/**
 * Unified factory for creating tokenizers from various sources. Delegates
 * the byte-level BPE (Qwen/GPT-2) and SentencePiece (LLaMA/Gemma) algorithms
 * to upstream `sk.ainet.io.tokenizer.*` and layers the
 * [SentencePieceSpecialTokens] decorator on top of SP-family results so
 * chat-template markers (`<bos>`, `<|turn>`, etc.) encode atomically.
 *
 * Usage:
 * ```kotlin
 * // From an already-parsed GGUF metadata map (cheapest — no extra file open):
 * val info = UnifiedModelLoader.peek { sourceProvider() }
 * val tokenizer = TokenizerFactory.fromGgufFields(info.fields)
 *
 * // From a GGUF RandomAccessSource (parses metadata via StreamingGGUFReader):
 * val tokenizer = TokenizerFactory.fromGgufSource(source)
 *
 * // From a HuggingFace tokenizer.json string (+ optional tokenizer_config.json):
 * val tokenizer = TokenizerFactory.fromTokenizerJsonString(json, configJson)
 * ```
 */
public object TokenizerFactory {

    /**
     * Create a tokenizer from a pre-parsed GGUF metadata field map.
     *
     * Routes byte-level BPE through upstream directly; routes SentencePiece
     * through [SentencePieceSpecialTokens] so chat-template control / user-
     * defined tokens encode atomically (Gemma 4, etc.).
     *
     * @param fields The GGUF metadata map, e.g. `GGUFModelInfo.fields` from
     *   `UnifiedModelLoader.peek`.
     */
    public fun fromGgufFields(fields: Map<String, Any?>): Tokenizer {
        val model = (fields["tokenizer.ggml.model"] as? String)?.lowercase()
        return when (model) {
            "llama", "sentencepiece" -> SentencePieceSpecialTokens.fromGgufFields(fields)
            else -> {
                val upstream = sk.ainet.io.tokenizer.TokenizerFactory.fromGguf(fields)
                UpstreamTokenizerAdapter(upstream)
            }
        }
    }

    /**
     * Convenience: open a [StreamingGGUFReader] over [source], parse metadata,
     * and build a tokenizer via [fromGgufFields]. The reader is closed before
     * returning. Use this when you only have the source handle and don't need
     * the rest of the GGUF metadata.
     */
    public fun fromGgufSource(source: RandomAccessSource): Tokenizer {
        return StreamingGGUFReader.open(source).use { reader ->
            fromGgufFields(reader.fields)
        }
    }

    /**
     * Create a tokenizer from a HuggingFace `tokenizer.json` string (+ optional
     * `tokenizer_config.json`).
     *
     * Routes BPE through upstream directly; routes Unigram / SentencePiece
     * through [SentencePieceSpecialTokens] so the `added_tokens` registry
     * (including bos/eos resolution) and `add_space_prefix` detection are
     * applied — both are upstream gaps that affect Gemma 4 SafeTensors.
     */
    public fun fromTokenizerJsonString(json: String, configJson: String? = null): Tokenizer {
        // Detect model.type with a minimal substring scan so we don't have to
        // round-trip through kotlinx.serialization twice (once here, once in
        // the dispatched factory).
        val modelType = detectModelType(json)
        return when (modelType) {
            "Unigram" -> SentencePieceSpecialTokens.fromTokenizerJson(json, configJson)
            else -> {
                val upstream = sk.ainet.io.tokenizer.TokenizerFactory.fromTokenizerJson(json)
                UpstreamTokenizerAdapter(upstream)
            }
        }
    }

    private fun detectModelType(json: String): String? {
        val modelIdx = json.indexOf("\"model\"")
        if (modelIdx < 0) return null
        val typeIdx = json.indexOf("\"type\"", modelIdx)
        if (typeIdx < 0) return null
        val colonIdx = json.indexOf(':', typeIdx)
        if (colonIdx < 0) return null
        val openQuote = json.indexOf('"', colonIdx)
        if (openQuote < 0) return null
        val closeQuote = json.indexOf('"', openQuote + 1)
        if (closeQuote < 0) return null
        return json.substring(openQuote + 1, closeQuote)
    }
}
