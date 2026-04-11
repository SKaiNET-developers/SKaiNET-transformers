package sk.ainet.apps.llm.tokenizer

import sk.ainet.apps.llm.Tokenizer
import sk.ainet.io.RandomAccessSource

/**
 * Unified factory for creating tokenizers from various sources.
 *
 * Usage:
 * ```kotlin
 * // From GGUF file (auto-detects BPE/SentencePiece/WordPiece)
 * val tokenizer = TokenizerFactory.fromGGUF(randomAccessSource)
 *
 * // From HuggingFace tokenizer.json
 * val tokenizer = TokenizerFactory.fromTokenizerJson(jsonString)
 * ```
 */
public object TokenizerFactory {

    /**
     * Create a tokenizer from GGUF file metadata (streaming, memory-efficient).
     * Auto-detects tokenizer type from vocabulary and metadata.
     *
     * @param source Random access source to the GGUF file.
     * @param debug Print debug information during loading.
     */
    public fun fromGGUF(source: RandomAccessSource, debug: Boolean = false): GGUFTokenizer {
        return GGUFTokenizer.fromRandomAccessSource(source, debug)
    }

    /**
     * Create a tokenizer from a HuggingFace `tokenizer.json` string.
     * Parses vocab and BPE merges from the JSON.
     *
     * @param json The tokenizer.json content.
     * @param debug Print debug information during loading.
     */
    public fun fromTokenizerJson(json: String, debug: Boolean = false): GGUFTokenizer {
        return GGUFTokenizer.fromTokenizerJson(json, debug)
    }

    /**
     * Create a HuggingFace BPE tokenizer from `tokenizer.json` + optional config.
     *
     * @param tokenizerJson Content of `tokenizer.json`.
     * @param tokenizerConfigJson Optional content of `tokenizer_config.json`.
     */
    public fun fromHuggingFace(tokenizerJson: String, tokenizerConfigJson: String? = null): Tokenizer {
        return createHuggingFaceBPETokenizerFromJson(tokenizerJson, tokenizerConfigJson)
    }
}
