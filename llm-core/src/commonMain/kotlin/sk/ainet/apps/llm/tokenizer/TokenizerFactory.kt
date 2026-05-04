package sk.ainet.apps.llm.tokenizer

import sk.ainet.apps.llm.Tokenizer
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.gguf.StreamingGGUFReader

/**
 * Unified factory for creating tokenizers from various sources.
 *
 * Per-architecture dispatch: a Qwen / GPT-2 / Mistral-Nemo model needs
 * byte-level BPE (correctly implemented in upstream
 * [sk.ainet.io.tokenizer.QwenByteLevelBpeTokenizer]); a Llama / Gemma /
 * TinyLlama model uses SentencePiece via the local [GGUFTokenizer].
 *
 * The split is because the local [GGUFTokenizer]'s `encodeBPE` is the
 * greedy-by-score SentencePiece algorithm — wrong for byte-level BPE,
 * which needs `bytes_to_unicode` mapping + GPT-2 pretokenization regex
 * + merge-rank-based merging. See #52.
 */
public object TokenizerFactory {

    /**
     * Create a tokenizer from GGUF file metadata (streaming, memory-efficient).
     *
     * Dispatches on `tokenizer.ggml.model`:
     * - `gpt2` / `bpe` → upstream [sk.ainet.io.tokenizer.QwenByteLevelBpeTokenizer]
     *   (correct byte-level BPE, matches HuggingFace transformers / llama.cpp)
     * - everything else → local [GGUFTokenizer] (SentencePiece path)
     *
     * @param source Random access source to the GGUF file.
     * @param debug Print debug information during loading.
     */
    public fun fromGGUF(source: RandomAccessSource, debug: Boolean = false): Tokenizer {
        return StreamingGGUFReader.open(source).use { reader ->
            val fields = reader.fields
            val model = (fields["tokenizer.ggml.model"] as? String)?.lowercase()
            when (model) {
                "gpt2", "bpe" -> {
                    if (debug) println("TokenizerFactory: model=$model → upstream QwenByteLevelBpeTokenizer")
                    val upstream = sk.ainet.io.tokenizer.TokenizerFactory.fromGguf(fields)
                    UpstreamTokenizerAdapter(upstream)
                }
                else -> {
                    if (debug) println("TokenizerFactory: model=$model → local GGUFTokenizer")
                    GGUFTokenizer.fromStreamingFields(fields, debug)
                }
            }
        }
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
