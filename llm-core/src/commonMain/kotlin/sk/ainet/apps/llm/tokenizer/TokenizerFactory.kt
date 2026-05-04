package sk.ainet.apps.llm.tokenizer

import sk.ainet.apps.llm.Tokenizer
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.gguf.StreamingGGUFReader

/**
 * Unified factory for creating tokenizers from various sources.
 *
 * Per-architecture dispatch routes Qwen / GPT-2 / Mistral-Nemo (byte-level
 * BPE) to upstream [sk.ainet.io.tokenizer.QwenByteLevelBpeTokenizer] —
 * the local [GGUFTokenizer]'s `encodeBPE` is the greedy-by-score
 * SentencePiece algorithm, wrong for byte-level BPE which needs
 * `bytes_to_unicode` mapping + GPT-2 pretokenization regex +
 * merge-rank-based merging. See #52.
 *
 * Three entry points:
 *
 * - [fromGgufFields] — preferred when GGUF metadata has already been
 *   parsed (e.g. via `UnifiedModelLoader.peek`); always routes through
 *   the upstream `sk.ainet.io.tokenizer.TokenizerFactory`.
 * - [fromGGUF] — convenience for a raw GGUF [RandomAccessSource]. Reads
 *   metadata once and dispatches gpt2/bpe → upstream byte-BPE; else →
 *   local SentencePiece path.
 * - [fromTokenizerJson] / [fromHuggingFace] — HuggingFace `tokenizer.json`.
 */
public object TokenizerFactory {

    /**
     * Create a tokenizer from a pre-parsed GGUF metadata field map.
     *
     * Delegates to upstream `sk.ainet.io.tokenizer.TokenizerFactory.fromGguf`,
     * which has the correct byte-level BPE for Qwen/GPT-2 (issue #52). The
     * result is wrapped in an adapter so callers continue to see the local
     * [Tokenizer] interface (with `decode(Int)` and non-null bos/eos).
     *
     * @param fields The GGUF metadata map, e.g. `GGUFModelInfo.fields` from
     *   `UnifiedModelLoader.peek`.
     */
    public fun fromGgufFields(fields: Map<String, Any?>): Tokenizer {
        val upstream = sk.ainet.io.tokenizer.TokenizerFactory.fromGguf(fields)
        return UpstreamTokenizerAdapter(upstream)
    }

    /**
     * Create a tokenizer from GGUF file metadata (streaming, memory-efficient).
     *
     * Dispatches on `tokenizer.ggml.model`:
     * - `gpt2` / `bpe` → upstream [sk.ainet.io.tokenizer.QwenByteLevelBpeTokenizer]
     *   (correct byte-level BPE, matches HuggingFace transformers / llama.cpp)
     * - everything else → local [GGUFTokenizer] (SentencePiece path)
     *
     * Prefer [fromGgufFields] when metadata is already in hand — avoids
     * re-reading the source.
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

