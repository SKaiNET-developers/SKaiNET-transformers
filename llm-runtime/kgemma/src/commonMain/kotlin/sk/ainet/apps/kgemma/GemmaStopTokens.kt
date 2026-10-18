package sk.ainet.apps.kgemma

import sk.ainet.frontend.text.Tokenizer

/**
 * Resolves the full Gemma 4 stop-token set from a tokenizer.
 *
 * Gemma 4 checkpoints ship multiple stop ids — the released
 * `generation_config.json` lists `[1, 106, 50]` (`<eos>`, the turn-close
 * marker `<turn|>`, and the chat-end variant) — but a GGUF only carries the
 * single `tokenizer.ggml.eos_token_id`. Stopping on that one id alone lets
 * greedy decoding keep emitting `<turn|>` past the natural turn boundary
 * until the step budget runs out (the exact failure mode reported for the
 * E2B Q4_K_M checkpoint).
 *
 * Resolution order per marker: exact vocab lookup of the literal token
 * string via [Tokenizer.vocabulary]; when a marker is absent from this
 * vocab it is simply skipped. The tokenizer's own `eos` and every id in
 * `specialTokens.stopIds` (the family-decorated extra stops) are always
 * included. Works for any engine tokenizer since SKEEP-006, not only the
 * legacy GGUF-embedded one.
 */
public object GemmaStopTokens {

    /**
     * Literal stop-marker strings as they appear in the Gemma 4 vocab.
     * `<end_of_turn>` covers Gemma 2/3-lineage vocabs (FunctionGemma 270M
     * included) that share this loader path.
     */
    private val STOP_TOKEN_STRINGS = listOf(
        "<eos>",
        "<turn|>",
        "<end_of_turn>",
        "<|chat_end|>",
    )

    /**
     * The stop ids Gemma 4's released `generation_config.json` declares. Only
     * merged in when the vocab is positively identified as Gemma 4 (its
     * `<turn|>` marker resolves) — on other vocabs sharing this loader path
     * (e.g. FunctionGemma 270M) these raw ids could be ordinary tokens.
     */
    private val GEMMA4_GENERATION_CONFIG_STOP_IDS = setOf(1, 106, 50)

    public fun resolve(tokenizer: Tokenizer): Set<Int> {
        val ids = LinkedHashSet<Int>()
        ids.addAll(tokenizer.specialTokens.stopIds)
        val vocabulary = tokenizer.vocabulary
        for (token in STOP_TOKEN_STRINGS) {
            vocabulary.tokenToId(token)?.let { ids.add(it) }
        }
        if (vocabulary.tokenToId("<turn|>") != null) {
            ids.addAll(GEMMA4_GENERATION_CONFIG_STOP_IDS)
        }
        return ids
    }
}
