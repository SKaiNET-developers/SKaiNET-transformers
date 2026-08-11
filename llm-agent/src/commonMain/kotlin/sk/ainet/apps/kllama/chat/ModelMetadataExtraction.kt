package sk.ainet.apps.kllama.chat

/**
 * Best-effort extraction of [ModelMetadata] from loader-side sources (#37).
 *
 * The functions here are pure and format-agnostic at the type level: callers
 * hand in whatever they already have (a GGUF key/value field map, the text of
 * a HuggingFace `tokenizer_config.json`), and get back a [ModelMetadata]
 * suitable for [ToolCallingSupportResolver] auto-detection.
 *
 * Extraction is **best-effort only** — missing or malformed input degrades to
 * empty/partial metadata, never to an exception. Metadata is a hint for
 * template/provider selection, not authoritative proof of tool-calling
 * capability (see #35).
 */
public object ModelMetadataExtraction {

    /**
     * Special-token markers that hint at tool-calling capability when they
     * appear in a chat template or the tokenizer vocabulary.
     */
    private val TOOL_HINT_MARKERS: List<String> = listOf(
        "<tool_call>",
        "<tool_response>",
        "<|tool_call|>",
        "<|python_tag|>",
        "[AVAILABLE_TOOLS]",
        "<start_of_turn>",
        "<|im_start|>",
        "<|start_header_id|>"
    )

    /**
     * Build a [ModelMetadata] from a GGUF key/value field map
     * (e.g. `StreamingGGUFReader.fields`) without loading any tensor data.
     *
     * Reads (all optional):
     * - `general.architecture` → [ModelMetadata.architecture] and the derived [ModelMetadata.family]
     * - `tokenizer.chat_template` → [ModelMetadata.chatTemplate]
     * - `tokenizer.ggml.tokens` (if present as a string collection) → scanned for
     *   known tool/chat special tokens to populate [ModelMetadata.tokenizerHints]
     */
    public fun fromGgufFields(fields: Map<String, Any?>): ModelMetadata {
        val arch = fields["general.architecture"] as? String
        val chatTemplate = fields["tokenizer.chat_template"] as? String
        val tokens = (fields["tokenizer.ggml.tokens"] as? Collection<*>)
            ?.filterIsInstance<String>()
        return ModelMetadata(
            family = familyFromArchitecture(arch),
            architecture = arch ?: "unknown",
            chatTemplate = chatTemplate,
            tokenizerHints = tokenizerHints(chatTemplate, tokens),
            sourceFormat = "gguf"
        )
    }

    /**
     * Map a GGUF `general.architecture` string to a coarse model-family
     * identifier understood by the built-in [ToolCallingSupport] providers.
     *
     * Unknown architectures pass through unchanged so custom providers can
     * still match on them; `null` stays `null`.
     */
    public fun familyFromArchitecture(architecture: String?): String? {
        val arch = architecture?.lowercase() ?: return null
        return when {
            arch.startsWith("qwen") -> "qwen"
            arch.startsWith("gemma") -> "gemma"
            arch == "llama" -> "llama"
            else -> arch
        }
    }

    /**
     * Collect tool/chat special-token hints from a chat template and/or the
     * tokenizer vocabulary. Order is stable (marker declaration order),
     * duplicates removed.
     */
    public fun tokenizerHints(
        chatTemplate: String?,
        vocabTokens: Collection<String>? = null
    ): List<String> {
        if (chatTemplate == null && vocabTokens.isNullOrEmpty()) return emptyList()
        val vocab: Set<String> = vocabTokens?.let { tokens ->
            // Only special-token-shaped entries matter; avoids holding a
            // 100k+ vocab set just to probe eight markers.
            tokens.filterTo(mutableSetOf()) { it.startsWith('<') || it.startsWith('[') }
        } ?: emptySet()
        return TOOL_HINT_MARKERS.filter { marker ->
            (chatTemplate?.contains(marker) == true) || marker in vocab
        }
    }
}
