package sk.ainet.apps.kllama.chat

/**
 * Common metadata about a loaded model, usable across different loaders
 * (GGUF, HuggingFace, etc.) for capability detection such as tool-calling
 * support and chat template selection.
 *
 * All fields are optional so that partially-available metadata still works.
 *
 * @param family Model family identifier, e.g. "qwen", "llama", "gemma".
 * @param architecture Low-level architecture name, e.g. "llama", "gemma3n".
 * @param chatTemplate Raw chat_template string from GGUF or tokenizer config.
 * @param tokenizerHints Special tokens that hint at capabilities, e.g. `<tool_call>`.
 * @param sourceFormat Origin of the metadata, e.g. "gguf", "hf".
 */
public data class ModelMetadata(
    val family: String? = null,
    val architecture: String? = null,
    val chatTemplate: String? = null,
    val tokenizerHints: List<String> = emptyList(),
    val sourceFormat: String? = null
)
