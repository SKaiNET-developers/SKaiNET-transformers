package sk.ainet.apps.kllama.chat

/**
 * Describes how a model supports tool calling.
 *
 * - [NATIVE]: The model family has a dedicated, well-tested implementation.
 * - [GENERIC]: Best-effort fallback using a common format (e.g. ChatML).
 * - [UNSUPPORTED]: Tool calling is not viable for this model.
 */
public enum class ToolCallingMode {
    NATIVE,
    GENERIC,
    UNSUPPORTED
}
