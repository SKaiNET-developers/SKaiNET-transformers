package sk.ainet.apps.kllama.chat

/**
 * Strategy interface for parsing tool calls from model output.
 *
 * Implementations handle a specific output format (Hermes XML tags,
 * bare JSON, Gemma `functionCall`, etc.).  [ToolCallParser] delegates
 * to a chain of strategies so new formats can be added without
 * duplicating parsing logic.
 */
public interface ToolCallParserStrategy {

    /** Human-readable name of this format, e.g. "hermes", "llama31", "gemma". */
    public val formatName: String

    /** Try to parse tool calls from [text]. Return an empty list if the format does not match. */
    public fun parse(text: String): List<ToolCall>

    /** Quick check — does [text] likely contain a tool call in this format? */
    public fun containsToolCall(text: String): Boolean
}
