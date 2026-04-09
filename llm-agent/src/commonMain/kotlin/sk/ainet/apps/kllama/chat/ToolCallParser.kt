package sk.ainet.apps.kllama.chat

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/**
 * Parses tool calls from model output text.
 *
 * Delegates to a chain of [ToolCallParserStrategy] instances. The built-in
 * strategies handle:
 * 1. **Hermes/ChatML style**: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
 * 2. **Llama 3.1 style**: Bare JSON `{"name": "...", "arguments": {...}}`
 *
 * Additional strategies (e.g. [GemmaToolCallParserStrategy]) can be used
 * through [parseWith] or by individual [ToolCallingSupport] providers.
 */
public object ToolCallParser {

    private val hermesStrategy = HermesToolCallParserStrategy()
    private val llama31Strategy = Llama31ToolCallParserStrategy()

    private val defaultStrategies: List<ToolCallParserStrategy> = listOf(
        hermesStrategy,
        llama31Strategy
    )

    private val json = Json { ignoreUnknownKeys = true }

    /**
     * Attempt to parse one or more tool calls from [text] using the default
     * strategies (Hermes, then Llama 3.1).
     *
     * @return List of parsed [ToolCall]s (empty if no tool calls were found).
     */
    public fun parse(text: String): List<ToolCall> = parseWith(text, defaultStrategies)

    /**
     * Parse tool calls using a custom list of strategies.
     * The first strategy that returns a non-empty result wins.
     */
    public fun parseWith(text: String, strategies: List<ToolCallParserStrategy>): List<ToolCall> {
        for (strategy in strategies) {
            val calls = strategy.parse(text)
            if (calls.isNotEmpty()) return calls
        }
        return emptyList()
    }

    /**
     * Check whether [text] appears to contain a tool call (default strategies).
     */
    public fun containsToolCall(text: String): Boolean =
        defaultStrategies.any { it.containsToolCall(text) }

    /**
     * Parse a single JSON object into a [ToolCall].
     *
     * Expects `{"name": "...", "arguments": {...}}` with an optional `"id"` field.
     * Returns `null` if parsing fails or the `"name"` field is missing.
     */
    public fun parseJsonToolCall(jsonStr: String): ToolCall? {
        return try {
            val obj = json.parseToJsonElement(jsonStr).jsonObject
            val name = obj["name"]?.jsonPrimitive?.content ?: return null
            val arguments = obj["arguments"]?.jsonObject ?: JsonObject(emptyMap())
            val id = obj["id"]?.jsonPrimitive?.content ?: generateCallId()
            ToolCall(id = id, name = name, arguments = arguments)
        } catch (_: Exception) {
            null
        }
    }

    private var callCounter = 0

    /** Generate a unique call ID for tool calls that don't include one. */
    public fun generateCallId(): String {
        return "call_${callCounter++}"
    }
}

// ---------------------------------------------------------------------------
// Built-in strategies
// ---------------------------------------------------------------------------

/** Parses `<tool_call>...</tool_call>` XML-wrapped JSON (Hermes / ChatML / Qwen). */
internal class HermesToolCallParserStrategy : ToolCallParserStrategy {

    override val formatName: String = "hermes"

    private val pattern = Regex("""<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>""")

    override fun parse(text: String): List<ToolCall> {
        val matches = buildList {
            var match = pattern.find(text)
            while (match != null) {
                add(match)
                match = pattern.find(text, match.range.last + 1)
            }
        }
        if (matches.isEmpty()) return emptyList()
        return matches.mapNotNull { ToolCallParser.parseJsonToolCall(it.groupValues[1]) }
    }

    override fun containsToolCall(text: String): Boolean =
        text.contains("<tool_call>")
}

/** Parses bare JSON objects with a `"name"` key (Llama 3.1 style). */
internal class Llama31ToolCallParserStrategy : ToolCallParserStrategy {

    override val formatName: String = "llama31"

    override fun parse(text: String): List<ToolCall> {
        val trimmed = text.trim()
        if (trimmed.startsWith("{") && trimmed.contains("\"name\"")) {
            val parsed = ToolCallParser.parseJsonToolCall(trimmed)
            if (parsed != null) return listOf(parsed)
        }
        return emptyList()
    }

    override fun containsToolCall(text: String): Boolean {
        val trimmed = text.trim()
        return trimmed.startsWith("{") && trimmed.contains("\"name\"") && trimmed.contains("\"arguments\"")
    }
}
