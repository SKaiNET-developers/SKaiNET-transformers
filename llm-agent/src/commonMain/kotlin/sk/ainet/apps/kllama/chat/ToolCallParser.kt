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
    private val llama3FunctionTagStrategy = Llama3FunctionTagParserStrategy()
    private val llama31Strategy = Llama31ToolCallParserStrategy()

    private val defaultStrategies: List<ToolCallParserStrategy> = listOf(
        hermesStrategy,
        llama3FunctionTagStrategy,
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
     * Accepts either `"arguments"` (Hermes / our internal shape) or `"parameters"`
     * (Meta's Llama 3.x docs) as the argument-object key. The `"id"` field is
     * optional and auto-generated when absent.
     *
     * Returns `null` if parsing fails or the `"name"` field is missing.
     */
    public fun parseJsonToolCall(jsonStr: String): ToolCall? {
        return try {
            val obj = json.parseToJsonElement(jsonStr).jsonObject
            val name = obj["name"]?.jsonPrimitive?.content ?: return null
            val arguments = obj["arguments"]?.jsonObject
                ?: obj["parameters"]?.jsonObject
                ?: JsonObject(emptyMap())
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

/**
 * Parses bare JSON objects with a `"name"` key (Llama 3.1 / 3.2 JSON tool
 * format — Meta's `text_prompt_format.md` for Llama 3.2 custom tools).
 *
 * Accepts:
 * - `{"name": "fn", "parameters": {...}}` — Llama 3.x docs
 * - `{"name": "fn", "arguments": {...}}` — Hermes / our internal shape
 *
 * Tolerates a leading `<|python_tag|>` marker (used in some Llama 3.2 variants
 * for the built-in tool calls; harmless on custom tools) and stray prose
 * before/after the JSON object — finds the first balanced `{...}` block that
 * starts the assistant turn after stripping the marker.
 */
internal class Llama31ToolCallParserStrategy : ToolCallParserStrategy {

    override val formatName: String = "llama3-json"

    override fun parse(text: String): List<ToolCall> {
        val candidate = stripPythonTag(text).trim()
        if (!candidate.startsWith("{") || !candidate.contains("\"name\"")) return emptyList()
        val firstObject = extractFirstJsonObject(candidate) ?: return emptyList()
        val parsed = ToolCallParser.parseJsonToolCall(firstObject) ?: return emptyList()
        return listOf(parsed)
    }

    override fun containsToolCall(text: String): Boolean {
        val candidate = stripPythonTag(text).trim()
        return candidate.startsWith("{") &&
            candidate.contains("\"name\"") &&
            (candidate.contains("\"arguments\"") || candidate.contains("\"parameters\""))
    }

    private fun stripPythonTag(text: String): String {
        val trimmed = text.trimStart()
        return if (trimmed.startsWith("<|python_tag|>")) trimmed.removePrefix("<|python_tag|>") else text
    }

    /** Find the first `{...}` block at the start of [text], respecting brace nesting and string literals. */
    private fun extractFirstJsonObject(text: String): String? {
        if (!text.startsWith("{")) return null
        var depth = 0
        var inString = false
        var escape = false
        for ((i, c) in text.withIndex()) {
            if (escape) {
                escape = false
                continue
            }
            if (inString) {
                when (c) {
                    '\\' -> escape = true
                    '"' -> inString = false
                }
                continue
            }
            when (c) {
                '"' -> inString = true
                '{' -> depth++
                '}' -> {
                    depth--
                    if (depth == 0) return text.substring(0, i + 1)
                }
            }
        }
        return null
    }
}

/**
 * Parses Llama 3.1 legacy `<function=name>{"arg": "value"}</function>` tool calls.
 *
 * This is the format `kllama --demo --template=llama3` emits when the
 * `Llama3ChatTemplate` is constructed with `Llama3ToolFormat.FUNCTION_TAG`.
 * Selectable rather than default because Meta's Llama 3.2 docs recommend
 * the bare-JSON format ([Llama31ToolCallParserStrategy]).
 */
internal class Llama3FunctionTagParserStrategy : ToolCallParserStrategy {

    override val formatName: String = "llama3-function-tag"

    private val pattern = Regex("""<function=([A-Za-z_][A-Za-z0-9_]*)>\s*(\{[\s\S]*?\})\s*</function>""")

    override fun parse(text: String): List<ToolCall> {
        val matches = pattern.findAll(text).toList()
        if (matches.isEmpty()) return emptyList()
        return matches.mapNotNull { m ->
            val name = m.groupValues[1]
            val args = m.groupValues[2]
            // Synthesize the JSON shape parseJsonToolCall expects so we get unified
            // id-generation + arguments/parameters handling.
            val synthesized = "{\"name\": \"$name\", \"arguments\": $args}"
            ToolCallParser.parseJsonToolCall(synthesized)
        }
    }

    override fun containsToolCall(text: String): Boolean = pattern.containsMatchIn(text)
}
