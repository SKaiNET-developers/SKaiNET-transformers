package sk.ainet.apps.kllama.chat

/**
 * Parses SmolLM2-style tool calls.
 *
 * SmolLM2's official recipe emits a JSON **array** inside the `<tool_call>`
 * markers:
 *
 * ```
 * <tool_call>[{"name": "...", "arguments": {...}}, ...]</tool_call>
 * ```
 *
 * The array form is what the model card prescribes. Small models drift,
 * however, so this strategy also accepts the bare-object form
 * `<tool_call>{...}</tool_call>` as a fallback — if the model omits the
 * outer brackets, we still recover the call. Note: keeping this distinct
 * from [HermesToolCallParserStrategy] preserves Qwen / generic ChatML
 * behavior, which expects the object-only form.
 */
internal class SmolLMToolCallParserStrategy : ToolCallParserStrategy {

    override val formatName: String = "smollm"

    private val arrayPattern = Regex("""<tool_call>\s*(\[[\s\S]*?\])\s*</tool_call>""")
    private val objectPattern = Regex("""<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>""")

    override fun parse(text: String): List<ToolCall> {
        val arrayCalls = parseArrayForm(text)
        if (arrayCalls.isNotEmpty()) return arrayCalls
        return parseObjectForm(text)
    }

    override fun containsToolCall(text: String): Boolean = text.contains("<tool_call>")

    private fun parseArrayForm(text: String): List<ToolCall> {
        val matches = arrayPattern.findAll(text).toList()
        if (matches.isEmpty()) return emptyList()
        val calls = mutableListOf<ToolCall>()
        for (match in matches) {
            val arrayJson = match.groupValues[1]
            calls += splitJsonArrayElements(arrayJson)
                .mapNotNull { ToolCallParser.parseJsonToolCall(it) }
        }
        return calls
    }

    private fun parseObjectForm(text: String): List<ToolCall> {
        val matches = objectPattern.findAll(text).toList()
        if (matches.isEmpty()) return emptyList()
        return matches.mapNotNull { ToolCallParser.parseJsonToolCall(it.groupValues[1]) }
    }

    /**
     * Split the inside of a JSON array into top-level element strings without
     * a JSON parser. Object boundaries are tracked via brace depth, ignoring
     * braces inside string literals (with backslash escapes).
     *
     * Why hand-rolled: the strategy chain prefers raw-text inputs and we don't
     * want to fail the whole parse if one element is malformed — splitting
     * lets [ToolCallParser.parseJsonToolCall] handle each element independently.
     */
    private fun splitJsonArrayElements(arrayJson: String): List<String> {
        val trimmed = arrayJson.trim()
        if (!trimmed.startsWith("[") || !trimmed.endsWith("]")) return emptyList()
        val body = trimmed.substring(1, trimmed.length - 1)

        val elements = mutableListOf<String>()
        var depth = 0
        var inString = false
        var escaped = false
        var start = 0

        for (i in body.indices) {
            val c = body[i]
            if (escaped) {
                escaped = false
                continue
            }
            if (inString) {
                when (c) {
                    '\\' -> escaped = true
                    '"' -> inString = false
                }
                continue
            }
            when (c) {
                '"' -> inString = true
                '{', '[' -> depth++
                '}', ']' -> depth--
                ',' -> if (depth == 0) {
                    val element = body.substring(start, i).trim()
                    if (element.isNotEmpty()) elements += element
                    start = i + 1
                }
            }
        }
        val tail = body.substring(start).trim()
        if (tail.isNotEmpty()) elements += tail
        return elements
    }
}
