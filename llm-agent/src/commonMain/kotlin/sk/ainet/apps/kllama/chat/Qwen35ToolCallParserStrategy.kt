package sk.ainet.apps.kllama.chat

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject

/**
 * Parses Qwen3.5's canonical tool-call format:
 *
 * ```
 * <tool_call>
 * <function=get_weather>
 * <parameter=city>San Francisco</parameter>
 * <parameter=unit>celsius</parameter>
 * </function>
 * </tool_call>
 * ```
 *
 * This format is emitted by the Qwen3.5 HF chat template and differs from
 * the Hermes JSON-in-XML format used by Qwen3 and earlier.
 */
internal class Qwen35ToolCallParserStrategy : ToolCallParserStrategy {

    override val formatName: String = "qwen35-xml"

    private val toolCallBlock = Regex("""<tool_call>\s*([\s\S]*?)\s*</tool_call>""")
    private val functionHeader = Regex("""<function=([^>\s]+)>\s*""")
    private val paramBlock = Regex("""<parameter=([^>\s]+)>([\s\S]*?)</parameter>""")

    private val json = Json { ignoreUnknownKeys = true }

    override fun containsToolCall(text: String): Boolean =
        text.contains("<tool_call>") && text.contains("<function=")

    override fun parse(text: String): List<ToolCall> {
        val results = mutableListOf<ToolCall>()

        var match = toolCallBlock.find(text)
        while (match != null) {
            val inner = match.groupValues[1]
            val fn = functionHeader.find(inner)
            if (fn != null) {
                val fnName = fn.groupValues[1]
                val args: JsonObject = buildJsonObject {
                    for (p in paramBlock.findAll(inner)) {
                        val key = p.groupValues[1]
                        val value = p.groupValues[2].trim()
                        put(key, tryParseJsonValue(value))
                    }
                }
                results.add(
                    ToolCall(
                        id = ToolCallParser.generateCallId(),
                        name = fnName,
                        arguments = args
                    )
                )
            }
            match = toolCallBlock.find(text, match.range.last + 1)
        }

        return results
    }

    /**
     * Attempt to parse [value] as a JSON primitive (number, boolean, null)
     * or JSON structure (object/array). Falls back to a plain string.
     */
    private fun tryParseJsonValue(value: String): kotlinx.serialization.json.JsonElement {
        if (value.isEmpty()) return JsonPrimitive(value)

        // Try parsing as a JSON element (object, array, number, boolean, null)
        return try {
            json.parseToJsonElement(value)
        } catch (_: Exception) {
            JsonPrimitive(value)
        }
    }
}
