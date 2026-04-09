package sk.ainet.apps.kllama.chat

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/**
 * Parses Gemma-style tool calls that use `functionCall` JSON format:
 * ```json
 * {"functionCall": {"name": "...", "args": {...}}}
 * ```
 */
public class GemmaToolCallParserStrategy : ToolCallParserStrategy {

    override val formatName: String = "gemma"

    private val json = Json { ignoreUnknownKeys = true }

    override fun parse(text: String): List<ToolCall> {
        val calls = mutableListOf<ToolCall>()
        var i = 0
        while (i < text.length) {
            val start = text.indexOf('{', i)
            if (start == -1) break
            val jsonStr = extractJsonObject(text, start)
            if (jsonStr != null && jsonStr.contains("\"functionCall\"")) {
                val call = parseFunctionCall(jsonStr)
                if (call != null) calls.add(call)
                i = start + jsonStr.length
            } else {
                i = start + 1
            }
        }
        return calls
    }

    override fun containsToolCall(text: String): Boolean =
        text.contains("\"functionCall\"")

    private fun extractJsonObject(text: String, start: Int): String? {
        if (start >= text.length || text[start] != '{') return null
        var depth = 0
        var inString = false
        var escape = false
        for (i in start until text.length) {
            val c = text[i]
            if (escape) { escape = false; continue }
            if (c == '\\' && inString) { escape = true; continue }
            if (c == '"') { inString = !inString; continue }
            if (inString) continue
            if (c == '{') depth++
            else if (c == '}') { depth--; if (depth == 0) return text.substring(start, i + 1) }
        }
        return null
    }

    private fun parseFunctionCall(jsonStr: String): ToolCall? {
        return try {
            val obj = json.parseToJsonElement(jsonStr).jsonObject
            val functionCall = obj["functionCall"]?.jsonObject ?: return null
            val name = functionCall["name"]?.jsonPrimitive?.content ?: return null
            val args = functionCall["args"]?.jsonObject ?: JsonObject(emptyMap())
            ToolCall(
                id = ToolCallParser.generateCallId(),
                name = name,
                arguments = args
            )
        } catch (_: Exception) {
            null
        }
    }
}
