package sk.ainet.apps.kllama.chat

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put

/**
 * Chat template for Gemma 2 / 3 family models with native tool calling.
 *
 * Uses `<start_of_turn>` / `<end_of_turn>` markers.
 *
 * Tool definition format (system instruction):
 * ```json
 * [{"function_declarations": [{"name": "...", "description": "...", "parameters": {...}}]}]
 * ```
 *
 * Tool call output format (model response):
 * ```json
 * {"functionCall": {"name": "...", "args": {...}}}
 * ```
 *
 * Tool result format (user turn):
 * ```json
 * {"functionResponse": {"name": "...", "response": {...}}}
 * ```
 */
public class GemmaChatTemplate : ChatTemplate {

    private val json = Json { ignoreUnknownKeys = true }

    override fun apply(
        messages: List<ChatMessage>,
        tools: List<ToolDefinition>,
        addGenerationPrompt: Boolean
    ): String {
        val sb = StringBuilder()

        // Tool definitions as function declarations in a system-like preamble
        if (tools.isNotEmpty()) {
            sb.append("<start_of_turn>user\n")

            val toolsJson = buildJsonArray {
                add(buildJsonObject {
                    put("function_declarations", buildJsonArray {
                        for (tool in tools) {
                            add(buildJsonObject {
                                put("name", tool.name)
                                put("description", tool.description)
                                put("parameters", tool.parameters)
                            })
                        }
                    })
                })
            }
            sb.append(Json.encodeToString(toolsJson))
            sb.append("\n<end_of_turn>\n")
        }

        for (msg in messages) {
            when (msg.role) {
                ChatRole.SYSTEM -> {
                    // Gemma treats system as a user turn
                    sb.append("<start_of_turn>user\n")
                    sb.append(msg.content)
                    sb.append("\n<end_of_turn>\n")
                }
                ChatRole.USER -> {
                    sb.append("<start_of_turn>user\n")
                    sb.append(msg.content)
                    sb.append("\n<end_of_turn>\n")
                }
                ChatRole.ASSISTANT -> {
                    sb.append("<start_of_turn>model\n")
                    sb.append(msg.content)
                    sb.append("\n<end_of_turn>\n")
                }
                ChatRole.TOOL -> {
                    // Tool results are passed as a user turn with functionResponse wrapper
                    sb.append("<start_of_turn>user\n")
                    val responseJson = buildJsonObject {
                        put("functionResponse", buildJsonObject {
                            put("name", msg.toolCallId ?: "")
                            put("response", buildJsonObject {
                                put("result", msg.content)
                            })
                        })
                    }
                    sb.append(Json.encodeToString(JsonObject.serializer(), responseJson))
                    sb.append("\n<end_of_turn>\n")
                }
            }
        }

        if (addGenerationPrompt) {
            sb.append("<start_of_turn>model\n")
        }

        return sb.toString()
    }

    override fun parseToolCalls(text: String): List<ToolCall> {
        val calls = mutableListOf<ToolCall>()

        // Find each top-level JSON object that contains "functionCall"
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

    override fun containsToolCall(text: String): Boolean {
        return text.contains("\"functionCall\"")
    }

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
