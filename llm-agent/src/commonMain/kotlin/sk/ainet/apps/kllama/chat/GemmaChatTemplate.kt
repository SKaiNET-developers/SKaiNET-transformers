package sk.ainet.apps.kllama.chat

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
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

    private val gemmaParser = GemmaToolCallParserStrategy()

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

    override fun parseToolCalls(text: String): List<ToolCall> =
        gemmaParser.parse(text)

    override fun containsToolCall(text: String): Boolean =
        gemmaParser.containsToolCall(text)
}
