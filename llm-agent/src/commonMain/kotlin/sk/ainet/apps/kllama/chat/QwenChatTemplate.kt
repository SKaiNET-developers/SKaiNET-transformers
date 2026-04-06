package sk.ainet.apps.kllama.chat

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * Chat template for Qwen 3 / 3.5 family models with native tool calling.
 *
 * Uses ChatML tokens (`<|im_start|>`/`<|im_end|>`) but formats tool definitions
 * in Qwen's expected function-schema style rather than the Hermes convention.
 *
 * Tool definition format (system prompt):
 * ```json
 * [
 *   {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
 * ]
 * ```
 *
 * Tool call output format (model response):
 * ```
 * <tool_call>
 * {"name": "...", "arguments": {...}}
 * </tool_call>
 * ```
 */
public class QwenChatTemplate : ChatTemplate {

    override fun apply(
        messages: List<ChatMessage>,
        tools: List<ToolDefinition>,
        addGenerationPrompt: Boolean
    ): String {
        val sb = StringBuilder()

        // If tools are provided, inject a system message with Qwen-style tool definitions
        if (tools.isNotEmpty()) {
            sb.append("<|im_start|>system\n")
            sb.append("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n")
            sb.append("# Tools\n\n")
            sb.append("You may call one or more functions to assist with the user query.\n\n")
            sb.append("You are provided with function signatures within <tools></tools> XML tags:\n")
            sb.append("<tools>\n")

            val toolsArray = buildJsonArray {
                for (tool in tools) {
                    add(buildJsonObject {
                        put("type", "function")
                        put("function", buildJsonObject {
                            put("name", tool.name)
                            put("description", tool.description)
                            put("parameters", tool.parameters)
                        })
                    })
                }
            }
            sb.append(Json.encodeToString(toolsArray))
            sb.append("\n</tools>\n\n")

            sb.append("For each function call, return a JSON object with function name and arguments ")
            sb.append("within <tool_call></tool_call> XML tags:\n")
            sb.append("<tool_call>\n")
            sb.append("{\"name\": <function-name>, \"arguments\": <args-json-object>}\n")
            sb.append("</tool_call>")
            sb.append("<|im_end|>\n")
        }

        for (msg in messages) {
            val role = when (msg.role) {
                ChatRole.TOOL -> "tool"
                else -> msg.role.roleName
            }
            sb.append("<|im_start|>").append(role).append("\n")
            sb.append(msg.content)
            sb.append("<|im_end|>\n")
        }

        if (addGenerationPrompt) {
            sb.append("<|im_start|>assistant\n")
        }

        return sb.toString()
    }
}
