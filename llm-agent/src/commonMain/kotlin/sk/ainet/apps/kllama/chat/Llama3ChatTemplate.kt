package sk.ainet.apps.kllama.chat

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * Chat template for the Llama 3 / 3.1 / 3.2 family.
 *
 * Without tools the template emits the standard Llama 3 turn structure:
 *
 * ```
 * <|begin_of_text|><|start_header_id|>system<|end_header_id|>
 *
 * {system content}<|eot_id|>
 * <|start_header_id|>user<|end_header_id|>
 *
 * {user content}<|eot_id|>
 * <|start_header_id|>assistant<|end_header_id|>
 *
 * ```
 *
 * With tools the template emits Meta's documented tool-calling system
 * prompt — see [Llama3ToolFormat] and `docs/llama3-tool-calling.md`. The
 * shape of the system message *must* match the response format the
 * parser expects, so the constructor takes a [format] that is the
 * single source of truth for both.
 *
 * @param format Response format the model should emit when calling a tool.
 *   Defaults to [Llama3ToolFormat.JSON] (Llama 3.2 default).
 */
public class Llama3ChatTemplate(
    private val format: Llama3ToolFormat = Llama3ToolFormat.JSON
) : ChatTemplate {

    override fun apply(
        messages: List<ChatMessage>,
        tools: List<ToolDefinition>,
        addGenerationPrompt: Boolean
    ): String {
        val sb = StringBuilder()
        sb.append("<|begin_of_text|>")

        val toolSystem = if (tools.isNotEmpty()) buildToolSystemPrompt(tools) else null

        // Merge any user-supplied system message with the tool-system block.
        val mergedMessages = mergeToolSystem(messages, toolSystem)

        for (msg in mergedMessages) {
            val role = when (msg.role) {
                ChatRole.TOOL -> "tool"
                else -> msg.role.roleName
            }
            sb.append("<|start_header_id|>").append(role).append("<|end_header_id|>\n\n")
            sb.append(msg.content)
            sb.append("<|eot_id|>")
        }

        if (addGenerationPrompt) {
            sb.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        }

        return sb.toString()
    }

    private fun buildToolSystemPrompt(tools: List<ToolDefinition>): String {
        val toolsJson = tools.joinToString("\n\n") { tool ->
            val obj = buildJsonObject {
                put("name", tool.name)
                put("description", tool.description)
                put("parameters", tool.parameters)
            }
            Json.encodeToString(JsonObject.serializer(), obj)
        }

        return when (format) {
            Llama3ToolFormat.JSON -> buildString {
                append("You are a helpful assistant with tool calling capabilities.\n")
                append("When you receive a tool call response, use the output to format an answer to the original user question.\n")
                append("\n")
                append("You have access to the following functions:\n\n")
                append(toolsJson)
                append("\n\n")
                append("If you choose to call a function, your reply MUST be a single JSON object on one line in the following format and nothing else:\n")
                append("{\"name\": <function-name>, \"parameters\": <arguments-object>}\n")
                append("Do not write the function definition. Do not include any prose. Do not use variables.")
            }
            Llama3ToolFormat.FUNCTION_TAG -> buildString {
                append("You are a helpful assistant with tool calling capabilities.\n")
                append("When you receive a tool call response, use the output to format an answer to the original user question.\n")
                append("\n")
                append("You have access to the following functions:\n\n")
                append(toolsJson)
                append("\n\n")
                append("If you choose to call a function, ONLY reply in the format below and nothing else:\n")
                append("<function=function_name>{\"arg_name\": \"arg_value\"}</function>\n")
                append("Function calls MUST be on a single line. Required parameters MUST be specified.")
            }
        }
    }

    /**
     * Prepend the tool-system block to an existing system message (if any),
     * otherwise insert it as the first message. This preserves any user-set
     * system prompt instead of clobbering it.
     */
    private fun mergeToolSystem(
        messages: List<ChatMessage>,
        toolSystem: String?
    ): List<ChatMessage> {
        if (toolSystem == null) return messages
        val firstIsSystem = messages.firstOrNull()?.role == ChatRole.SYSTEM
        return if (firstIsSystem) {
            val first = messages.first()
            val merged = ChatMessage(
                role = ChatRole.SYSTEM,
                content = toolSystem + "\n\n" + first.content
            )
            listOf(merged) + messages.drop(1)
        } else {
            listOf(ChatMessage(role = ChatRole.SYSTEM, content = toolSystem)) + messages
        }
    }
}
