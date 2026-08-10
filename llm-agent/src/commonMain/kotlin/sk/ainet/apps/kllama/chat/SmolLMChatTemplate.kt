package sk.ainet.apps.kllama.chat

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * Chat template for SmolLM2-Instruct family models with native tool calling.
 *
 * Wraps messages in ChatML envelopes (`<|im_start|>{role}\n{content}<|im_end|>`)
 * and follows the official model-card recipe for tool calling: tools are
 * embedded in the system prompt inside `<tools>...</tools>`, and the model is
 * instructed to emit a JSON **array** inside `<tool_call>...</tool_call>`.
 *
 * Tool definition format (system prompt):
 * ```
 * <tools>[
 *   {"name": "...", "description": "...", "parameters": {...}}
 * ]</tools>
 * ```
 *
 * Tool call output format (model response):
 * ```
 * <tool_call>[{"name": "...", "arguments": {...}}]</tool_call>
 * ```
 *
 * The system prompt is taken verbatim from
 * https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct so the model
 * sees the exact phrasing it was trained on.
 */
public class SmolLMChatTemplate : ChatTemplate {

    private val parserStrategy = SmolLMToolCallParserStrategy()

    override fun apply(
        messages: List<ChatMessage>,
        tools: List<ToolDefinition>,
        addGenerationPrompt: Boolean
    ): String {
        val sb = StringBuilder()

        if (tools.isNotEmpty()) {
            val explicitSystem = messages.firstOrNull { it.role == ChatRole.SYSTEM }
            sb.append("<|im_start|>system\n")
            sb.append(buildSystemPromptWithTools(explicitSystem?.content, tools))
            sb.append("<|im_end|>\n")

            for (msg in messages) {
                if (msg.role == ChatRole.SYSTEM) continue
                appendMessage(sb, msg)
            }
        } else {
            for (msg in messages) {
                appendMessage(sb, msg)
            }
        }

        if (addGenerationPrompt) {
            sb.append("<|im_start|>assistant\n")
        }

        return sb.toString()
    }

    override fun parseToolCalls(text: String): List<ToolCall> = parserStrategy.parse(text)

    override fun containsToolCall(text: String): Boolean = parserStrategy.containsToolCall(text)

    private fun appendMessage(sb: StringBuilder, msg: ChatMessage) {
        val role = when (msg.role) {
            ChatRole.TOOL -> "tool"
            else -> msg.role.roleName
        }
        sb.append("<|im_start|>").append(role).append("\n")
        sb.append(msg.content)
        sb.append("<|im_end|>\n")
    }

    private fun buildSystemPromptWithTools(
        userSystem: String?,
        tools: List<ToolDefinition>
    ): String {
        val toolsJson = buildJsonArray {
            for (tool in tools) {
                add(buildJsonObject {
                    put("name", tool.name)
                    put("description", tool.description)
                    put("parameters", tool.parameters)
                })
            }
        }
        val rendered = Json.encodeToString(toolsJson)
        return buildString {
            append(
                userSystem?.takeIf { it.isNotBlank() }
                    ?: "You are an expert in composing functions. " +
                    "You are given a question and a set of possible functions. " +
                    "Based on the question, you will need to make one or more function/tool calls " +
                    "to achieve the purpose. If none of the function can be used, point it out. " +
                    "If the given question lacks the parameters required by the function, also point it out."
            )
            append("\n\n")
            append("You have access to the following tools:\n")
            append("<tools>")
            append(rendered)
            append("</tools>\n\n")
            append("The output MUST strictly adhere to the following format, and NO other text MUST be included.\n")
            append("The example format is as follows. Please make sure the parameter type is correct. ")
            append("If no function call is needed, please make the tool calls an empty list '[]'.\n")
            append("<tool_call>[\n")
            append("{\"name\": \"func_name1\", \"arguments\": {\"argument1\": \"value1\", \"argument2\": \"value2\"}},\n")
            append("... (more tool calls as required)\n")
            append("]</tool_call>")
        }
    }
}
