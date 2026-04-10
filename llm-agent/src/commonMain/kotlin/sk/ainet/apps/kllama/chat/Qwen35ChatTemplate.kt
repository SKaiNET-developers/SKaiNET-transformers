package sk.ainet.apps.kllama.chat

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * Chat template for Qwen3.5 family models with native tool calling.
 *
 * Uses ChatML tokens (`<|im_start|>`/`<|im_end|>`) like [QwenChatTemplate], but
 * instructs the model to emit tool calls in Qwen3.5's canonical format:
 *
 * ```
 * <tool_call>
 * <function=get_weather>
 * <parameter=city>San Francisco</parameter>
 * </function>
 * </tool_call>
 * ```
 *
 * This differs from [QwenChatTemplate] which expects JSON inside `<tool_call>` tags.
 * The format matches the Qwen3.5 HF `tokenizer_config.json` chat template.
 */
public class Qwen35ChatTemplate : ChatTemplate {

    private val parser = Qwen35ToolCallParserStrategy()

    override fun apply(
        messages: List<ChatMessage>,
        tools: List<ToolDefinition>,
        addGenerationPrompt: Boolean
    ): String {
        val sb = StringBuilder()

        // If tools are provided, inject a system message with Qwen3.5-style tool definitions
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

            sb.append("For each function call, return the function name and arguments ")
            sb.append("within <tool_call></tool_call> XML tags using the following format:\n")
            sb.append("<tool_call>\n")
            sb.append("<function=function_name>\n")
            sb.append("<parameter=param_name>value</parameter>\n")
            sb.append("</function>\n")
            sb.append("</tool_call>")
            sb.append("<|im_end|>\n")
        }

        for (msg in messages) {
            when (msg.role) {
                ChatRole.TOOL -> {
                    sb.append("<|im_start|>tool\n")
                    sb.append("<tool_response>\n")
                    sb.append(msg.content)
                    sb.append("\n</tool_response>")
                    sb.append("<|im_end|>\n")
                }
                else -> {
                    sb.append("<|im_start|>").append(msg.role.roleName).append("\n")
                    sb.append(msg.content)
                    sb.append("<|im_end|>\n")
                }
            }
        }

        if (addGenerationPrompt) {
            sb.append("<|im_start|>assistant\n")
        }

        return sb.toString()
    }

    override fun parseToolCalls(text: String): List<ToolCall> =
        parser.parse(text)

    override fun containsToolCall(text: String): Boolean =
        parser.containsToolCall(text)
}
