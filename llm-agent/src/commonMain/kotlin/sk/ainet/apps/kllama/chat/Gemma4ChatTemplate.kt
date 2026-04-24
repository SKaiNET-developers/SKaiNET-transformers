package sk.ainet.apps.kllama.chat

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put

/**
 * Chat template for Gemma 4 family models with native tool calling.
 *
 * Uses `<|turn>` / `<turn|>` markers (different from Gemma 2/3's `<start_of_turn>` / `<end_of_turn>`).
 *
 * Key differences from Gemma 2/3:
 * - Native system role support (system messages use `<|turn>system` instead of being mapped to user)
 * - Tool definitions wrapped in `<|tool>...<tool|>` delimiters
 * - Tool calls use `<|tool_call>...<tool_call|>` delimiters
 * - Tool responses use `<|tool_response>...<tool_response|>` delimiters
 * - Thinking mode: `<|think>...<think|>` blocks in model output (same paired-
 *   delimiter convention as the other markers). These are reasoning traces
 *   the model is allowed to emit; the agent loop consumes them separately
 *   and does not feed them back into subsequent prompts or persist them in
 *   the assistant message.
 *
 * Turn format:
 * ```
 * <|turn>system
 * {system_message}<turn|>
 * <|turn>user
 * {user_message}<turn|>
 * <|turn>model
 * ```
 */
public class Gemma4ChatTemplate : ChatTemplate {

    private val json = Json { ignoreUnknownKeys = true }

    override fun apply(
        messages: List<ChatMessage>,
        tools: List<ToolDefinition>,
        addGenerationPrompt: Boolean
    ): String {
        val sb = StringBuilder()

        // Tool definitions as a system turn with <|tool>...<tool|> delimiters
        if (tools.isNotEmpty()) {
            sb.append("<|turn>system\n")
            sb.append("<|tool>\n")

            val toolsJson = buildJsonArray {
                for (tool in tools) {
                    add(buildJsonObject {
                        put("name", tool.name)
                        put("description", tool.description)
                        put("parameters", tool.parameters)
                    })
                }
            }
            sb.append(Json.encodeToString(toolsJson))
            sb.append("\n<tool|>")
            sb.append("<turn|>\n")
        }

        for (msg in messages) {
            when (msg.role) {
                ChatRole.SYSTEM -> {
                    sb.append("<|turn>system\n")
                    sb.append(msg.content)
                    sb.append("<turn|>\n")
                }
                ChatRole.USER -> {
                    sb.append("<|turn>user\n")
                    sb.append(msg.content)
                    sb.append("<turn|>\n")
                }
                ChatRole.ASSISTANT -> {
                    sb.append("<|turn>model\n")
                    sb.append(msg.content)
                    sb.append("<turn|>\n")
                }
                ChatRole.TOOL -> {
                    sb.append("<|turn>user\n")
                    sb.append("<|tool_response>\n")
                    val responseJson = buildJsonObject {
                        put("name", msg.toolCallId ?: "")
                        put("response", buildJsonObject {
                            put("result", msg.content)
                        })
                    }
                    sb.append(Json.encodeToString(JsonObject.serializer(), responseJson))
                    sb.append("\n<tool_response|>")
                    sb.append("<turn|>\n")
                }
            }
        }

        if (addGenerationPrompt) {
            sb.append("<|turn>model\n")
        }

        return sb.toString()
    }

    override fun parseToolCalls(text: String): List<ToolCall> {
        val calls = mutableListOf<ToolCall>()

        // Find tool calls within <|tool_call>...<tool_call|> delimiters
        var searchFrom = 0
        while (searchFrom < text.length) {
            val start = text.indexOf("<|tool_call>", searchFrom)
            if (start == -1) break
            val end = text.indexOf("<tool_call|>", start)
            if (end == -1) break

            val content = text.substring(start + "<|tool_call>".length, end).trim()
            val call = parseFunctionCall(content)
            if (call != null) calls.add(call)
            searchFrom = end + "<tool_call|>".length
        }

        // Fallback: also check for bare JSON with functionCall key (Gemma 2/3 style)
        if (calls.isEmpty()) {
            var i = 0
            while (i < text.length) {
                val jsonStart = text.indexOf('{', i)
                if (jsonStart == -1) break
                val jsonStr = extractJsonObject(text, jsonStart)
                if (jsonStr != null && jsonStr.contains("\"functionCall\"")) {
                    val call = parseLegacyFunctionCall(jsonStr)
                    if (call != null) calls.add(call)
                    i = jsonStart + jsonStr.length
                } else {
                    i = jsonStart + 1
                }
            }
        }

        return calls
    }

    override fun containsToolCall(text: String): Boolean {
        return text.contains("<|tool_call>") || text.contains("\"functionCall\"")
    }

    override fun parseThinkingBlocks(text: String): List<String> {
        val blocks = mutableListOf<String>()
        var searchFrom = 0
        while (searchFrom < text.length) {
            val start = text.indexOf(THINK_OPEN, searchFrom)
            if (start == -1) break
            val contentStart = start + THINK_OPEN.length
            val end = text.indexOf(THINK_CLOSE, contentStart)
            if (end == -1) break
            blocks += text.substring(contentStart, end)
            searchFrom = end + THINK_CLOSE.length
        }
        return blocks
    }

    override fun stripThinking(text: String): String {
        if (!text.contains(THINK_OPEN)) return text
        val sb = StringBuilder(text.length)
        var cursor = 0
        while (cursor < text.length) {
            val start = text.indexOf(THINK_OPEN, cursor)
            if (start == -1) {
                sb.append(text, cursor, text.length)
                break
            }
            sb.append(text, cursor, start)
            val end = text.indexOf(THINK_CLOSE, start + THINK_OPEN.length)
            if (end == -1) {
                // Unterminated block — drop everything from the opener on so the
                // thinking text doesn't leak into conversation history.
                break
            }
            cursor = end + THINK_CLOSE.length
        }
        // Collapse the whitespace we may have just created around the removed block.
        return sb.toString().replace(Regex("""[\t ]*\n[\t ]*\n[\t ]*\n+"""), "\n\n").trim('\n', ' ', '\t')
    }

    private fun parseFunctionCall(content: String): ToolCall? {
        return try {
            val obj = json.parseToJsonElement(content).jsonObject
            val name = obj["name"]?.jsonPrimitive?.content ?: return null
            val args = obj["args"]?.jsonObject ?: JsonObject(emptyMap())
            ToolCall(
                id = ToolCallParser.generateCallId(),
                name = name,
                arguments = args
            )
        } catch (_: Exception) {
            null
        }
    }

    private fun parseLegacyFunctionCall(jsonStr: String): ToolCall? {
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

    private companion object {
        const val THINK_OPEN = "<|think>"
        const val THINK_CLOSE = "<think|>"
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
}
