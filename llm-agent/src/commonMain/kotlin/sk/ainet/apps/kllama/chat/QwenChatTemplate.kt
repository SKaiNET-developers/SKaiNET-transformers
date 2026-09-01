package sk.ainet.apps.kllama.chat

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * Chat template for Qwen 3 / 3.5 family models with native tool calling, faithful to the
 * `chat_template` shipped in the official Qwen3 tokenizer configs (verified against
 * `Qwen/Qwen3-0.6B`). Qwen 2.5 instruct models use the same ChatML + Hermes-style contract.
 *
 * The five points where the official template is stricter than generic ChatML — each one
 * measured to matter, especially on the small (0.6B/1.7B) checkpoints:
 *
 * 1. **Tool results are `user` turns.** A `ChatRole.TOOL` message renders as
 *    `<|im_start|>user\n<tool_response>\n…\n</tool_response><|im_end|>`, and *consecutive*
 *    tool results merge into ONE user turn with stacked `<tool_response>` blocks. The
 *    literal `tool` role string never appears in Qwen's training data.
 * 2. **Tools are listed one JSON object per line** inside `<tools></tools>` — not as a
 *    single packed JSON array.
 * 3. **No injected persona.** The system block carries the caller's own system message (if
 *    any) followed by the `# Tools` section; the official template adds no
 *    "You are Qwen…" line.
 * 4. **Thinking mode.** Qwen3 emits `<think>…</think>` before its answer by default;
 *    [parseThinkingBlocks]/[stripThinking] handle it so reasoning is surfaced to listeners
 *    but never persisted into history or shown as the final answer. Construct with
 *    `enableThinking = false` to reproduce the official `enable_thinking=false` behaviour —
 *    the generation prompt is pre-filled with an empty `<think>` block, which is what stops
 *    the model from reasoning. Recommended for small checkpoints in tool-calling loops:
 *    a 0.6B model can spend hundreds of tokens thinking before it reaches the
 *    `<tool_call>`.
 * 5. **Assistant tool calls replay as `<tool_call>` blocks**, rebuilt from the structured
 *    [ChatMessage.toolCalls] — not from whatever raw text the model happened to emit.
 *
 * Tool call output format (model response):
 * ```
 * <tool_call>
 * {"name": "...", "arguments": {...}}
 * </tool_call>
 * ```
 */
public class QwenChatTemplate(
    private val enableThinking: Boolean = true,
) : ChatTemplate {

    override fun apply(
        messages: List<ChatMessage>,
        tools: List<ToolDefinition>,
        addGenerationPrompt: Boolean
    ): String {
        val sb = StringBuilder()

        val leadingSystem = messages.firstOrNull()?.takeIf { it.role == ChatRole.SYSTEM }

        if (tools.isNotEmpty()) {
            // System block: caller's system message (if any), then the # Tools section —
            // one tool JSON per line, exactly like the official template's
            // `{{ tool | tojson }}` loop.
            sb.append("<|im_start|>system\n")
            if (leadingSystem != null) {
                sb.append(leadingSystem.content).append("\n\n")
            }
            sb.append("# Tools\n\n")
            sb.append("You may call one or more functions to assist with the user query.\n\n")
            sb.append("You are provided with function signatures within <tools></tools> XML tags:\n")
            sb.append("<tools>")
            for (tool in tools) {
                sb.append("\n")
                sb.append(
                    Json.encodeToString(
                        buildJsonObject {
                            put("type", "function")
                            put("function", buildJsonObject {
                                put("name", tool.name)
                                put("description", tool.description)
                                put("parameters", tool.parameters)
                            })
                        }
                    )
                )
            }
            sb.append("\n</tools>\n\n")
            sb.append("For each function call, return a json object with function name and arguments ")
            sb.append("within <tool_call></tool_call> XML tags:\n")
            sb.append("<tool_call>\n")
            sb.append("{\"name\": <function-name>, \"arguments\": <args-json-object>}\n")
            sb.append("</tool_call><|im_end|>\n")
        } else if (leadingSystem != null) {
            sb.append("<|im_start|>system\n").append(leadingSystem.content).append("<|im_end|>\n")
        }

        val body = if (leadingSystem != null) messages.drop(1) else messages
        var i = 0
        while (i < body.size) {
            val msg = body[i]
            when (msg.role) {
                ChatRole.TOOL -> {
                    // Consecutive tool results merge into a single user turn of
                    // stacked <tool_response> blocks (official template behaviour).
                    sb.append("<|im_start|>user")
                    while (i < body.size && body[i].role == ChatRole.TOOL) {
                        sb.append("\n<tool_response>\n")
                        sb.append(body[i].content)
                        sb.append("\n</tool_response>")
                        i++
                    }
                    sb.append("<|im_end|>\n")
                    continue
                }
                ChatRole.ASSISTANT -> {
                    // Structured toolCalls are the source of truth for replay; any raw
                    // <tool_call> XML the model emitted (which AgentLoop keeps in content)
                    // is dropped so the call isn't rendered twice.
                    var content = stripThinking(msg.content).trim('\n')
                    if (msg.toolCalls != null) {
                        content = TOOL_CALL_BLOCK.replace(content, "").trim('\n')
                    }
                    sb.append("<|im_start|>assistant\n")
                    sb.append(content)
                    msg.toolCalls?.forEachIndexed { idx, call ->
                        if (idx > 0 || content.isNotBlank()) sb.append("\n")
                        sb.append("<tool_call>\n{\"name\": \"")
                        sb.append(call.name)
                        sb.append("\", \"arguments\": ")
                        sb.append(Json.encodeToString(call.arguments))
                        sb.append("}\n</tool_call>")
                    }
                    sb.append("<|im_end|>\n")
                }
                else -> {
                    sb.append("<|im_start|>").append(msg.role.roleName).append("\n")
                    sb.append(msg.content)
                    sb.append("<|im_end|>\n")
                }
            }
            i++
        }

        if (addGenerationPrompt) {
            sb.append("<|im_start|>assistant\n")
            if (!enableThinking) {
                // The official enable_thinking=false: pre-fill an empty think block so
                // the model answers directly instead of reasoning first.
                sb.append("<think>\n\n</think>\n\n")
            }
        }

        return sb.toString()
    }

    override fun parseThinkingBlocks(text: String): List<String> =
        THINK_BLOCK.findAll(text).map { it.groupValues[1].trim('\n') }.filter { it.isNotBlank() }.toList()

    override fun stripThinking(text: String): String {
        var out = THINK_BLOCK.replace(text, "")
        // An unterminated block (generation budget ran out mid-thought) is still thinking —
        // don't leak it as the visible answer.
        val open = out.indexOf("<think>")
        if (open >= 0) out = out.substring(0, open)
        return out.trimStart('\n')
    }

    private companion object {
        val THINK_BLOCK = Regex("""<think>([\s\S]*?)</think>\n?""")
        val TOOL_CALL_BLOCK = Regex("""<tool_call>[\s\S]*?</tool_call>\n?""")
    }
}
