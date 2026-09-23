package sk.ainet.apps.kllama.chat

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * Chat template for Qwen 2.5 instruct models with tool calling, faithful to the official
 * `chat_template` shipped in `Qwen/Qwen2.5-0.5B-Instruct`'s tokenizer config (verified 2026-09-02;
 * closes SKaiNET-transformers#409). Structurally the same ChatML + Hermes-style contract
 * [QwenChatTemplate] implements for Qwen3 — ONE query template family covers both — with the two
 * real differences between the two models' official templates:
 *
 * 1. **A default persona.** When the caller supplies no system message, Qwen2.5's official
 *    template still opens the system turn with `"You are Qwen, created by Alibaba Cloud. You are
 *    a helpful assistant."` — Qwen3's official template dropped this (see [QwenChatTemplate]'s
 *    "No injected persona" point). Qwen2.5-0.5B was measured 0/8 and 1/8 golden-8 (host,
 *    zero-shot) under generic ChatML / [QwenChatTemplate], vs 4-5/8 with this exact template
 *    (`tvv/nlu-llm-harness`, P0.3/Q0.2) — the persona line and the exact section wording both
 *    matter on a checkpoint this small.
 * 2. **No thinking mode at all.** Qwen2.5 predates the `<think>…</think>` convention entirely —
 *    it never saw a `<think>` token in training, not even an empty one. [QwenChatTemplate]'s
 *    `enableThinking = false` mode still pre-fills an empty `<think>\n\n</think>\n\n` block to
 *    *suppress* Qwen3's native thinking, which is exactly the wrong move here: that block is
 *    itself out-of-distribution for Qwen2.5. This template never emits, strips or parses
 *    `<think>` anywhere; [parseThinkingBlocks] always returns an empty list and [stripThinking]
 *    is a no-op.
 *
 * Everything else — tool listing (one JSON object per line inside `<tools></tools>`), tool
 * results as merged `user` turns of stacked `<tool_response>` blocks, and assistant tool-call
 * replay from the structured [ChatMessage.toolCalls] rather than raw text — is the same Hermes
 * convention [QwenChatTemplate] already implements correctly; this class reuses that logic
 * unchanged rather than the narrower, single-turn version `tvv/nlu-llm-harness`'s
 * `Qwen25ToolTemplate.kt` got away with (that harness only ever scores one utterance -> one call,
 * so it never needed multi-turn tool-response merging or assistant-side replay to be correct).
 *
 * Tool call output format (model response) is identical to [QwenChatTemplate]'s:
 * ```
 * <tool_call>
 * {"name": "...", "arguments": {...}}
 * </tool_call>
 * ```
 */
public class Qwen25ChatTemplate : ChatTemplate {

    override fun apply(
        messages: List<ChatMessage>,
        tools: List<ToolDefinition>,
        addGenerationPrompt: Boolean
    ): String {
        val sb = StringBuilder()

        val leadingSystem = messages.firstOrNull()?.takeIf { it.role == ChatRole.SYSTEM }
        val systemContent = leadingSystem?.content ?: DEFAULT_PERSONA

        // System block: caller's system message or the default persona, then the # Tools
        // section when tools are offered — one tool JSON per line, exactly like the official
        // template's `{{ tool | tojson }}` loop. Unlike QwenChatTemplate, this system turn is
        // ALWAYS emitted (persona or not), matching Qwen2.5's official template.
        sb.append("<|im_start|>system\n").append(systemContent)
        if (tools.isNotEmpty()) {
            sb.append("\n\n# Tools\n\n")
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
            sb.append("</tool_call>")
        }
        sb.append("<|im_end|>\n")

        val body = if (leadingSystem != null) messages.drop(1) else messages
        var i = 0
        while (i < body.size) {
            val msg = body[i]
            when (msg.role) {
                ChatRole.TOOL -> {
                    // Consecutive tool results merge into a single user turn of stacked
                    // <tool_response> blocks (same Hermes convention QwenChatTemplate uses).
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
                    // is dropped so the call isn't rendered twice. No thinking to strip.
                    var content = msg.content.trim('\n')
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
        }

        return sb.toString()
    }

    /** Qwen2.5 has no thinking mode; nothing to find. */
    override fun parseThinkingBlocks(text: String): List<String> = emptyList()

    /** Qwen2.5 has no thinking mode; the text is already the whole answer. */
    override fun stripThinking(text: String): String = text

    private companion object {
        const val DEFAULT_PERSONA = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        val TOOL_CALL_BLOCK = Regex("""<tool_call>[\s\S]*?</tool_call>\n?""")
    }
}
