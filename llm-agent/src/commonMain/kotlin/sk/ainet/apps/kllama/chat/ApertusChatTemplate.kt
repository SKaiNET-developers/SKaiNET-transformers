package sk.ainet.apps.kllama.chat

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/**
 * Chat template for Apertus models (Swiss AI / EPFL).
 *
 * Implements the prompt format documented in
 * `docs/specs/apertus-chat-template.md`, sourced from
 * `swiss-ai/Apertus-8B-Instruct-2509`'s `chat_template.jinja`. Apertus
 * has its own role tokens — `<|system_start|>`, `<|developer_start|>`,
 * `<|user_start|>`, `<|assistant_start|>`, `<|inner_prefix|>`,
 * `<|tools_prefix|>` — and renders tool definitions as
 * **TypeScript-style type declarations**, NOT JSON Schema.
 *
 * Construction params let callers override pieces that the Jinja
 * template injects dynamically:
 *
 * @param defaultSystemPrompt The text used when the caller doesn't
 *   supply a system message. Apertus is trained on the literal default
 *   string `"You are Apertus, a helpful assistant created by the
 *   SwissAI initiative.\nKnowledge cutoff: 2024-04\nCurrent date: ..."`.
 *   Override only if the caller has a verified alternative; the model
 *   may behave worse with a different system prompt.
 * @param currentDate Date string substituted into the default system
 *   prompt at the `Current date:` line. Defaults to a fixed value so
 *   tests are deterministic; production callers can pass today's date
 *   via the auxiliary [withCurrentDate] helper if they want.
 * @param enableThinking Whether to emit `Deliberation: enabled` (vs
 *   `disabled`) in the developer block. Defaults to `false`. Apertus's
 *   thinking mode is documented but not yet exercised by SKaiNET's
 *   agent loop, so the safe default is `disabled`.
 *
 * Limitations vs the upstream Jinja template (carried as TODOs for
 * future hardening):
 * - The TypeScript type renderer covers `string`, `number`, `integer`,
 *   `boolean`, primitive `array`, and shallow `object` (object with
 *   primitive properties). `oneOf`, `nullable`, deep-nested objects,
 *   `enum`, and union arrays fall back to `any`.
 * - Multi-block assistant content (`thoughts` / `tool_calls` /
 *   `tool_outputs` / `response` blocks) isn't expressible in our
 *   [ChatMessage] model, which carries `content: String` plus optional
 *   `toolCalls: List<ToolCall>?`. The renderer maps the latter to the
 *   `<|tools_prefix|>...<|tools_suffix|>` form and treats `TOOL` role
 *   messages as `[output]` continuations of the current assistant
 *   turn — equivalent to Apertus's "Shape 1 string + tool_calls"
 *   handling for our current callers.
 */
public class ApertusChatTemplate(
    private val defaultSystemPrompt: String? = null,
    private val currentDate: String = "2026-05-01",
    private val enableThinking: Boolean = false,
) : ChatTemplate {

    override fun apply(
        messages: List<ChatMessage>,
        tools: List<ToolDefinition>,
        addGenerationPrompt: Boolean,
    ): String = buildString {
        // BOS at the very start; Apertus's tokenizer_config has add_bos_token=true
        // and the Jinja emits {{ bos_token }} as the first token.
        append("<s>")

        // System block — caller-supplied or default
        val (system, rest) = if (messages.isNotEmpty() && messages[0].role == ChatRole.SYSTEM) {
            messages[0].content to messages.drop(1)
        } else {
            renderDefaultSystemPrompt() to messages
        }
        append(SYSTEM_START).append(system).append(SYSTEM_END)

        // Developer block — always emitted, even with no tools / disabled thinking
        append(DEVELOPER_START)
        append("Deliberation: ").append(if (enableThinking) "enabled\n" else "disabled\n")
        if (tools.isNotEmpty()) {
            append("Tool Capabilities:\n")
            append(renderTools(tools))
        } else {
            append("Tool Capabilities: disabled")
        }
        append(DEVELOPER_END)

        // Per-message turns
        var inAssistant = false
        var inToolOutputs = false
        for (message in rest) {
            when (message.role) {
                ChatRole.SYSTEM -> {
                    // Multiple system messages aren't part of the spec; skip silently.
                    // The Jinja raises in that case; we degrade to a no-op for forward
                    // compatibility with callers that pass duplicate system prompts.
                }
                ChatRole.USER -> {
                    if (inToolOutputs) {
                        append(']')
                        inToolOutputs = false
                    }
                    if (inAssistant) {
                        append(ASSISTANT_END)
                        inAssistant = false
                    }
                    append(USER_START).append(message.content).append(USER_END)
                }
                ChatRole.ASSISTANT -> {
                    if (!inAssistant) {
                        append(ASSISTANT_START)
                        inAssistant = true
                    }
                    if (inToolOutputs) {
                        append(']')
                        inToolOutputs = false
                    }
                    append(message.content)
                    val tcs = message.toolCalls
                    if (!tcs.isNullOrEmpty()) {
                        append(TOOLS_PREFIX).append('[')
                        for ((index, call) in tcs.withIndex()) {
                            if (index > 0) append(", ")
                            append('{').append('"').append(call.name).append("\": ")
                            append(call.arguments.toString())
                            append('}')
                        }
                        append(']').append(TOOLS_SUFFIX)
                    }
                }
                ChatRole.TOOL -> {
                    // Tool outputs render as [output1, output2, ...] inside the
                    // current assistant turn — matches the Jinja's `role == 'tool'`
                    // branch which keeps `ns.in_tool` open across consecutive tool
                    // messages.
                    if (!inAssistant) {
                        // Defensive: open an assistant turn if the caller forgot to.
                        append(ASSISTANT_START)
                        inAssistant = true
                    }
                    if (!inToolOutputs) {
                        append('[')
                        inToolOutputs = true
                    } else {
                        append(", ")
                    }
                    append(message.content)
                }
            }
        }
        if (inToolOutputs) {
            append(']')
        }
        if (inAssistant) {
            append(ASSISTANT_END)
        }
        if (addGenerationPrompt) {
            append(ASSISTANT_START)
        }
    }

    override fun parseToolCalls(text: String): List<ToolCall> =
        ApertusToolCallParserStrategy.parse(text)

    override fun containsToolCall(text: String): Boolean =
        ApertusToolCallParserStrategy.containsToolCall(text)

    /**
     * Apertus uses `<|inner_prefix|>...<|inner_suffix|>` to wrap
     * deliberation / chain-of-thought. Strip when feeding back into a
     * subsequent prompt or surfacing to a non-debugging UI.
     */
    override fun parseThinkingBlocks(text: String): List<String> {
        val results = mutableListOf<String>()
        var idx = 0
        while (true) {
            val start = text.indexOf(INNER_PREFIX, idx)
            if (start < 0) break
            val end = text.indexOf(INNER_SUFFIX, start + INNER_PREFIX.length)
            if (end < 0) break
            results.add(text.substring(start + INNER_PREFIX.length, end))
            idx = end + INNER_SUFFIX.length
        }
        return results
    }

    override fun stripThinking(text: String): String {
        val sb = StringBuilder(text.length)
        var idx = 0
        while (true) {
            val start = text.indexOf(INNER_PREFIX, idx)
            if (start < 0) {
                sb.append(text, idx, text.length)
                break
            }
            sb.append(text, idx, start)
            val end = text.indexOf(INNER_SUFFIX, start + INNER_PREFIX.length)
            if (end < 0) {
                // Unterminated thinking block — keep the original content.
                sb.append(text, start, text.length)
                break
            }
            idx = end + INNER_SUFFIX.length
        }
        return sb.toString()
    }

    /** Build a copy with [date] substituted into the default system prompt. */
    public fun withCurrentDate(date: String): ApertusChatTemplate =
        ApertusChatTemplate(defaultSystemPrompt, date, enableThinking)

    private fun renderDefaultSystemPrompt(): String =
        defaultSystemPrompt
            ?: "You are Apertus, a helpful assistant created by the SwissAI initiative.\n" +
                "Knowledge cutoff: 2024-04\n" +
                "Current date: $currentDate"

    /**
     * Render a list of [ToolDefinition] as TypeScript-style type
     * declarations. Mirrors the Jinja `render_tools` macro for the
     * cases we currently need; complex schemas (oneOf, nullable, enum,
     * union arrays, deep nesting) collapse to `any`.
     */
    private fun renderTools(tools: List<ToolDefinition>): String = buildString {
        for ((toolIndex, tool) in tools.withIndex()) {
            append("// ").append(tool.description).append('\n')
            append("type ").append(tool.name).append(" = ")
            val properties = (tool.parameters["properties"] as? JsonObject)
            val required = (tool.parameters["required"] as? JsonArray)
                ?.mapNotNull { it.jsonPrimitive.contentOrNull }
                ?.toSet()
                ?: emptySet()
            if (properties.isNullOrEmpty()) {
                append("() => any;")
            } else {
                append("(_: {\n")
                val entries = properties.entries.toList()
                for ((idx, entry) in entries.withIndex()) {
                    val (paramName, paramSpec) = entry
                    val paramObj = paramSpec as? JsonObject
                    val description = paramObj?.get("description")?.jsonPrimitive?.contentOrNull
                    if (description != null) {
                        append("// ").append(description).append('\n')
                    }
                    append(paramName)
                    if (paramName !in required) append('?')
                    append(": ").append(renderTypescriptType(paramObj))
                    if (idx < entries.size - 1) append(",\n") else append('\n')
                }
                append("}) => any;")
            }
            if (toolIndex < tools.size - 1) append('\n')
        }
    }

    private fun renderTypescriptType(spec: JsonObject?): String {
        if (spec == null) return "any"
        val type = spec["type"]?.jsonPrimitive?.contentOrNull
        val nullable = spec["nullable"]?.let { runCatching { (it as? JsonPrimitive)?.boolean }.getOrNull() } == true
        val base = when (type) {
            "string" -> {
                val enum = spec["enum"] as? JsonArray
                if (enum != null && enum.isNotEmpty()) {
                    enum.joinToString(prefix = "\"", postfix = "\"", separator = "\" | \"") {
                        it.jsonPrimitive.contentOrNull.orEmpty()
                    }
                } else {
                    "string"
                }
            }
            "number", "integer" -> "number"
            "boolean" -> "boolean"
            "array" -> renderArrayType(spec)
            "object" -> renderObjectType(spec)
            else -> "any"
        }
        return if (nullable) "$base | null" else base
    }

    private fun renderArrayType(spec: JsonObject): String {
        val items = spec["items"] as? JsonObject ?: return "any[]"
        return when (items["type"]?.jsonPrimitive?.contentOrNull) {
            "string" -> "string[]"
            "number", "integer" -> "number[]"
            "boolean" -> "boolean[]"
            else -> "any[]"
        }
    }

    private fun renderObjectType(spec: JsonObject): String {
        val properties = spec["properties"] as? JsonObject ?: return "object"
        if (properties.isEmpty()) return "object"
        val required = (spec["required"] as? JsonArray)
            ?.mapNotNull { it.jsonPrimitive.contentOrNull }
            ?.toSet()
            ?: emptySet()
        return buildString {
            append("{\n")
            val entries = properties.entries.toList()
            for ((idx, entry) in entries.withIndex()) {
                val (name, valueSpec) = entry
                append(name)
                if (name !in required) append('?')
                append(": ")
                append(renderTypescriptType(valueSpec as? JsonObject))
                if (idx < entries.size - 1) append(", ") else append('\n')
            }
            append('}')
        }
    }

    public companion object {
        public const val SYSTEM_START: String = "<|system_start|>"
        public const val SYSTEM_END: String = "<|system_end|>"
        public const val DEVELOPER_START: String = "<|developer_start|>"
        public const val DEVELOPER_END: String = "<|developer_end|>"
        public const val USER_START: String = "<|user_start|>"
        public const val USER_END: String = "<|user_end|>"
        public const val ASSISTANT_START: String = "<|assistant_start|>"
        public const val ASSISTANT_END: String = "<|assistant_end|>"
        public const val INNER_PREFIX: String = "<|inner_prefix|>"
        public const val INNER_SUFFIX: String = "<|inner_suffix|>"
        public const val TOOLS_PREFIX: String = "<|tools_prefix|>"
        public const val TOOLS_SUFFIX: String = "<|tools_suffix|>"
    }
}
