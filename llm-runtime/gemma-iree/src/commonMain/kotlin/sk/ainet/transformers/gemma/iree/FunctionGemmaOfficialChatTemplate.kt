package sk.ainet.transformers.gemma.iree

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import sk.ainet.apps.kllama.chat.ChatMessage
import sk.ainet.apps.kllama.chat.ChatRole
import sk.ainet.apps.kllama.chat.ChatTemplate
import sk.ainet.apps.kllama.chat.ToolDefinition
import sk.ainet.apps.kllama.chat.ToolCall as AgentToolCall

/**
 * [ChatTemplate] for the OFFICIAL `google/functiongemma-270m-it` release — the
 * `<start_function_declaration>` token family — as opposed to
 * [FunctionGemmaChatTemplate], which serves the Octopus-v2 `<tool_N>` fine-tune
 * and stays untouched.
 *
 * The rendering is a faithful Kotlin port of the Jinja `tokenizer.chat_template`
 * embedded in the released GGUF (extracted verbatim and pinned by
 * `FunctionGemmaOfficialGgufTest`). The non-obvious release conventions, all
 * confirmed from that template:
 *
 *  - String values are wrapped in the **`<escape>` special token**, NOT quotes:
 *    `description:<escape>City name<escape>`. Rendering quotes instead makes the
 *    model decline to call tools (observed empirically).
 *  - Inside a property object, `description` renders first and `type` renders
 *    LAST, `<escape>`-wrapped and uppercased: `{description:<escape>…<escape>,type:<escape>STRING<escape>}`.
 *  - Declaration parameter order is `properties`, `required`, `type`; property
 *    maps iterate in sorted-key order (Jinja `dictsort`).
 *  - The developer preamble and each `<start_function_declaration>…` block are
 *    concatenated WITHOUT newlines; the developer turn closes with `<end_of_turn>\n`.
 *  - `<bos>` is NOT emitted here — the runtime prepends the BOS id.
 *
 * Rendered shape:
 *
 * ```
 * <start_of_turn>developer
 * You are a model that can do function calling with the following functions<start_function_declaration>declaration:get_weather{description:<escape>…<escape>,parameters:{properties:{location:{description:<escape>City name<escape>,type:<escape>STRING<escape>}},required:[<escape>location<escape>],type:<escape>OBJECT<escape>}}<end_function_declaration><end_of_turn>
 * <start_of_turn>user
 * What's the weather in Paris?<end_of_turn>
 * <start_of_turn>model
 * ```
 *
 * The model answers with an optional `<think>…</think>` span followed by
 * `<start_function_call>call:name{location:<escape>Paris<escape>}<end_function_call>`,
 * parsed by [FunctionGemmaOfficialToolCallParserStrategy].
 */
public class FunctionGemmaOfficialChatTemplate(
    private val parser: FunctionGemmaOfficialToolCallParserStrategy =
        FunctionGemmaOfficialToolCallParserStrategy(),
) : ChatTemplate {

    override fun apply(
        messages: List<ChatMessage>,
        tools: List<ToolDefinition>,
        addGenerationPrompt: Boolean,
    ): String {
        val sb = StringBuilder()
        var loopMessages = messages
        val leadingSystem = messages.firstOrNull()?.role == ChatRole.SYSTEM
        if (tools.isNotEmpty() || leadingSystem) {
            sb.append("<start_of_turn>developer\n")
            if (leadingSystem) {
                sb.append(messages.first().content.trim())
                loopMessages = messages.drop(1)
            } else {
                sb.append(DEVELOPER_PREAMBLE)
            }
            for (tool in tools) {
                sb.append("<start_function_declaration>")
                sb.append(renderDeclaration(tool).trim())
                sb.append("<end_function_declaration>")
            }
            sb.append("<end_of_turn>\n")
        }
        for (msg in loopMessages) {
            when (msg.role) {
                ChatRole.TOOL -> {
                    // Tool responses render as response blocks without their own
                    // <start_of_turn> framing (they continue the pre-opened
                    // function-response context per the release template).
                    sb.append("<start_function_response>response:")
                    sb.append(msg.content)
                    sb.append("<end_function_response>")
                }
                else -> {
                    val role = if (msg.role == ChatRole.ASSISTANT) "model" else "user"
                    sb.append("<start_of_turn>").append(role).append('\n')
                    sb.append(msg.content.trim())
                    sb.append("<end_of_turn>\n")
                }
            }
        }
        if (addGenerationPrompt) {
            sb.append("<start_of_turn>model\n")
        }
        return sb.toString()
    }

    override fun parseToolCalls(text: String): List<AgentToolCall> = parser.parse(text)

    override fun containsToolCall(text: String): Boolean = parser.containsToolCall(text)

    override fun parseThinkingBlocks(text: String): List<String> =
        THINK_RE.findAll(text).map { it.groupValues[1] }.toList()

    override fun stripThinking(text: String): String = THINK_RE.replace(text, "")

    public companion object {
        public const val DEVELOPER_PREAMBLE: String =
            "You are a model that can do function calling with the following functions"

        private val THINK_RE = Regex("""<think>(.*?)</think>""", RegexOption.DOT_MATCHES_ALL)

        private const val ESC = "<escape>"
        private val STANDARD_KEYS = setOf("description", "type", "properties", "required", "nullable")

        private fun esc(s: String): String = ESC + s + ESC

        /** Port of the template's `format_function_declaration` macro. */
        public fun renderDeclaration(tool: ToolDefinition): String = buildString {
            append("declaration:").append(tool.name)
            append("{description:").append(esc(tool.description))
            val params = tool.parameters
            if (params.isNotEmpty()) {
                append(",parameters:{")
                val props = params["properties"] as? JsonObject
                if (!props.isNullOrEmpty()) {
                    append("properties:{").append(renderProperties(props)).append("},")
                }
                val required = params["required"] as? JsonArray
                if (!required.isNullOrEmpty()) {
                    append("required:[")
                    append(required.joinToString(",") { esc(it.stringContent()) })
                    append("],")
                }
                (params["type"] as? JsonPrimitive)?.let {
                    append("type:").append(esc(it.content.uppercase())).append("}")
                }
            }
            append("}")
        }

        /** Port of the template's `format_parameters` macro (sorted-key iteration). */
        private fun renderProperties(properties: JsonObject): String =
            properties.entries
                .filter { it.key !in STANDARD_KEYS }
                .sortedBy { it.key }
                .joinToString(",") { (key, value) ->
                    renderProperty(key, value as? JsonObject ?: JsonObject(emptyMap()))
                }

        private fun renderProperty(key: String, prop: JsonObject): String = buildString {
            val type = (prop["type"] as? JsonPrimitive)?.content?.uppercase() ?: ""
            append(key).append(":{description:")
            append(esc((prop["description"] as? JsonPrimitive)?.content ?: ""))
            when (type) {
                "STRING" -> (prop["enum"] as? JsonArray)?.let { enum ->
                    append(",enum:").append(formatArgument(enum))
                }
                "OBJECT" -> {
                    append(",properties:{")
                    (prop["properties"] as? JsonObject)?.let { append(renderProperties(it)) }
                    append("}")
                    (prop["required"] as? JsonArray)?.takeIf { it.isNotEmpty() }?.let { req ->
                        append(",required:[")
                        append(req.joinToString(",") { esc(it.stringContent()) })
                        append("]")
                    }
                }
                "ARRAY" -> (prop["items"] as? JsonObject)?.takeIf { it.isNotEmpty() }?.let { items ->
                    append(",items:{")
                    append(
                        items.entries.sortedBy { it.key }.joinToString(",") { (ik, iv) ->
                            when (ik) {
                                "properties" -> "properties:{" +
                                    ((iv as? JsonObject)?.let { renderProperties(it) } ?: "") + "}"
                                "required" -> "required:[" +
                                    ((iv as? JsonArray)?.joinToString(",") { esc(it.stringContent()) } ?: "") + "]"
                                "type" -> "type:" + formatArgument(
                                    JsonPrimitive((iv as? JsonPrimitive)?.content?.uppercase() ?: "")
                                )
                                else -> "$ik:" + formatArgument(iv)
                            }
                        }
                    )
                    append("}")
                }
            }
            append(",type:").append(esc(type)).append("}")
        }

        /** Port of the template's `format_argument` macro (escape_keys=true form). */
        private fun formatArgument(argument: JsonElement): String = when (argument) {
            is JsonObject -> argument.entries.sortedBy { it.key }.joinToString(
                prefix = "{", postfix = "}", separator = ","
            ) { (k, v) -> esc(k) + ":" + formatArgument(v) }
            is JsonArray -> argument.joinToString(prefix = "[", postfix = "]", separator = ",") {
                formatArgument(it)
            }
            is JsonPrimitive -> if (argument.isString) esc(argument.content) else argument.content
        }

        private fun JsonElement.stringContent(): String = (this as? JsonPrimitive)?.content ?: toString()
    }
}
