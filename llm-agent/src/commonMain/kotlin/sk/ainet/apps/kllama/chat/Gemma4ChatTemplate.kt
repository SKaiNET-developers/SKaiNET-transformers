package sk.ainet.apps.kllama.chat

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put

/**
 * Chat template for the Gemma 4 family with native tool calling.
 *
 * Faithful Kotlin port of the official HuggingFace chat template shipped
 * with `google/gemma-4-e2b-it` (`chat_template.jinja`). Empirical
 * observation against the real E2B checkpoint showed that the prior
 * JSON-flavored template was not the format the model was trained on —
 * the model produced prose instead of tool calls. This implementation
 * matches the actual training format:
 *
 * - **`<bos>` is emitted by the template.** GGUF metadata sets
 *   `tokenizer.ggml.add_bos_token=false` precisely because the chat
 *   template prepends BOS itself.
 * - **One `<|tool>...<tool|>` block per tool**, body in
 *   `declaration:NAME{description:<|"|>...<|"|>,parameters:{...}}` form.
 *   NOT a JSON array.
 * - **`<|"|>` (id 168 in the GGUF vocab, USER_DEFINED) is the quote
 *   token for string literals**, not the regular `"`. The tokenizer
 *   encodes it atomically (see `GGUFTokenizer.specialTokensByLength`).
 * - **JSON-schema type names are uppercased** (`STRING`, `OBJECT`,
 *   `ARRAY`, `NUMBER`, `BOOLEAN`).
 * - **Tool calls use `<|tool_call>call:NAME{key:value,...}<tool_call|>`**,
 *   with bare keys and `<|"|>...<|"|>` string values. Numbers and
 *   booleans serialize bare (`42`, `true`).
 * - **Tool responses use `<|tool_response>response:NAME{...}<tool_response|>`**
 *   on assistant continuation, NOT wrapped in a separate user turn.
 * - **Thinking mode** has two pieces:
 *     1. A bare `<|think|>` token (single, unpaired) at the top of the
 *        first system turn signals "reasoning is enabled". Controlled by
 *        the `enableThinking` constructor flag.
 *     2. The model emits actual reasoning blocks bracketed by
 *        `<|channel>thought\n…<channel|>`. `parseThinkingBlocks` /
 *        `stripThinking` operate on those.
 *
 * Reference: the `chat_template.jinja` shipped in the
 * `google/gemma-4-e2b-it` snapshot under
 * `~/.cache/huggingface/hub/models--google--gemma-4-e2b-it/snapshots/`.
 *
 * @param enableThinking If true, emit `<|think|>` at the top of the
 *   first system turn so the model is primed to use the reasoning
 *   channel. Off by default — opt-in only.
 */
public class Gemma4ChatTemplate(
    private val enableThinking: Boolean = false
) : ChatTemplate {

    override fun apply(
        messages: List<ChatMessage>,
        tools: List<ToolDefinition>,
        addGenerationPrompt: Boolean
    ): String {
        val sb = StringBuilder()
        sb.append(BOS)

        // Find any leading system message in the input. HF's template
        // folds it into the same system turn that holds the tool block.
        val firstIsSystem = messages.firstOrNull()?.role == ChatRole.SYSTEM
        val systemHead = if (firstIsSystem) messages.first().content.trim() else ""
        val rest = if (firstIsSystem) messages.drop(1) else messages

        val needSystemTurn = enableThinking || tools.isNotEmpty() || systemHead.isNotEmpty()
        if (needSystemTurn) {
            sb.append("<|turn>system\n")
            if (enableThinking) sb.append("<|think|>\n")
            if (systemHead.isNotEmpty()) sb.append(systemHead)
            for (tool in tools) {
                sb.append(TOOL_OPEN)
                sb.append(formatFunctionDeclaration(tool))
                sb.append(TOOL_CLOSE)
            }
            sb.append(TURN_CLOSE).append('\n')
        }

        // Walk remaining messages. Tool messages are NOT visited in the
        // outer loop — HF emits tool responses inline INSIDE the
        // assistant message turn that triggered them via forward-scan.
        // Assistant continuations (same role twice in a row, no tool in
        // between) suppress the duplicate `<|turn>model` opener.
        // `lastEmittedTurnState` mirrors HF's `prev_message_type` logic:
        // determines whether the trailing generation prompt fires.
        var lastEmittedTurnState = TurnState.NONE
        for ((i, msg) in rest.withIndex()) {
            if (msg.role == ChatRole.TOOL) continue

            val role = if (msg.role == ChatRole.ASSISTANT) "model" else msg.role.roleName
            val prevNonTool = previousNonToolRole(rest, i)
            val continueSameModelTurn =
                role == "model" && prevNonTool == ChatRole.ASSISTANT
            if (!continueSameModelTurn) {
                sb.append("<|turn>").append(role).append('\n')
            }

            // Render tool calls embedded on the assistant message.
            val toolCalls = msg.toolCalls
            if (toolCalls != null) {
                for (tc in toolCalls) {
                    sb.append(TOOL_CALL_OPEN)
                    sb.append("call:").append(tc.name).append('{')
                    formatArgumentsBareKeys(sb, tc.arguments)
                    sb.append('}')
                    sb.append(TOOL_CALL_CLOSE)
                }
            }

            // Forward-scan for consecutive TOOL messages immediately
            // following this assistant turn, and emit their response
            // blocks inline (still inside the same model turn).
            var emittedToolResponse = false
            if (toolCalls != null) {
                var j = i + 1
                while (j < rest.size && rest[j].role == ChatRole.TOOL) {
                    val tool = rest[j]
                    val toolName = toolCalls.firstOrNull { it.id == tool.toolCallId }?.name ?: "unknown"
                    sb.append(formatToolResponseBlock(toolName, tool.content))
                    emittedToolResponse = true
                    j++
                }
            }

            // Body text. For assistant messages we strip prior thinking
            // channels so they don't leak back into the prompt.
            val content = if (role == "model") stripThinking(msg.content) else msg.content.trim()
            if (content.isNotEmpty()) sb.append(content)

            // Turn close. Per HF: when an assistant message has a
            // tool_call but no follow-up tool_response yet (waiting on
            // the tool), emit a bare `<|tool_response>` opener instead
            // of `<turn|>` to prime the model. Otherwise close normally
            // unless we just emitted a tool response with no content
            // (HF skips the close in that case to keep the model in
            // continuation mode).
            val isPendingToolCall = toolCalls != null && !emittedToolResponse && content.isEmpty()
            val skipClose = emittedToolResponse && content.isEmpty()
            lastEmittedTurnState = when {
                isPendingToolCall -> {
                    sb.append(TOOL_RESPONSE_OPEN)
                    TurnState.TOOL_CALL_PENDING
                }
                skipClose -> TurnState.TOOL_RESPONSE_OPEN
                else -> {
                    sb.append(TURN_CLOSE).append('\n')
                    TurnState.CLOSED
                }
            }
        }

        // Per HF: skip the generation prompt when prev message left the
        // turn dangling on a tool_call or tool_response — the model just
        // continues the open turn from where the prompt cut off.
        val turnStillOpen = lastEmittedTurnState == TurnState.TOOL_CALL_PENDING ||
            lastEmittedTurnState == TurnState.TOOL_RESPONSE_OPEN
        if (addGenerationPrompt && !turnStillOpen) {
            sb.append("<|turn>model\n")
        }
        return sb.toString()
    }

    private enum class TurnState { NONE, CLOSED, TOOL_CALL_PENDING, TOOL_RESPONSE_OPEN }

    override fun parseToolCalls(text: String): List<ToolCall> {
        val out = mutableListOf<ToolCall>()
        var cursor = 0
        while (cursor < text.length) {
            val open = text.indexOf(TOOL_CALL_OPEN, cursor)
            if (open == -1) break
            val close = text.indexOf(TOOL_CALL_CLOSE, open + TOOL_CALL_OPEN.length)
            if (close == -1) break
            val body = text.substring(open + TOOL_CALL_OPEN.length, close)
            val parsed = parseCallBody(body)
            if (parsed != null) out += parsed
            cursor = close + TOOL_CALL_CLOSE.length
        }
        return out
    }

    override fun containsToolCall(text: String): Boolean = text.contains(TOOL_CALL_OPEN)

    override fun parseThinkingBlocks(text: String): List<String> {
        val blocks = mutableListOf<String>()
        var cursor = 0
        while (cursor < text.length) {
            val open = text.indexOf(CHANNEL_OPEN, cursor)
            if (open == -1) break
            val contentStart = open + CHANNEL_OPEN.length
            // HF channels are typed: `<|channel>thought\n...`. We only
            // surface the `thought` channel as thinking; other channels
            // (if any future Gemma adds them) flow through unchanged.
            val newline = text.indexOf('\n', contentStart)
            val close = text.indexOf(CHANNEL_CLOSE, contentStart)
            if (close == -1) break
            if (newline in (contentStart + 1)..close) {
                val channelType = text.substring(contentStart, newline).trim()
                if (channelType == "thought") {
                    blocks += text.substring(newline + 1, close)
                }
            }
            cursor = close + CHANNEL_CLOSE.length
        }
        return blocks
    }

    override fun stripThinking(text: String): String {
        if (!text.contains(CHANNEL_OPEN)) return text
        val sb = StringBuilder(text.length)
        var cursor = 0
        while (cursor < text.length) {
            val open = text.indexOf(CHANNEL_OPEN, cursor)
            if (open == -1) {
                sb.append(text, cursor, text.length)
                break
            }
            sb.append(text, cursor, open)
            val close = text.indexOf(CHANNEL_CLOSE, open + CHANNEL_OPEN.length)
            if (close == -1) break  // unterminated → drop tail
            cursor = close + CHANNEL_CLOSE.length
        }
        return sb.toString().trim('\n', ' ', '\t')
    }

    // ---- internal: HF Jinja macro ports ----

    /**
     * Port of `format_function_declaration`. Emits:
     *   declaration:NAME{description:<|"|>...<|"|>[,parameters:{...}][,response:{...}]}
     */
    internal fun formatFunctionDeclaration(tool: ToolDefinition): String {
        val sb = StringBuilder()
        sb.append("declaration:").append(tool.name).append('{')
        sb.append("description:").append(QUOTE).append(tool.description).append(QUOTE)

        // parameters
        val params = tool.parameters
        val properties = params["properties"] as? JsonObject
        val required = (params["required"] as? JsonArray)?.mapNotNull {
            (it as? JsonPrimitive)?.contentOrNull()
        } ?: emptyList()
        val typeName = (params["type"] as? JsonPrimitive)?.contentOrNull()
        if (properties != null || required.isNotEmpty() || typeName != null) {
            sb.append(",parameters:{")
            var addComma = false
            if (properties != null && properties.isNotEmpty()) {
                // HF Jinja's `properties:{ {{- … -}} },` whitespace-strips
                // the spaces around the macro call (the `{{-` / `-}}`
                // markers eat surrounding whitespace), so the rendered
                // text is `properties:{…},` with no padding.
                sb.append("properties:{")
                formatProperties(sb, properties, required)
                sb.append("},")
                addComma = false
            }
            if (required.isNotEmpty()) {
                sb.append("required:[")
                for ((i, r) in required.withIndex()) {
                    if (i > 0) sb.append(',')
                    sb.append(QUOTE).append(r).append(QUOTE)
                }
                sb.append("],")
                addComma = false
            }
            if (typeName != null) {
                sb.append("type:").append(QUOTE).append(typeName.uppercase()).append(QUOTE)
                sb.append('}')
            } else {
                // Trim a trailing ',' if the object had props/required but no type
                if (sb.last() == ',') sb.setLength(sb.length - 1)
                sb.append('}')
            }
        }

        sb.append('}')
        return sb.toString()
    }

    /**
     * Port of `format_parameters`. Walks the JSON-schema `properties`
     * object and emits, for each property, `name:{...}` with the
     * description, optional enum/items/nullable, and finally
     * `type:<|"|>UPPER<|"|>}`.
     */
    private fun formatProperties(
        sb: StringBuilder,
        properties: JsonObject,
        required: List<String>
    ) {
        @Suppress("UNUSED_PARAMETER")
        // `required` is present in HF's signature but not consumed inside the
        // properties walk (it's emitted at the parent level). Kept for parity.
        val ignored = required

        var first = true
        // dictsort — stable alphabetical key order (matches HF Jinja default)
        val sortedKeys = properties.keys.sorted()
        for (key in sortedKeys) {
            val raw = properties[key] ?: continue
            val obj = raw as? JsonObject ?: continue

            if (!first) sb.append(',')
            first = false

            sb.append(key).append(":{")
            var addComma = false

            val description = (obj["description"] as? JsonPrimitive)?.contentOrNull()
            if (description != null) {
                sb.append("description:").append(QUOTE).append(description).append(QUOTE)
                addComma = true
            }

            val type = (obj["type"] as? JsonPrimitive)?.contentOrNull()
            val typeUpper = type?.uppercase()

            when (typeUpper) {
                "STRING" -> {
                    val enumArr = obj["enum"] as? JsonArray
                    if (enumArr != null) {
                        if (addComma) sb.append(',')
                        sb.append("enum:")
                        sb.append(formatArgument(enumArr, escapeKeys = true))
                        addComma = true
                    }
                }
                "ARRAY" -> {
                    val items = obj["items"] as? JsonObject
                    if (items != null) {
                        if (addComma) sb.append(',')
                        sb.append("items:{")
                        var itemsFirst = true
                        // HF iterates over items dict in dictsort order
                        for (ik in items.keys.sorted()) {
                            val iv = items[ik] ?: continue
                            if (!itemsFirst) sb.append(',')
                            itemsFirst = false
                            when (ik) {
                                "properties" -> {
                                    val p = iv as? JsonObject
                                    sb.append("properties:{")
                                    if (p != null) {
                                        val itemsRequired = (items["required"] as? JsonArray)
                                            ?.mapNotNull { (it as? JsonPrimitive)?.contentOrNull() }
                                            ?: emptyList()
                                        formatProperties(sb, p, itemsRequired)
                                    }
                                    sb.append('}')
                                }
                                "required" -> {
                                    val arr = iv as? JsonArray
                                    sb.append("required:[")
                                    if (arr != null) {
                                        for ((idx, r) in arr.withIndex()) {
                                            val rs = (r as? JsonPrimitive)?.contentOrNull() ?: continue
                                            if (idx > 0) sb.append(',')
                                            sb.append(QUOTE).append(rs).append(QUOTE)
                                        }
                                    }
                                    sb.append(']')
                                }
                                "type" -> {
                                    val tp = (iv as? JsonPrimitive)?.contentOrNull()
                                    if (tp != null) {
                                        sb.append("type:").append(formatArgument(JsonPrimitive(tp.uppercase()), escapeKeys = true))
                                    }
                                }
                                else -> {
                                    sb.append(ik).append(':').append(formatArgument(iv, escapeKeys = true))
                                }
                            }
                        }
                        sb.append('}')
                        addComma = true
                    }
                }
            }

            val nullable = (obj["nullable"] as? JsonPrimitive)?.booleanOrNull
            if (nullable == true) {
                if (addComma) sb.append(',')
                sb.append("nullable:true")
                addComma = true
            }

            if (typeUpper == "OBJECT") {
                val nested = obj["properties"] as? JsonObject
                val nestedRequired = (obj["required"] as? JsonArray)?.mapNotNull {
                    (it as? JsonPrimitive)?.contentOrNull()
                } ?: emptyList()
                if (nested != null) {
                    if (addComma) sb.append(',')
                    sb.append("properties:{")
                    formatProperties(sb, nested, nestedRequired)
                    sb.append('}')
                    addComma = true
                }
                if (nestedRequired.isNotEmpty()) {
                    if (addComma) sb.append(',')
                    sb.append("required:[")
                    for ((i, r) in nestedRequired.withIndex()) {
                        if (i > 0) sb.append(',')
                        sb.append(QUOTE).append(r).append(QUOTE)
                    }
                    sb.append(']')
                    addComma = true
                }
            }

            if (typeUpper != null) {
                if (addComma) sb.append(',')
                sb.append("type:").append(QUOTE).append(typeUpper).append(QUOTE)
            }
            sb.append('}')
        }
    }

    /**
     * Port of `format_argument`. Renders a JSON value to the chat-
     * template wire format. With `escapeKeys=true`, dict keys get the
     * `<|"|>` wrap (used in tool *declarations*); with `escapeKeys=false`,
     * dict keys are bare (used in tool *calls* and *responses*).
     */
    internal fun formatArgument(value: JsonElement, escapeKeys: Boolean): String {
        val sb = StringBuilder()
        appendArgument(sb, value, escapeKeys)
        return sb.toString()
    }

    private fun appendArgument(sb: StringBuilder, value: JsonElement, escapeKeys: Boolean) {
        when (value) {
            is JsonPrimitive -> {
                val b = value.booleanOrNull
                if (b != null && !value.isString) {
                    sb.append(if (b) "true" else "false")
                } else if (value.isString) {
                    sb.append(QUOTE).append(value.content).append(QUOTE)
                } else {
                    // bare number / null
                    sb.append(value.content)
                }
            }
            is JsonObject -> {
                sb.append('{')
                val keys = value.keys.sorted()
                for ((i, k) in keys.withIndex()) {
                    val v = value[k] ?: continue
                    if (i > 0) sb.append(',')
                    if (escapeKeys) sb.append(QUOTE).append(k).append(QUOTE) else sb.append(k)
                    sb.append(':')
                    appendArgument(sb, v, escapeKeys)
                }
                sb.append('}')
            }
            is JsonArray -> {
                sb.append('[')
                for ((i, v) in value.withIndex()) {
                    if (i > 0) sb.append(',')
                    appendArgument(sb, v, escapeKeys)
                }
                sb.append(']')
            }
        }
    }

    /**
     * Tool-call argument body — same as `appendArgument` for a JsonObject
     * but with the outer braces omitted (the caller already emitted `{` and
     * `}` around the body) and with bare keys (escapeKeys=false).
     */
    private fun formatArgumentsBareKeys(sb: StringBuilder, args: JsonObject) {
        val keys = args.keys.sorted()
        for ((i, k) in keys.withIndex()) {
            val v = args[k] ?: continue
            if (i > 0) sb.append(',')
            sb.append(k).append(':')
            appendArgument(sb, v, escapeKeys = false)
        }
    }

    /**
     * Port of `format_tool_response_block`. Maps the tool's text output
     * to either `response:NAME{key:value,...}` (when content parses as
     * JSON object) or `response:NAME{value:<text>}` (otherwise). HF's
     * macro accepts both shapes.
     */
    internal fun formatToolResponseBlock(toolName: String, content: String): String {
        val sb = StringBuilder()
        sb.append(TOOL_RESPONSE_OPEN)
        sb.append("response:").append(toolName).append('{')

        // Try to interpret content as a JSON object — that's the only
        // shape HF treats as a structured response (matching
        // `response is mapping`). Otherwise treat the entire content as
        // a string literal and wrap as `value:<|"|>...<|"|>`. We do NOT
        // auto-coerce numeric-looking strings to bare numbers; the tool
        // chose to return text, so render as text.
        val parsed = runCatching {
            kotlinx.serialization.json.Json.parseToJsonElement(content)
        }.getOrNull()

        if (parsed is JsonObject) {
            val keys = parsed.keys.sorted()
            for ((i, k) in keys.withIndex()) {
                val v = parsed[k] ?: continue
                if (i > 0) sb.append(',')
                sb.append(k).append(':')
                appendArgument(sb, v, escapeKeys = false)
            }
        } else {
            sb.append("value:").append(QUOTE).append(content).append(QUOTE)
        }
        sb.append('}')
        sb.append(TOOL_RESPONSE_CLOSE)
        return sb.toString()
    }

    private fun resolveToolName(msg: ChatMessage, all: List<ChatMessage>, idx: Int): String {
        val id = msg.toolCallId ?: return "unknown"
        // Walk backwards to find the assistant message whose toolCalls includes this id
        for (j in idx - 1 downTo 0) {
            val prev = all[j]
            val match = prev.toolCalls?.firstOrNull { it.id == id }
            if (match != null) return match.name
        }
        return "unknown"
    }

    private fun previousNonToolRole(messages: List<ChatMessage>, idx: Int): ChatRole? {
        for (j in idx - 1 downTo 0) {
            if (messages[j].role != ChatRole.TOOL) return messages[j].role
        }
        return null
    }

    /**
     * Parses a `call:NAME{key:value,...}` body recovered from inside a
     * `<|tool_call>...<tool_call|>` block. Returns null if the body
     * isn't well-formed.
     */
    private fun parseCallBody(rawBody: String): ToolCall? {
        var body = rawBody.trim()
        if (!body.startsWith("call:")) return null
        body = body.substring("call:".length)
        val openIdx = body.indexOf('{')
        if (openIdx < 0) return null
        val name = body.substring(0, openIdx).trim()
        if (name.isEmpty()) return null
        // Body between matching braces (the body string is already inside
        // the tool_call markers, so we expect the trailing brace at end).
        val argsBody = matchingBraceContents(body, openIdx) ?: return null
        val argsObj = parseArgsBareKeys(argsBody) ?: return null
        return ToolCall(
            id = ToolCallParser.generateCallId(),
            name = name,
            arguments = argsObj
        )
    }

    /**
     * Returns the contents inside the brace at `openIdx`, or null if
     * unbalanced. Handles nested braces and `<|"|>...<|"|>` quoted
     * strings (whose contents are not parsed for braces).
     */
    private fun matchingBraceContents(s: String, openIdx: Int): String? {
        require(s[openIdx] == '{')
        var depth = 1
        var i = openIdx + 1
        while (i < s.length) {
            // Skip past quoted strings
            if (s.regionMatches(i, QUOTE, 0, QUOTE.length)) {
                val end = s.indexOf(QUOTE, i + QUOTE.length)
                if (end < 0) return null
                i = end + QUOTE.length
                continue
            }
            when (s[i]) {
                '{' -> depth++
                '}' -> {
                    depth--
                    if (depth == 0) return s.substring(openIdx + 1, i)
                }
            }
            i++
        }
        return null
    }

    /**
     * Parses a comma-separated `key:value` body (the inside of a tool-
     * call or tool-response argument block) into a JsonObject. Bare keys,
     * `<|"|>...<|"|>` strings, bare numbers/booleans, nested objects and
     * arrays.
     */
    private fun parseArgsBareKeys(body: String): JsonObject? {
        val parser = ArgParser(body)
        val obj = parser.parseBareKeyMap() ?: return null
        if (!parser.atEnd()) return null
        return obj
    }

    private class ArgParser(private val s: String) {
        private var i = 0

        fun atEnd(): Boolean { skipWs(); return i >= s.length }

        fun parseBareKeyMap(): JsonObject? {
            skipWs()
            if (i >= s.length) return JsonObject(emptyMap())
            val out = mutableMapOf<String, JsonElement>()
            while (i < s.length) {
                skipWs()
                if (i >= s.length) break
                val key = readBareKey() ?: return null
                skipWs()
                if (i >= s.length || s[i] != ':') return null
                i++  // consume ':'
                val v = parseValue() ?: return null
                out[key] = v
                skipWs()
                if (i < s.length && s[i] == ',') { i++; continue }
                break
            }
            return JsonObject(out)
        }

        fun parseValue(): JsonElement? {
            skipWs()
            if (i >= s.length) return null
            return when {
                s.regionMatches(i, QUOTE, 0, QUOTE.length) -> readQuotedString()?.let { JsonPrimitive(it) }
                s[i] == '{' -> { i++; val o = parseBareKeyMap(); skipWs(); if (i < s.length && s[i] == '}') i++; o }
                s[i] == '[' -> readArray()
                else -> readScalar()
            }
        }

        private fun readBareKey(): String? {
            val start = i
            while (i < s.length && (s[i].isLetterOrDigit() || s[i] == '_')) i++
            return if (i > start) s.substring(start, i) else null
        }

        private fun readQuotedString(): String? {
            if (!s.regionMatches(i, QUOTE, 0, QUOTE.length)) return null
            i += QUOTE.length
            val end = s.indexOf(QUOTE, i)
            if (end < 0) return null
            val v = s.substring(i, end)
            i = end + QUOTE.length
            return v
        }

        private fun readArray(): JsonElement? {
            require(s[i] == '[')
            i++
            val items = mutableListOf<JsonElement>()
            skipWs()
            if (i < s.length && s[i] == ']') { i++; return JsonArray(items) }
            while (i < s.length) {
                val v = parseValue() ?: return null
                items += v
                skipWs()
                if (i < s.length && s[i] == ',') { i++; continue }
                if (i < s.length && s[i] == ']') { i++; return JsonArray(items) }
                return null
            }
            return null
        }

        private fun readScalar(): JsonElement? {
            val start = i
            while (i < s.length) {
                val c = s[i]
                if (c == ',' || c == '}' || c == ']') break
                i++
            }
            val token = s.substring(start, i).trim()
            if (token.isEmpty()) return null
            return when (token) {
                "true" -> JsonPrimitive(true)
                "false" -> JsonPrimitive(false)
                "null" -> JsonPrimitive(null as String?)
                else -> {
                    val asLong = token.toLongOrNull()
                    if (asLong != null) JsonPrimitive(asLong) else {
                        val asDouble = token.toDoubleOrNull()
                        if (asDouble != null) JsonPrimitive(asDouble) else JsonPrimitive(token)
                    }
                }
            }
        }

        private fun skipWs() { while (i < s.length && s[i].isWhitespace()) i++ }
    }

    private companion object {
        // Atomic special tokens used by Gemma 4. All present in the GGUF vocab
        // as either CONTROL (type=3) or USER_DEFINED (type=4) — both atomic.
        const val BOS = "<bos>"
        const val TURN_CLOSE = "<turn|>"
        const val TOOL_OPEN = "<|tool>"
        const val TOOL_CLOSE = "<tool|>"
        const val TOOL_CALL_OPEN = "<|tool_call>"
        const val TOOL_CALL_CLOSE = "<tool_call|>"
        const val TOOL_RESPONSE_OPEN = "<|tool_response>"
        const val TOOL_RESPONSE_CLOSE = "<tool_response|>"
        const val CHANNEL_OPEN = "<|channel>"
        const val CHANNEL_CLOSE = "<channel|>"
        // Quote token — USER_DEFINED in the GGUF vocab. Custom Gemma 4
        // string-literal delimiter used everywhere in the chat template.
        const val QUOTE = "<|\"|>"

        // contentOrNull on JsonPrimitive: returns null for `null` or non-string
        // primitives. Local helper because kotlinx.serialization doesn't ship one.
        private fun JsonPrimitive.contentOrNull(): String? =
            if (isString || booleanOrNull == null && content != "null") content else null
    }
}
