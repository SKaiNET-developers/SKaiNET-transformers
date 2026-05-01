package sk.ainet.apps.kllama.chat

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean

/**
 * Parses Apertus tool calls from model output.
 *
 * Apertus emits tool calls between special tokens:
 *
 *   `<|tools_prefix|>[{"<tool_name>": <args_json>}, ...]<|tools_suffix|>`
 *
 * - The bracket contains a JSON array.
 * - Each element is a JSON object with **exactly one** key: the tool
 *   name. The value is the args object.
 * - Multiple parallel tool calls in one assistant turn are
 *   comma-separated objects in the array.
 *
 * Args are real JSON (not stringified TypeScript), so we can parse
 * with `kotlinx.serialization`. See `docs/specs/apertus-chat-template.md`
 * for the format derivation.
 *
 * The parser tolerates leading / trailing whitespace inside the
 * bracket and ignores any text outside the `<|tools_prefix|>` /
 * `<|tools_suffix|>` markers.
 */
public object ApertusToolCallParserStrategy : ToolCallParserStrategy {

    override val formatName: String = "apertus"

    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
    }

    override fun parse(text: String): List<ToolCall> {
        val start = text.indexOf(ApertusChatTemplate.TOOLS_PREFIX)
        if (start < 0) return emptyList()
        val arrayStart = start + ApertusChatTemplate.TOOLS_PREFIX.length
        val end = text.indexOf(ApertusChatTemplate.TOOLS_SUFFIX, arrayStart)
        if (end < 0) return emptyList()
        val payload = text.substring(arrayStart, end).trim()
        if (payload.isEmpty()) return emptyList()

        val parsed = runCatching { json.parseToJsonElement(payload) }.getOrNull() ?: return emptyList()
        val array = parsed as? JsonArray ?: return emptyList()

        val calls = mutableListOf<ToolCall>()
        for ((index, element) in array.withIndex()) {
            val obj = element as? JsonObject ?: continue
            // Single-key object: name -> args
            if (obj.size != 1) continue
            val (name, argsElement) = obj.entries.first()
            val args = argsElement as? JsonObject ?: continue
            calls.add(
                ToolCall(
                    id = "apertus-tool-call-$index",
                    name = name,
                    arguments = args,
                )
            )
        }
        return calls
    }

    override fun containsToolCall(text: String): Boolean =
        text.contains(ApertusChatTemplate.TOOLS_PREFIX) &&
            text.contains(ApertusChatTemplate.TOOLS_SUFFIX)
}
