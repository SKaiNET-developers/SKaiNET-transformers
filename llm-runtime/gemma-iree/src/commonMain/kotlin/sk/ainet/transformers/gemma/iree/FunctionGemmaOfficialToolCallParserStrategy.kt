package sk.ainet.transformers.gemma.iree

import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import sk.ainet.apps.kllama.chat.ToolCallParser
import sk.ainet.apps.kllama.chat.ToolCallParserStrategy
import sk.ainet.apps.kllama.chat.ToolCall as AgentToolCall

/**
 * [ToolCallParserStrategy] for the OFFICIAL `google/functiongemma-270m-it`
 * output format (as opposed to [FunctionGemmaToolCallParserStrategy]'s
 * Octopus-v2 `<tool_N>` compact format):
 *
 * ```
 * <think>the user wants the weather…</think>
 * <start_function_call>call:get_weather{location:<escape>Paris<escape>}<end_function_call>
 * ```
 *
 * Per the release's embedded chat template, string argument values are wrapped
 * in the `<escape>` special token (NOT quotes); numbers/booleans are bare
 * literals. Every value is surfaced as a [JsonPrimitive] string — the tool
 * dispatcher coerces per its own schema. `<think>` spans are ignored by the
 * matcher (the call regex simply doesn't match inside them unless the model
 * emits a call there, which the fine-tune does not).
 */
public class FunctionGemmaOfficialToolCallParserStrategy : ToolCallParserStrategy {

    override val formatName: String = "functiongemma-official"

    override fun parse(text: String): List<AgentToolCall> =
        parseCompact(text).map { call ->
            AgentToolCall(
                id = ToolCallParser.generateCallId(),
                name = call.tool,
                arguments = JsonObject(call.args.mapValues { (_, v) -> JsonPrimitive(v) }),
            )
        }

    /** The same calls as gemma-iree's own [ToolCall] type, for non-agent consumers. */
    public fun parseCompact(text: String): List<ToolCall> =
        CALL_RE.findAll(text).map { m ->
            val name = m.groupValues[1]
            val args = ARG_RE.findAll(m.groupValues[2]).associate { arg ->
                val key = arg.groupValues[1]
                val escaped = arg.groups[2]
                key to (escaped?.value ?: arg.groupValues[3].trim())
            }
            ToolCall(name, args)
        }.toList()

    override fun containsToolCall(text: String): Boolean = CALL_RE.containsMatchIn(text)

    public companion object {
        /** `<start_function_call>call:name{…}<end_function_call>` — end token optional on truncation. */
        // `[\s\S]` rather than `.` + DOT_MATCHES_ALL: that RegexOption is JVM-only, and this
        // file is commonMain. Call bodies span newlines whenever an argument does.
        private val CALL_RE = Regex(
            """<start_function_call>\s*call:([\w.-]+)\s*\{([\s\S]*?)}\s*(?:<end_function_call>|$)""",
        )

        /**
         * `key:<escape>string value<escape>` or `key:bare_literal` pairs inside
         * the call braces (the release's `format_argument` conventions).
         */
        private val ARG_RE = Regex("""([\w.-]+)\s*:\s*(?:<escape>(.*?)<escape>|([^,{}<]+))""")
    }
}
