package sk.ainet.transformers.gemma.iree

/**
 * A tool call decoded from the model's output: the tool the model chose + the
 * named args it emitted. This is the runtime's OWN type — it deliberately does
 * NOT depend on any consuming app's action/intent type, so the module stays
 * reusable. The consumer maps [ToolCall] onto whatever action type it uses.
 */
public data class ToolCall(
    val tool: String,
    val args: Map<String, String> = emptyMap(),
)

/**
 * Decodes FunctionGemma's compact tool-call format into [ToolCall]s (v10
 * Octopus-v2 named-arg style, which the FunctionGemma fine-tune emits):
 *   <tool_0>(state="on")<end>          -> ToolCall("set_lights", {state:on})
 *   <tool_4>(metric="all")<end>        -> ToolCall("get_system_status", {metric:all})
 *   <tool_5>(message="hi")<end>        -> ToolCall("respond", {message:hi})
 */
public object CompactCodec {
    private val TOKEN_TO_NAME: Map<String, String?> = mapOf(
        "0" to "set_lights", "1" to "play_buzzer", "2" to "set_alarm",
        "3" to "cancel_alarm", "4" to "get_system_status", "5" to "respond",
        "none" to null,
    )
    private val CALL_RE = Regex("""<tool_(\d+|none)>\(([^)]*)\)(?:<end>)?""")
    private val NAMED_ARG_RE = Regex("""(\w+)\s*=\s*"([^"]*)"""")

    /** Parse all tool calls in [raw] (special-token text from the model). */
    public fun parse(raw: String): List<ToolCall> =
        CALL_RE.findAll(raw).mapNotNull { m ->
            val name = TOKEN_TO_NAME[m.groupValues[1]] ?: return@mapNotNull null
            val args = NAMED_ARG_RE.findAll(m.groupValues[2])
                .associate { it.groupValues[1] to it.groupValues[2] }
            ToolCall(name, args)
        }.toList()
}
