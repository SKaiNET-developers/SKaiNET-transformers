package sk.ainet.transformers.gemma.iree

import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import sk.ainet.apps.kllama.chat.ToolCallParser
import sk.ainet.apps.kllama.chat.ToolCallParserStrategy
import sk.ainet.apps.kllama.chat.ToolCall as AgentToolCall

/**
 * [ToolCallParserStrategy] over FunctionGemma's compact functional-token
 * format, bridging [CompactToolCodec] into llm-agent's tool-calling
 * architecture:
 *
 * ```
 * <tool_0>(state="on")<end>  ->  ToolCall(name = "set_lights", arguments = {"state": "on"})
 * ```
 *
 * The compact format carries flat string-valued named args, so every argument
 * is surfaced as a [JsonPrimitive] string. `<tool_none>` (the fine-tune's
 * explicit no-op) parses to no call, matching [CompactCodec] semantics.
 *
 * @param codec Codec holding the token → tool-name vocabulary. Inject a
 *   [CompactToolCodec] with a custom map to support fine-tunes with a
 *   different tool set (issue #35/#36).
 */
public class FunctionGemmaToolCallParserStrategy(
    private val codec: CompactToolCodec = CompactToolCodec(),
) : ToolCallParserStrategy {

    override val formatName: String = "functiongemma"

    override fun parse(text: String): List<AgentToolCall> =
        codec.parse(text).map { call ->
            AgentToolCall(
                id = ToolCallParser.generateCallId(),
                name = call.tool,
                arguments = JsonObject(call.args.mapValues { (_, v) -> JsonPrimitive(v) }),
            )
        }

    override fun containsToolCall(text: String): Boolean = codec.containsCall(text)
}
