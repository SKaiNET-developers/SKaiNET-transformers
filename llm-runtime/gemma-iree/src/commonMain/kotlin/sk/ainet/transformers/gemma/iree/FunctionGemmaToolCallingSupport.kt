package sk.ainet.transformers.gemma.iree

import sk.ainet.apps.kllama.chat.ChatTemplate
import sk.ainet.apps.kllama.chat.ModelMetadata
import sk.ainet.apps.kllama.chat.ToolCallingMode
import sk.ainet.apps.kllama.chat.ToolCallingSupport
import sk.ainet.apps.kllama.chat.ToolCall as AgentToolCall

/**
 * [ToolCallingSupport] for the FunctionGemma fine-tune (issues #35/#36): the
 * compact `<tool_N>(k="v")<end>` functional-token format on a Gemma 3 270M
 * base, wired into llm-agent's provider architecture.
 *
 * Not registered in [sk.ainet.apps.kllama.chat.ToolCallingSupportResolver]'s
 * built-in list — that lives in `:llm-agent`, which cannot depend on this
 * module. Consumers opt in at startup:
 *
 * ```kotlin
 * ToolCallingSupportResolver.register(FunctionGemmaToolCallingSupport())
 * ```
 *
 * `register` prepends, so this provider is consulted BEFORE the generic
 * `GemmaToolCallingSupport` — necessary because FunctionGemma checkpoints
 * report a plain `gemma3` architecture that the generic provider would
 * otherwise claim (handing out the JSON `functionCall` template the fine-tune
 * was never trained on).
 *
 * @param codec Token → tool-name vocabulary. Default is the stock v10 six-tool
 *   set; inject `CompactToolCodec(customMap)` to serve a fine-tune with a
 *   different or extended vocabulary without a library edit.
 */
public class FunctionGemmaToolCallingSupport(
    codec: CompactToolCodec = CompactToolCodec(),
) : ToolCallingSupport {

    override val family: String = "functiongemma"

    private val parser = FunctionGemmaToolCallParserStrategy(codec)
    private val template = FunctionGemmaChatTemplate(parser)

    override fun supports(metadata: ModelMetadata): Boolean {
        val f = metadata.family?.lowercase()
        if (f == "functiongemma") return true
        // The architecture field is NOT a discriminator: FunctionGemma reports
        // the base "gemma3" arch. The compact functional tokens are — the
        // fine-tune's tokenizer carries <tool_0>..<tool_N>/<tool_none> as
        // special tokens, which no stock Gemma checkpoint has.
        if (metadata.tokenizerHints.any { it.startsWith("<tool_") }) return true
        val tpl = metadata.chatTemplate ?: return false
        return tpl.contains("<tool_", ignoreCase = true)
    }

    override fun createChatTemplate(): ChatTemplate = template

    override fun parseToolCalls(content: String): List<AgentToolCall> = parser.parse(content)

    override fun toolCallingMode(metadata: ModelMetadata): ToolCallingMode = ToolCallingMode.NATIVE
}
