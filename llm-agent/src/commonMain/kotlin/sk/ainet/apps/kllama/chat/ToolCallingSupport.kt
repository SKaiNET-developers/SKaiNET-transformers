package sk.ainet.apps.kllama.chat

/**
 * Abstraction for model-family-specific tool-calling behaviour.
 *
 * Each supported model family provides an implementation that knows how to
 * format tool definitions, parse tool calls from generated text, and create
 * the appropriate [ChatTemplate].
 *
 * New families can be added without touching core agent logic — just
 * implement this interface and register the provider.
 */
public interface ToolCallingSupport {

    /** Identifier for this model family, e.g. "llama3", "qwen", "gemma". */
    public val family: String

    /**
     * Return `true` if this provider can handle the model described by [metadata].
     *
     * Implementations typically inspect [ModelMetadata.family],
     * [ModelMetadata.chatTemplate], or [ModelMetadata.tokenizerHints].
     */
    public fun supports(metadata: ModelMetadata): Boolean

    /** Create the [ChatTemplate] used for prompt formatting. */
    public fun createChatTemplate(): ChatTemplate

    /** Parse tool calls from the model's generated [content]. */
    public fun parseToolCalls(content: String): List<ToolCall>

    /**
     * Determine the tool-calling mode for the given model.
     *
     * Most native providers return [ToolCallingMode.NATIVE].
     */
    public fun toolCallingMode(metadata: ModelMetadata): ToolCallingMode
}

// ---------------------------------------------------------------------------
// Built-in providers
// ---------------------------------------------------------------------------

/** Tool-calling support for the Llama 3 / 3.1 / 3.2 family. */
public class Llama3ToolCallingSupport : ToolCallingSupport {
    override val family: String = "llama3"

    override fun supports(metadata: ModelMetadata): Boolean {
        val f = metadata.family?.lowercase()
        if (f == "llama3" || f == "llama") return true
        val tpl = metadata.chatTemplate ?: return false
        return tpl.contains("<|start_header_id|>")
    }

    override fun createChatTemplate(): ChatTemplate = Llama3ChatTemplate()

    override fun parseToolCalls(content: String): List<ToolCall> =
        ToolCallParser.parse(content)

    override fun toolCallingMode(metadata: ModelMetadata): ToolCallingMode =
        ToolCallingMode.NATIVE
}

/** Tool-calling support for ChatML / Hermes models. */
public class ChatMLToolCallingSupport : ToolCallingSupport {
    override val family: String = "chatml"

    override fun supports(metadata: ModelMetadata): Boolean {
        val f = metadata.family?.lowercase()
        if (f == "chatml" || f == "hermes") return true
        val tpl = metadata.chatTemplate ?: return false
        // ChatML uses <|im_start|> but is NOT Qwen (which also uses these tokens)
        return tpl.contains("<|im_start|>") && !tpl.contains("Qwen")
    }

    override fun createChatTemplate(): ChatTemplate = ChatMLTemplate()

    override fun parseToolCalls(content: String): List<ToolCall> =
        ToolCallParser.parse(content)

    override fun toolCallingMode(metadata: ModelMetadata): ToolCallingMode =
        ToolCallingMode.NATIVE
}

/**
 * Tool-calling support for Qwen3.5 family with canonical XML-like tool call format.
 *
 * Qwen3.5 emits tool calls as `<tool_call><function=...><parameter=...>` blocks,
 * which differs from the JSON-in-XML format used by Qwen3 and earlier.
 * This provider is checked before [QwenToolCallingSupport] so that Qwen3.5 models
 * get the correct template and parser.
 */
public class Qwen35ToolCallingSupport : ToolCallingSupport {
    override val family: String = "qwen35"

    private val qwen35Parser = Qwen35ToolCallParserStrategy()
    private val hermesParser = HermesToolCallParserStrategy()

    override fun supports(metadata: ModelMetadata): Boolean {
        val f = metadata.family?.lowercase()
        if (f == "qwen35" || f == "qwen3.5") return true
        val arch = metadata.architecture?.lowercase()
        if (arch == "qwen35" || arch == "qwen3_5" || arch == "qwen3_5_moe") return true
        return false
    }

    override fun createChatTemplate(): ChatTemplate = Qwen35ChatTemplate()

    override fun parseToolCalls(content: String): List<ToolCall> {
        // Try Qwen3.5 format first, fall back to Hermes JSON-in-XML
        val calls = qwen35Parser.parse(content)
        if (calls.isNotEmpty()) return calls
        return hermesParser.parse(content)
    }

    override fun toolCallingMode(metadata: ModelMetadata): ToolCallingMode =
        ToolCallingMode.NATIVE
}

/** Tool-calling support for Qwen 2 / 3 family (JSON-in-XML format). */
public class QwenToolCallingSupport : ToolCallingSupport {
    override val family: String = "qwen"

    override fun supports(metadata: ModelMetadata): Boolean {
        val f = metadata.family?.lowercase()
        if (f == "qwen") return true
        val tpl = metadata.chatTemplate ?: return false
        return tpl.contains("Qwen")
    }

    override fun createChatTemplate(): ChatTemplate = QwenChatTemplate()

    override fun parseToolCalls(content: String): List<ToolCall> =
        ToolCallParser.parse(content)

    override fun toolCallingMode(metadata: ModelMetadata): ToolCallingMode =
        ToolCallingMode.NATIVE
}

/** Tool-calling support for Gemma 2 / 3 family with functionCall format. */
public class GemmaToolCallingSupport : ToolCallingSupport {
    override val family: String = "gemma"

    private val template = GemmaChatTemplate()

    override fun supports(metadata: ModelMetadata): Boolean {
        val f = metadata.family?.lowercase()
        if (f == "gemma") return true
        val arch = metadata.architecture?.lowercase()
        if (arch != null && arch.startsWith("gemma")) return true
        val tpl = metadata.chatTemplate ?: return false
        return tpl.contains("<start_of_turn>")
    }

    override fun createChatTemplate(): ChatTemplate = template

    override fun parseToolCalls(content: String): List<ToolCall> =
        template.parseToolCalls(content)

    override fun toolCallingMode(metadata: ModelMetadata): ToolCallingMode =
        ToolCallingMode.NATIVE
}
