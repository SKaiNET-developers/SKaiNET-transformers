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

/**
 * Tool-calling support for the Llama 3 / 3.1 / 3.2 family.
 *
 * @param format Tool-calling response format the chat template instructs the
 *   model to emit. Defaults to [Llama3ToolFormat.JSON] (Llama 3.2 default).
 *   See [Llama3ToolFormat] and `docs/llama3-tool-calling.md`.
 */
public class Llama3ToolCallingSupport(
    private val format: Llama3ToolFormat = Llama3ToolFormat.JSON
) : ToolCallingSupport {
    override val family: String = "llama3"

    override fun supports(metadata: ModelMetadata): Boolean {
        val f = metadata.family?.lowercase()
        if (f == "llama3" || f == "llama") return true
        val tpl = metadata.chatTemplate ?: return false
        return tpl.contains("<|start_header_id|>")
    }

    override fun createChatTemplate(): ChatTemplate = Llama3ChatTemplate(format)

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

/** Tool-calling support for Qwen 3 / 3.5 family. */
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
