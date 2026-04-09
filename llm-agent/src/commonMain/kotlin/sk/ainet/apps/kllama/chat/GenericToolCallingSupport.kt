package sk.ainet.apps.kllama.chat

/**
 * Generic / fallback tool-calling provider.
 *
 * Uses [ChatMLTemplate] (the most widely adopted chat format) and the
 * default [ToolCallParser] so that models which are loosely compatible
 * can still attempt tool calling without a dedicated provider.
 *
 * The [toolCallingMode] always returns [ToolCallingMode.GENERIC] so
 * callers can clearly distinguish fallback usage from native support.
 */
public class GenericToolCallingSupport : ToolCallingSupport {

    override val family: String = "generic"

    /**
     * The generic fallback never matches automatically — it must be
     * selected explicitly via [ToolCallingSupportResolver.resolveOrFallback].
     */
    override fun supports(metadata: ModelMetadata): Boolean = false

    override fun createChatTemplate(): ChatTemplate = ChatMLTemplate()

    override fun parseToolCalls(content: String): List<ToolCall> =
        ToolCallParser.parse(content)

    override fun toolCallingMode(metadata: ModelMetadata): ToolCallingMode =
        ToolCallingMode.GENERIC
}
