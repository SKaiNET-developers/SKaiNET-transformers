package sk.ainet.llm.api

import kotlin.jvm.JvmOverloads
import kotlin.jvm.JvmStatic

/** Role of a message in a chat conversation. */
public enum class Role { SYSTEM, USER, ASSISTANT, TOOL }

/**
 * A single message in a chat conversation.
 *
 * @param role Speaker role.
 * @param content Text content. Empty string is allowed (e.g., assistant message that only contains tool calls).
 * @param toolCalls Tool calls emitted by an assistant message (empty otherwise).
 * @param toolCallId If [role] is [Role.TOOL], the id of the originating tool call this message responds to.
 * @param name Optional speaker name (some templates use it for tool messages).
 */
public data class Message @JvmOverloads constructor(
    public val role: Role,
    public val content: String,
    public val toolCalls: List<ToolCall> = emptyList(),
    public val toolCallId: String? = null,
    public val name: String? = null,
) {
    public companion object {
        @JvmStatic
        public fun system(content: String): Message = Message(Role.SYSTEM, content)

        @JvmStatic
        public fun user(content: String): Message = Message(Role.USER, content)

        @JvmStatic
        @JvmOverloads
        public fun assistant(content: String, toolCalls: List<ToolCall> = emptyList()): Message =
            Message(Role.ASSISTANT, content, toolCalls)

        @JvmStatic
        @JvmOverloads
        public fun tool(content: String, toolCallId: String, name: String? = null): Message =
            Message(Role.TOOL, content, toolCallId = toolCallId, name = name)
    }
}
