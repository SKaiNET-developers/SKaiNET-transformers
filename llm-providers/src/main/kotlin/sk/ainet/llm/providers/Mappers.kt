package sk.ainet.llm.providers

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import sk.ainet.llm.api.Message
import sk.ainet.llm.api.Role
import sk.ainet.llm.api.ToolCall as ApiToolCall
import sk.ainet.llm.api.ToolDefinition as ApiToolDefinition
import sk.ainet.apps.kllama.chat.ChatMessage
import sk.ainet.apps.kllama.chat.ChatRole
import sk.ainet.apps.kllama.chat.ToolCall as AgentToolCall
import sk.ainet.apps.kllama.chat.ToolDefinition as AgentToolDefinition

private val json = Json { ignoreUnknownKeys = true; isLenient = true }

internal fun Message.toAgent(): ChatMessage = ChatMessage(
    role = role.toAgent(),
    content = content,
    toolCalls = toolCalls.takeIf { it.isNotEmpty() }?.map { it.toAgent() },
    toolCallId = toolCallId,
)

internal fun Role.toAgent(): ChatRole = when (this) {
    Role.SYSTEM -> ChatRole.SYSTEM
    Role.USER -> ChatRole.USER
    Role.ASSISTANT -> ChatRole.ASSISTANT
    Role.TOOL -> ChatRole.TOOL
}

internal fun ApiToolDefinition.toAgent(): AgentToolDefinition = AgentToolDefinition(
    name = name,
    description = description,
    parameters = json.parseToJsonElement(parametersJsonSchema) as JsonObject,
)

internal fun ApiToolCall.toAgent(): AgentToolCall = AgentToolCall(
    id = id,
    name = name,
    arguments = json.parseToJsonElement(argumentsJson) as JsonObject,
)

internal fun AgentToolCall.toApi(): ApiToolCall = ApiToolCall(
    id = id,
    name = name,
    argumentsJson = arguments.toString(),
)
