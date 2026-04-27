package sk.ainet.llm.providers

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import sk.ainet.apps.kllama.chat.ChatRole
import sk.ainet.apps.kllama.chat.ToolCall as AgentToolCall
import sk.ainet.llm.api.Message
import sk.ainet.llm.api.Role
import sk.ainet.llm.api.ToolCall as ApiToolCall
import sk.ainet.llm.api.ToolDefinition as ApiToolDefinition

class MappersTest {

    @Test fun `Role enum maps to ChatRole 1to1`() {
        assertEquals(ChatRole.SYSTEM, Role.SYSTEM.toAgent())
        assertEquals(ChatRole.USER, Role.USER.toAgent())
        assertEquals(ChatRole.ASSISTANT, Role.ASSISTANT.toAgent())
        assertEquals(ChatRole.TOOL, Role.TOOL.toAgent())
    }

    @Test fun `Message preserves content, role, toolCallId`() {
        val msg = Message.tool(content = "42", toolCallId = "call-7", name = "calc")
        val agent = msg.toAgent()
        assertEquals(ChatRole.TOOL, agent.role)
        assertEquals("42", agent.content)
        assertEquals("call-7", agent.toolCallId)
    }

    @Test fun `ToolDefinition parametersJsonSchema parses to JsonObject`() {
        val td = ApiToolDefinition(
            name = "search",
            description = "Web search",
            parametersJsonSchema = """{"type":"object","properties":{"q":{"type":"string"}}}""",
        )
        val agent = td.toAgent()
        assertEquals("search", agent.name)
        assertEquals("Web search", agent.description)
        assertEquals("object", (agent.parameters["type"] as JsonPrimitive).content)
    }

    @Test fun `agent ToolCall round-trips through API ToolCall`() {
        val original = AgentToolCall(
            id = "tc-1",
            name = "lookup",
            arguments = buildJsonObject { /* empty */ },
        )
        val viaApi = original.toApi().toAgent()
        assertEquals(original.id, viaApi.id)
        assertEquals(original.name, viaApi.name)
        assertEquals(original.arguments, viaApi.arguments)
    }

    @Test fun `API ToolCall back-and-forth preserves arguments JSON`() {
        val api = ApiToolCall(
            id = "tc-2",
            name = "lookup",
            argumentsJson = """{"key":"value","n":7}""",
        )
        val agent = api.toAgent()
        // Serialized form may reorder keys; check semantic equality via re-parse.
        assertEquals(api.id, agent.id)
        assertEquals(api.name, agent.name)
        assertEquals("value", (agent.arguments["key"] as JsonPrimitive).content)
        assertEquals(7, (agent.arguments["n"] as JsonPrimitive).content.toInt())
    }
}
