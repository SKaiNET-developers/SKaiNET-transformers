package sk.ainet.apps.kllama.chat

import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Structural tests for [ApertusChatTemplate]. These don't assert
 * byte-for-byte parity with the upstream Jinja template (a real
 * parity harness would need a Jinja runtime, which is JVM-only and
 * not available in commonTest); instead they verify the canonical
 * tokens land in the expected order for the four scenarios called
 * out in `docs/specs/apertus-chat-template.md`.
 *
 * The canonical cases:
 *   1. user-only (default system + disabled developer block)
 *   2. system + user (caller-supplied system)
 *   3. system + user + assistant string content
 *   4. system + user + assistant with tool calls + tool message
 */
class ApertusChatTemplateTest {

    private val fixedDate = "2026-05-01"

    private fun template(): ApertusChatTemplate =
        ApertusChatTemplate(currentDate = fixedDate)

    @Test
    fun case1_user_only_emits_default_system_and_disabled_developer_block() {
        val out = template().apply(
            messages = listOf(
                ChatMessage(role = ChatRole.USER, content = "Hi")
            ),
            tools = emptyList(),
            addGenerationPrompt = true,
        )
        // Sequence is: bos + system block (default) + developer block + user + assistant_start
        val expected = buildString {
            append("<s>")
            append("<|system_start|>")
            append("You are Apertus, a helpful assistant created by the SwissAI initiative.\n")
            append("Knowledge cutoff: 2024-04\n")
            append("Current date: $fixedDate")
            append("<|system_end|>")
            append("<|developer_start|>")
            append("Deliberation: disabled\n")
            append("Tool Capabilities: disabled")
            append("<|developer_end|>")
            append("<|user_start|>Hi<|user_end|>")
            append("<|assistant_start|>")
        }
        assertEquals(expected, out)
    }

    @Test
    fun case2_caller_supplied_system_replaces_default() {
        val out = template().apply(
            messages = listOf(
                ChatMessage(role = ChatRole.SYSTEM, content = "You are a calculator."),
                ChatMessage(role = ChatRole.USER, content = "1+1?"),
            ),
            tools = emptyList(),
            addGenerationPrompt = true,
        )
        assertTrue(out.startsWith("<s><|system_start|>You are a calculator.<|system_end|>"), "system override missing: $out")
        assertTrue(out.contains("<|developer_start|>Deliberation: disabled\nTool Capabilities: disabled<|developer_end|>"))
        assertTrue(out.contains("<|user_start|>1+1?<|user_end|>"))
        assertTrue(out.endsWith("<|assistant_start|>"))
        assertTrue(!out.contains("Apertus"), "default system message must not leak when caller supplied one: $out")
    }

    @Test
    fun case3_assistant_string_content_round_trips() {
        val out = template().apply(
            messages = listOf(
                ChatMessage(role = ChatRole.SYSTEM, content = "Be helpful."),
                ChatMessage(role = ChatRole.USER, content = "Hi"),
                ChatMessage(role = ChatRole.ASSISTANT, content = "Hello there."),
            ),
            tools = emptyList(),
            addGenerationPrompt = true,
        )
        // Assistant turn opens, content, closes, then a fresh assistant_start for generation
        assertTrue(out.contains("<|assistant_start|>Hello there.<|assistant_end|>"), "assistant content not framed correctly: $out")
        assertTrue(out.endsWith("<|assistant_start|>"), "missing trailing generation prompt: $out")
    }

    @Test
    fun case3_assistant_string_content_no_generation_prompt() {
        val out = template().apply(
            messages = listOf(
                ChatMessage(role = ChatRole.SYSTEM, content = "Be helpful."),
                ChatMessage(role = ChatRole.USER, content = "Hi"),
                ChatMessage(role = ChatRole.ASSISTANT, content = "Hello there."),
            ),
            tools = emptyList(),
            addGenerationPrompt = false,
        )
        // No trailing assistant_start
        assertTrue(out.endsWith("<|assistant_end|>"), "should end with assistant_end when no generation prompt: $out")
    }

    @Test
    fun case4_tool_calls_in_assistant_emit_tools_prefix_array_suffix() {
        val calc = ToolDefinition(
            name = "calculator",
            description = "Evaluate an arithmetic expression.",
            parameters = buildJsonObject {
                put("type", "object")
                put("properties", buildJsonObject {
                    put("expression", buildJsonObject {
                        put("type", "string")
                        put("description", "Expression like '1 + 2'.")
                    })
                })
                put("required", buildJsonArray { add(JsonPrimitive("expression")) })
            }
        )
        val out = template().apply(
            messages = listOf(
                ChatMessage(role = ChatRole.SYSTEM, content = "Use tools."),
                ChatMessage(role = ChatRole.USER, content = "What is 1+1?"),
                ChatMessage(
                    role = ChatRole.ASSISTANT,
                    content = "",
                    toolCalls = listOf(
                        ToolCall(
                            id = "tc-0",
                            name = "calculator",
                            arguments = buildJsonObject { put("expression", "1+1") }
                        )
                    )
                ),
                ChatMessage(role = ChatRole.TOOL, content = "2"),
                ChatMessage(role = ChatRole.ASSISTANT, content = "The answer is 2."),
            ),
            tools = listOf(calc),
            addGenerationPrompt = true,
        )
        // Developer block lists the tool with TypeScript syntax
        assertTrue(
            out.contains("Tool Capabilities:\n// Evaluate an arithmetic expression.\ntype calculator = (_: {"),
            "tool capabilities not rendered as TypeScript: $out",
        )
        assertTrue(out.contains("expression: string"), "expression param missing: $out")
        // Assistant emits tool_calls block
        assertTrue(
            out.contains("""<|tools_prefix|>[{"calculator": {"expression":"1+1"}}]<|tools_suffix|>"""),
            "tool call block missing or malformed: $out",
        )
        // Tool message renders as bracketed output continuing the same assistant turn
        assertTrue(out.contains("<|tools_suffix|>[2]"), "tool output bracket missing: $out")
        // Final response continues the SAME assistant turn (closes only at the end)
        assertTrue(out.contains("The answer is 2."), "final response missing: $out")
    }

    @Test
    fun add_generation_prompt_false_omits_trailing_assistant_start() {
        val out = template().apply(
            messages = listOf(ChatMessage(role = ChatRole.USER, content = "Hi")),
            tools = emptyList(),
            addGenerationPrompt = false,
        )
        assertTrue(!out.endsWith("<|assistant_start|>"), "trailing assistant_start should be absent: $out")
    }

    @Test
    fun parseThinkingBlocks_extracts_inner_pairs() {
        val text = "answer prefix<|inner_prefix|>step 1<|inner_suffix|>middle<|inner_prefix|>step 2<|inner_suffix|>tail"
        val blocks = template().parseThinkingBlocks(text)
        assertEquals(listOf("step 1", "step 2"), blocks)
    }

    @Test
    fun stripThinking_removes_inner_blocks() {
        val text = "before<|inner_prefix|>thoughts<|inner_suffix|>after"
        assertEquals("beforeafter", template().stripThinking(text))
    }

    @Test
    fun stripThinking_keeps_unterminated_block_intact() {
        val text = "before<|inner_prefix|>oops"
        assertEquals(text, template().stripThinking(text))
    }

    @Test
    fun enable_thinking_flag_flips_developer_line() {
        val out = ApertusChatTemplate(currentDate = fixedDate, enableThinking = true).apply(
            messages = listOf(ChatMessage(role = ChatRole.USER, content = "Hi")),
            tools = emptyList(),
            addGenerationPrompt = true,
        )
        assertTrue(out.contains("Deliberation: enabled"), "enable_thinking=true should flip developer line: $out")
    }
}
