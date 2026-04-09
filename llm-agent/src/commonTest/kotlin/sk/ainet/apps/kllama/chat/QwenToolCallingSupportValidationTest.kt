package sk.ainet.apps.kllama.chat

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertNotNull
import kotlin.test.assertContains
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonObject

/**
 * Validates that the generalized ToolCallingSupport design works correctly
 * with existing Qwen support — exercising the full path from metadata-based
 * resolution through template creation, prompt formatting, and tool-call parsing.
 *
 * See issue #43.
 */
class QwenToolCallingSupportValidationTest {

    private val qwenMetadata = ModelMetadata(
        family = "qwen",
        architecture = "qwen2",
        chatTemplate = "{% if messages[0].role == 'system' %}Qwen...",
        sourceFormat = "gguf"
    )

    private val sampleTool = ToolDefinition(
        name = "list_files",
        description = "List files in a directory",
        parameters = buildJsonObject {
            put("type", "object")
            putJsonObject("properties") {
                putJsonObject("path") {
                    put("type", "string")
                    put("description", "Directory path")
                }
            }
        }
    )

    @Test
    fun resolverSelectsQwenProvider() {
        val provider = ToolCallingSupportResolver.resolve(qwenMetadata)
        assertNotNull(provider)
        assertIs<QwenToolCallingSupport>(provider)
        assertEquals("qwen", provider.family)
    }

    @Test
    fun qwenProviderModeIsNative() {
        val provider = ToolCallingSupportResolver.resolve(qwenMetadata)!!
        assertEquals(ToolCallingMode.NATIVE, provider.toolCallingMode(qwenMetadata))
    }

    @Test
    fun qwenProviderCreatesCorrectTemplate() {
        val provider = ToolCallingSupportResolver.resolve(qwenMetadata)!!
        val template = provider.createChatTemplate()
        assertIs<QwenChatTemplate>(template)
    }

    @Test
    fun qwenTemplateFormatsToolDefinitions() {
        val provider = ToolCallingSupportResolver.resolve(qwenMetadata)!!
        val template = provider.createChatTemplate()

        val prompt = template.apply(
            messages = listOf(ChatMessage(ChatRole.USER, "List files in /tmp")),
            tools = listOf(sampleTool)
        )

        assertContains(prompt, "<tools>")
        assertContains(prompt, "\"name\":\"list_files\"")
        assertContains(prompt, "\"type\":\"function\"")
        assertContains(prompt, "<|im_start|>user")
    }

    @Test
    fun qwenProviderParsesToolCalls() {
        val provider = ToolCallingSupportResolver.resolve(qwenMetadata)!!
        val modelOutput = """
            <tool_call>
            {"name": "list_files", "arguments": {"path": "/tmp"}}
            </tool_call>
        """.trimIndent()

        val calls = provider.parseToolCalls(modelOutput)
        assertEquals(1, calls.size)
        assertEquals("list_files", calls[0].name)
        assertEquals("/tmp", calls[0].arguments["path"]?.toString()?.trim('"'))
    }

    @Test
    fun qwenResolvedViaExplicitFamilyAlso() {
        // Backward compat: explicit "qwen" still works
        val provider = ToolCallingSupportResolver.resolve(
            metadata = ModelMetadata(),
            explicitFamily = "qwen"
        )
        assertNotNull(provider)
        assertIs<QwenToolCallingSupport>(provider)
    }

    @Test
    fun qwenSpecificLogicNotInSharedCode() {
        // GenericToolCallingSupport should NOT behave like Qwen
        val generic = GenericToolCallingSupport()
        val template = generic.createChatTemplate()
        assertIs<ChatMLTemplate>(template)

        // ChatML template should NOT have Qwen-specific preamble
        val prompt = template.apply(
            messages = listOf(ChatMessage(ChatRole.USER, "hello")),
            tools = listOf(sampleTool)
        )
        assert(!prompt.contains("Alibaba Cloud")) {
            "Generic fallback should not contain Qwen-specific text"
        }
    }

    @Test
    fun fullResolutionPathEndToEnd() {
        // Simulate: load model metadata → resolve provider → create template → format → parse
        val metadata = ModelMetadata(family = "qwen", sourceFormat = "gguf")

        val provider = ToolCallingSupportResolver.resolveOrFallback(metadata)
        assertIs<QwenToolCallingSupport>(provider)

        val template = provider.createChatTemplate()
        val prompt = template.apply(
            messages = listOf(
                ChatMessage(ChatRole.SYSTEM, "You are helpful."),
                ChatMessage(ChatRole.USER, "What files are in /tmp?")
            ),
            tools = listOf(sampleTool)
        )

        // Verify prompt is well-formed Qwen format
        assertContains(prompt, "<|im_start|>system")
        assertContains(prompt, "<tools>")
        assertContains(prompt, "<|im_start|>user")
        assertContains(prompt, "<|im_start|>assistant")

        // Simulate model output and parse
        val simulatedOutput = """<tool_call>
{"name": "list_files", "arguments": {"path": "/tmp"}}
</tool_call>"""
        val calls = template.parseToolCalls(simulatedOutput)
        assertEquals(1, calls.size)
        assertEquals("list_files", calls[0].name)
    }
}
