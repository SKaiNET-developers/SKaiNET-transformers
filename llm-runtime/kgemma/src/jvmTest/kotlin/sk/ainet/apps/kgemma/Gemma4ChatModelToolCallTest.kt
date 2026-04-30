package sk.ainet.apps.kgemma

import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.io.path.isDirectory
import kotlin.test.Test
import kotlin.test.assertTrue
import sk.ainet.llm.api.ChatOptions
import sk.ainet.llm.api.ChatRequest
import sk.ainet.llm.api.FinishReason
import sk.ainet.llm.api.Message
import sk.ainet.llm.api.ToolDefinition

/**
 * Regression smoke for tool calling against a real Gemma 4 E2B SafeTensors
 * checkpoint via [Gemma4ChatModel.fromSafeTensors].
 *
 * Counterpart of the GGUF-flavoured `Gemma4E2BToolCallSmokeTest`, which is
 * `@Ignore`d on a separate Q4_K_M parity question. The SafeTensors path no
 * longer suffers the prompt-corruption issue (`GGUFTokenizer.fromTokenizerJson`
 * was patched to recognise `added_tokens` as atomic CONTROL tokens), so the
 * model receives the chat-template grammar the trainer used and is expected
 * to emit `<|tool_call>call:NAME{...}<tool_call|>` on a calculator-style
 * prompt.
 *
 * Self-skips when `GEMMA4_E2B_SAFETENSORS_PATH` is not set.
 *
 * Asserts shape, not exact wording: any `<|tool_call>` marker in the
 * generated text — or any parsed tool call in the response — counts as a
 * pass. Don't byte-equality-check the arguments JSON; the goal here is the
 * grammar, not the arithmetic.
 */
class Gemma4ChatModelToolCallTest {

    @Test
    fun `chat model emits a tool_call against a calculator tool`() {
        val indexPath = locateCheckpoint() ?: return

        val calculator = ToolDefinition(
            name = "calculator",
            description = "Evaluate a mathematical expression. Supports +, -, *, / and parentheses.",
            parametersJsonSchema = """
                {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate, e.g. '17 * 23'"
                        }
                    }
                }
            """.trimIndent(),
        )

        val model = Gemma4ChatModel.fromSafeTensors(
            indexPath = indexPath.toString(),
            options = ChatOptions(
                temperature = 0f,
                maxTokens = 64,
            ),
        )

        try {
            val response = model.call(
                ChatRequest(
                    messages = listOf(
                        Message.user("What is 17 * 23? Use the calculator tool."),
                    ),
                    tools = listOf(calculator),
                ),
            )

            val generation = response.generations.firstOrNull()
            val text = generation?.message?.content.orEmpty()
            val toolCalls = generation?.message?.toolCalls.orEmpty()
            val finish = generation?.finishReason

            println("[toolcall-smoke] finish=$finish")
            println("[toolcall-smoke] toolCalls.size=${toolCalls.size}")
            println("[toolcall-smoke] text='${text.replace("\n", "\\n")}'")
            toolCalls.forEachIndexed { i, c ->
                println("[toolcall-smoke]   [$i] name=${c.name} args=${c.argumentsJson}")
            }

            // Pass criterion: any of the following indicates the model emitted
            // (or attempted) a tool-call grammar, which is what we're verifying:
            //   - the chat template parser found a parseable tool_call,
            //   - the runtime saw `<|tool_call>` literally in the stream, or
            //   - the finish reason was tagged TOOL_CALL.
            val markerInText = "<|tool_call>" in text
            val sawToolCall = toolCalls.isNotEmpty() ||
                markerInText ||
                finish == FinishReason.TOOL_CALL

            assertTrue(
                sawToolCall,
                "Expected a tool_call marker or parsed tool call. Got finish=$finish, " +
                    "toolCalls=${toolCalls.size}, marker_in_text=$markerInText, text='$text'",
            )

            // If the model produced a structured tool call, sanity-check it
            // resolved to the registered tool. Don't assert on argument
            // contents — that's a separate concern from grammar.
            if (toolCalls.isNotEmpty()) {
                assertTrue(
                    toolCalls.any { it.name == "calculator" },
                    "Tool call name didn't resolve to 'calculator': ${toolCalls.map { it.name }}",
                )
            }
        } finally {
            model.close()
        }
    }

    private fun locateCheckpoint(): Path? {
        val raw = System.getenv("GEMMA4_E2B_SAFETENSORS_PATH")?.trim().orEmpty()
        if (raw.isEmpty()) {
            println("[skip] GEMMA4_E2B_SAFETENSORS_PATH not set.")
            return null
        }
        val p = Path.of(raw)
        if (!p.exists()) {
            println("[skip] Path does not exist: $p")
            return null
        }
        val resolved = when {
            p.isDirectory() -> {
                val idx = p.resolve("model.safetensors.index.json")
                val single = p.resolve("model.safetensors")
                when {
                    idx.exists() -> idx
                    single.exists() -> single
                    else -> {
                        println("[skip] No SafeTensors checkpoint found in $p")
                        return null
                    }
                }
            }
            else -> p
        }
        return resolved
    }
}
