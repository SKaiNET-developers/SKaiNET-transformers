package sk.ainet.apps.kllama.chat

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.fail
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.add
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.putJsonObject

/**
 * Byte-for-byte parity check between our Kotlin [Gemma4ChatTemplate]
 * and HuggingFace's official Jinja `chat_template.jinja` shipped with
 * `google/gemma-4-e2b-it`.
 *
 * Reference renderings are produced by `/tmp/render_hf_chat_template.py`
 * (run separately via `uv run`) — that script reads `TEST_CASE` and
 * writes to stdout. The bytes for each case are stored under
 * `/tmp/hf_rendered_<case>.txt`. Our render is produced from the same
 * logical input and written to `/tmp/dsl_rendered_<case>.txt` for
 * off-test inspection.
 *
 * Tests are gated on the reference files existing — when missing, they
 * skip (CI without `uv` doesn't fail). To regenerate references:
 *
 * ```
 * for case in calculator two_tools no_tools multi_arg; do
 *     TEST_CASE=$case uv run --no-project --with 'transformers' --with 'torch' \
 *         --with 'protobuf' --with 'sentencepiece' --with 'jinja2' \
 *         python /tmp/render_hf_chat_template.py > /tmp/hf_rendered_$case.txt
 * done
 * ```
 *
 * Then:
 *
 * ```
 * ./gradlew :llm-agent:jvmTest --tests \
 *     "sk.ainet.apps.kllama.chat.Gemma4ChatTemplateHfParityTest"
 * ```
 */
class Gemma4ChatTemplateHfParityTest {

    private val baseMessages = listOf(
        ChatMessage(ChatRole.SYSTEM, "You are a helpful assistant with access to tools."),
        ChatMessage(ChatRole.USER, "What is 17 * 23? Use the calculator tool.")
    )

    @Test
    fun `case calculator — single string-arg tool`() = runParityCase("calculator") {
        listOf(
            ToolDefinition(
                name = "calculator",
                description = "Evaluate a mathematical expression. Supports +, -, *, / and parentheses.",
                parameters = buildJsonObject {
                    put("type", "object")
                    putJsonObject("properties") {
                        putJsonObject("expression") {
                            put("type", "string")
                            put("description", "The mathematical expression to evaluate, e.g. '17 * 23'")
                        }
                    }
                    putJsonArray("required") { add(JsonPrimitive("expression")) }
                }
            )
        )
    }

    @Test
    fun `case two_tools — multiple per-tool blocks`() = runParityCase("two_tools") {
        listOf(
            ToolDefinition(
                name = "calculator",
                description = "Evaluate math.",
                parameters = buildJsonObject {
                    put("type", "object")
                    putJsonObject("properties") {
                        putJsonObject("expression") { put("type", "string") }
                    }
                    putJsonArray("required") { add(JsonPrimitive("expression")) }
                }
            ),
            ToolDefinition(
                name = "lookup",
                description = "Look up a value.",
                parameters = buildJsonObject {
                    put("type", "object")
                    putJsonObject("properties") {
                        putJsonObject("key") {
                            put("type", "string")
                            put("description", "Key to look up")
                        }
                    }
                }
            )
        )
    }

    @Test
    fun `case no_tools — system message only, no tool block`() =
        runParityCase("no_tools") { emptyList() }

    @Test
    fun `case multi_arg — string + number + boolean params`() = runParityCase("multi_arg") {
        listOf(
            ToolDefinition(
                name = "search",
                description = "Search the web.",
                parameters = buildJsonObject {
                    put("type", "object")
                    putJsonObject("properties") {
                        putJsonObject("query") {
                            put("type", "string")
                            put("description", "Search terms")
                        }
                        putJsonObject("limit") {
                            put("type", "number")
                            put("description", "Max results")
                        }
                        putJsonObject("include_archived") {
                            put("type", "boolean")
                        }
                    }
                    putJsonArray("required") { add(JsonPrimitive("query")) }
                }
            )
        )
    }

    private fun runParityCase(name: String, toolsBuilder: () -> List<ToolDefinition>) {
        val refPath = Path.of("/tmp/hf_rendered_$name.txt")
        if (!Files.exists(refPath)) {
            println("[skip] $refPath missing — run /tmp/render_hf_chat_template.py with TEST_CASE=$name first.")
            return
        }
        val expected = Files.readString(refPath)
        val tools = toolsBuilder()
        val actual = Gemma4ChatTemplate().apply(baseMessages, tools, addGenerationPrompt = true)

        Files.writeString(Path.of("/tmp/dsl_rendered_$name.txt"), actual)

        if (actual == expected) return

        val firstDiff = firstDifferenceIndex(actual, expected)
        val ctxStart = (firstDiff - 50).coerceAtLeast(0)
        val ctxEnd = (firstDiff + 50)
        val expSlice = expected.substring(ctxStart.coerceAtMost(expected.length), ctxEnd.coerceAtMost(expected.length))
        val actSlice = actual.substring(ctxStart.coerceAtMost(actual.length), ctxEnd.coerceAtMost(actual.length))
        fail(
            buildString {
                appendLine("[$name] Kotlin render diverges from HF reference at byte $firstDiff.")
                appendLine("  expected (HF Jinja): ${escape(expSlice)}")
                appendLine("  actual   (Kotlin) :  ${escape(actSlice)}")
                appendLine()
                appendLine("Lengths: expected=${expected.length}, actual=${actual.length}")
                appendLine("Inspect: diff /tmp/hf_rendered_$name.txt /tmp/dsl_rendered_$name.txt")
            }
        )
    }

    private fun firstDifferenceIndex(a: String, b: String): Int {
        val n = minOf(a.length, b.length)
        for (i in 0 until n) if (a[i] != b[i]) return i
        return n
    }

    private fun escape(s: String): String =
        s.replace("\n", "\\n").replace("\t", "\\t")
}
