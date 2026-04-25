package sk.ainet.apps.kllama.chat

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.add
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonObject

class ToolCallValidatorTest {

    private val calculator = ToolDefinition(
        name = "calculator",
        description = "Evaluate math expressions",
        parameters = buildJsonObject {
            put("type", "object")
            putJsonObject("properties") {
                putJsonObject("expression") { put("type", "string") }
                putJsonObject("precision") { put("type", "integer") }
            }
            put("required", buildJsonArray { add("expression") })
        }
    )

    private fun call(args: JsonObject) = ToolCall(id = "call_test", name = "calculator", arguments = args)

    @Test
    fun `valid arguments pass`() {
        val result = ToolCallValidator.validate(
            call(buildJsonObject {
                put("expression", "1+2")
                put("precision", 4)
            }),
            calculator
        )
        assertEquals(ToolCallValidationResult.Valid, result)
    }

    @Test
    fun `missing required field is rejected`() {
        val result = ToolCallValidator.validate(
            call(buildJsonObject { put("precision", 4) }),
            calculator
        )
        val invalid = assertIs<ToolCallValidationResult.Invalid>(result)
        assertTrue(invalid.reason.contains("missing required argument 'expression'"))
    }

    @Test
    fun `wrong type on string field is rejected`() {
        val result = ToolCallValidator.validate(
            call(buildJsonObject { put("expression", 42) }),
            calculator
        )
        val invalid = assertIs<ToolCallValidationResult.Invalid>(result)
        assertTrue(invalid.reason.contains("expected string"))
        assertTrue(invalid.reason.contains("'expression'"))
    }

    @Test
    fun `wrong type on integer field is rejected`() {
        val result = ToolCallValidator.validate(
            call(buildJsonObject {
                put("expression", "1+2")
                put("precision", "four")  // string where integer expected
            }),
            calculator
        )
        val invalid = assertIs<ToolCallValidationResult.Invalid>(result)
        assertTrue(invalid.reason.contains("expected integer"))
    }

    @Test
    fun `extra unknown field is allowed`() {
        val result = ToolCallValidator.validate(
            call(buildJsonObject {
                put("expression", "1+2")
                put("bogus", "ignored")
            }),
            calculator
        )
        assertEquals(ToolCallValidationResult.Valid, result)
    }

    @Test
    fun `empty schema accepts anything`() {
        val openTool = ToolDefinition(
            name = "open",
            description = "Open tool",
            parameters = JsonObject(emptyMap())
        )
        val result = ToolCallValidator.validate(
            ToolCall(id = "c", name = "open", arguments = buildJsonObject {
                put("arbitrary", "value")
                put("nested", buildJsonObject { put("k", 1) })
            }),
            openTool
        )
        assertEquals(ToolCallValidationResult.Valid, result)
    }

    @Test
    fun `boolean number array object types validate correctly`() {
        val tool = ToolDefinition(
            name = "multi",
            description = "x",
            parameters = buildJsonObject {
                putJsonObject("properties") {
                    putJsonObject("flag") { put("type", "boolean") }
                    putJsonObject("ratio") { put("type", "number") }
                    putJsonObject("items") { put("type", "array") }
                    putJsonObject("cfg") { put("type", "object") }
                }
            }
        )

        val good = ToolCallValidator.validate(
            ToolCall(id = "c", name = "multi", arguments = buildJsonObject {
                put("flag", true)
                put("ratio", 1.5)
                put("items", buildJsonArray { add("a") })
                put("cfg", buildJsonObject { put("k", "v") })
            }),
            tool
        )
        assertEquals(ToolCallValidationResult.Valid, good)

        val bad = ToolCallValidator.validate(
            ToolCall(id = "c", name = "multi", arguments = buildJsonObject {
                put("items", "not-an-array")
            }),
            tool
        )
        assertIs<ToolCallValidationResult.Invalid>(bad)
    }
}
