package sk.ainet.apps.kllama.chat

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.longOrNull

/**
 * Outcome of validating a [ToolCall] against a [ToolDefinition]'s JSON Schema.
 */
public sealed class ToolCallValidationResult {
    public object Valid : ToolCallValidationResult()
    public data class Invalid(val reason: String) : ToolCallValidationResult()
}

/**
 * Lightweight JSON-Schema-aware validator for parsed tool calls.
 *
 * Checks that:
 * 1. Every name listed in `parameters.required` is present in `ToolCall.arguments`.
 * 2. For every argument whose schema declares a `type`, the supplied value matches
 *    one of: `string`, `integer`, `number`, `boolean`, `array`, `object`, `null`.
 *
 * Extra arguments not declared in `parameters.properties` are accepted (matching the
 * JSON Schema default of `additionalProperties: true`). Nested object/array contents
 * are not recursively validated — that would require a full schema library; the goal
 * here is to catch the common model failure modes (missing required field, wrong
 * primitive type) with a minimal multiplatform-friendly implementation.
 */
public object ToolCallValidator {

    public fun validate(call: ToolCall, definition: ToolDefinition): ToolCallValidationResult {
        val schema = definition.parameters
        val properties = (schema["properties"] as? JsonObject) ?: JsonObject(emptyMap())
        val required: List<String> = (schema["required"] as? JsonArray)
            ?.mapNotNull { (it as? JsonPrimitive)?.contentOrNull }
            ?: emptyList()

        for (field in required) {
            if (!call.arguments.containsKey(field)) {
                return ToolCallValidationResult.Invalid(
                    "missing required argument '$field' for tool '${definition.name}'"
                )
            }
        }

        for ((name, value) in call.arguments) {
            val propSchema = properties[name] as? JsonObject ?: continue
            val expectedType = (propSchema["type"] as? JsonPrimitive)?.contentOrNull ?: continue
            if (!typeMatches(expectedType, value)) {
                return ToolCallValidationResult.Invalid(
                    "argument '$name' for tool '${definition.name}' expected $expectedType but got ${describeType(value)}"
                )
            }
        }

        return ToolCallValidationResult.Valid
    }

    private fun typeMatches(expected: String, value: JsonElement): Boolean = when (expected.lowercase()) {
        "string" -> value is JsonPrimitive && value.isString
        "integer" -> value is JsonPrimitive && !value.isString && value.longOrNull != null
        "number" -> value is JsonPrimitive && !value.isString && value.doubleOrNull != null
        "boolean" -> value is JsonPrimitive && !value.isString && value.booleanOrNull != null
        "array" -> value is JsonArray
        "object" -> value is JsonObject
        "null" -> value is JsonNull
        else -> true
    }

    private fun describeType(value: JsonElement): String = when (value) {
        is JsonNull -> "null"
        is JsonArray -> "array"
        is JsonObject -> "object"
        is JsonPrimitive -> when {
            value.isString -> "string"
            value.booleanOrNull != null -> "boolean"
            value.longOrNull != null -> "integer"
            value.doubleOrNull != null -> "number"
            else -> "primitive"
        }
    }
}
