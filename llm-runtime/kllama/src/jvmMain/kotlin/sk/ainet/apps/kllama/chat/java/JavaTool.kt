package sk.ainet.apps.kllama.chat.java

import kotlinx.serialization.json.*
import sk.ainet.apps.kllama.chat.Tool
import sk.ainet.apps.kllama.chat.ToolDefinition

/**
 * Java-friendly factories for the kllama tool-calling surface.
 *
 * Avoids forcing Java consumers to import `kotlinx.serialization` types directly
 * when constructing a [ToolDefinition].
 *
 * Example usage from Java:
 * ```java
 * ToolDefinition calc = JavaTools.definition(
 *     "calculator",
 *     "Evaluate a simple arithmetic expression",
 *     "{\"type\":\"object\",\"properties\":{\"expression\":{\"type\":\"string\"}},\"required\":[\"expression\"]}"
 * );
 * ```
 */
public object JavaTools {

    /**
     * Construct a [ToolDefinition] from a JSON Schema string.
     *
     * @param name Unique tool name (e.g. "calculator").
     * @param description Human-readable description.
     * @param parametersJsonSchema JSON Schema document describing the tool's argument shape.
     */
    @JvmStatic
    public fun definition(
        name: String,
        description: String,
        parametersJsonSchema: String,
    ): ToolDefinition {
        val parameters = Json.parseToJsonElement(parametersJsonSchema) as JsonObject
        return ToolDefinition(name, description, parameters)
    }
}

/**
 * Java-friendly tool interface that uses `Map<String, Object>` instead of
 * kotlinx.serialization.json.JsonObject.
 *
 * Example usage from Java:
 * ```java
 * JavaTool calculator = new JavaTool() {
 *     public ToolDefinition getDefinition() { ... }
 *     public String execute(Map<String, Object> arguments) {
 *         String expression = (String) arguments.get("expression");
 *         return String.valueOf(eval(expression));
 *     }
 * };
 * ```
 */
public interface JavaTool {

    /** The tool's definition (name, description, parameters schema). */
    public val definition: ToolDefinition

    /**
     * Execute the tool with the given arguments.
     *
     * @param arguments Parsed arguments as a Java Map (String keys, Object values).
     *        Values may be String, Number, Boolean, List, or nested Map.
     * @return A string result to feed back to the model.
     */
    public fun execute(arguments: Map<String, Any?>): String
}

/**
 * Adapter that wraps a [JavaTool] into the internal [Tool] interface,
 * converting between JsonObject and Map<String, Object>.
 */
internal class JavaToolAdapter(private val javaTool: JavaTool) : Tool {

    override val definition: ToolDefinition
        get() = javaTool.definition

    override fun execute(arguments: JsonObject): String {
        val map = jsonObjectToMap(arguments)
        return javaTool.execute(map)
    }

    private fun jsonObjectToMap(obj: JsonObject): Map<String, Any?> {
        val result = mutableMapOf<String, Any?>()
        for ((key, value) in obj) {
            result[key] = jsonElementToJava(value)
        }
        return result
    }

    private fun jsonElementToJava(element: JsonElement): Any? = when (element) {
        is JsonNull -> null
        is JsonPrimitive -> when {
            element.isString -> element.content
            element.content == "true" -> true
            element.content == "false" -> false
            element.content.contains('.') -> element.content.toDoubleOrNull() ?: element.content
            else -> element.content.toLongOrNull() ?: element.content
        }
        is JsonArray -> element.map { jsonElementToJava(it) }
        is JsonObject -> jsonObjectToMap(element)
    }
}
