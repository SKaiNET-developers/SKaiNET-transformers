package sk.ainet.llm.api

/**
 * Description of a tool the model may call.
 *
 * The [parametersJsonSchema] is a JSON Schema document (as a raw string) describing
 * the tool's argument shape — kept as a plain string here so the neutral SPI does not
 * pull in any JSON library. Adapters are free to construct it from `kotlinx.serialization`,
 * Jackson, or any other source.
 */
public data class ToolDefinition(
    public val name: String,
    public val description: String,
    public val parametersJsonSchema: String,
)

/** A tool call emitted by the model. */
public data class ToolCall(
    public val id: String,
    public val name: String,
    /** Raw JSON-encoded argument object as produced by the model. */
    public val argumentsJson: String,
)
