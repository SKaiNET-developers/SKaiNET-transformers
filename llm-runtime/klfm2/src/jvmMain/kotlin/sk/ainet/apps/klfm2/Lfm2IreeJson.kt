package sk.ainet.apps.klfm2

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonPrimitive

/** Reject duplicate (including escaped-equivalent) keys before constructing a JSON object. */
internal fun parseIreeObject(bytes: ByteArray): JsonObject {
    require(bytes.size <= 4 * 1024 * 1024) { "JSON payload exceeds 4 MiB" }
    val text = Charsets.UTF_8.newDecoder().onMalformedInput(java.nio.charset.CodingErrorAction.REPORT)
        .onUnmappableCharacter(java.nio.charset.CodingErrorAction.REPORT).decode(java.nio.ByteBuffer.wrap(bytes)).toString()
    IreeJsonStructureScanner(text).validate()
    return Json.parseToJsonElement(text) as? JsonObject ?: throw IllegalArgumentException("Expected a JSON object")
}

private class IreeJsonStructureScanner(private val text: String) {
    private var index = 0
    fun validate() {
        value(0)
        whitespace()
        require(index == text.length) { "Trailing JSON content" }
    }
    private fun whitespace() {
        while (index < text.length && text[index] in " \t\r\n") index++
    }
    private fun character(): Char {
        whitespace()
        require(index < text.length) { "Incomplete JSON" }
        return text[index]
    }
    private fun expect(value: Char) {
        require(character() == value) { "Unexpected JSON punctuation" }
        index++
    }
    private fun value(depth: Int) {
        require(depth <= 64) { "JSON nesting exceeds 64 levels" }
        when (character()) {
            '{' -> {
                index++
                val keys = HashSet<String>()
                if (character() == '}') { index++; return }
                while (true) {
                    require(keys.add(string())) { "Duplicate JSON key" }
                    expect(':')
                    value(depth + 1)
                    when (character()) {
                        '}' -> { index++; return }
                        ',' -> index++
                        else -> throw IllegalArgumentException("Malformed JSON object")
                    }
                }
            }
            '[' -> {
                index++
                if (character() == ']') { index++; return }
                while (true) {
                    value(depth + 1)
                    when (character()) {
                        ']' -> { index++; return }
                        ',' -> index++
                        else -> throw IllegalArgumentException("Malformed JSON array")
                    }
                }
            }
            '"' -> string()
            else -> {
                val start = index
                while (index < text.length && text[index] !in " \t\r\n,]}:") index++
                require(index > start) { "Missing JSON value" }
                // The standard parser checks the scalar grammar after this structural scan.
            }
        }
    }
    private fun string(): String {
        require(character() == '"') { "JSON object keys must be strings" }
        val start = index++
        var escaped = false
        while (index < text.length) {
            val current = text[index++]
            if (escaped) { escaped = false; continue }
            if (current == '\\') { escaped = true; continue }
            if (current == '"') {
                val decoded = Json.parseToJsonElement(text.substring(start, index)).jsonPrimitive.content
                var pos = 0
                while (pos < decoded.length) {
                    val char = decoded[pos++]
                    if (Character.isHighSurrogate(char)) {
                        require(pos < decoded.length && Character.isLowSurrogate(decoded[pos++])) { "Unpaired JSON surrogate" }
                    } else require(!Character.isLowSurrogate(char)) { "Unpaired JSON surrogate" }
                }
                return decoded
            }
        }
        throw IllegalArgumentException("Unterminated JSON string")
    }
}
