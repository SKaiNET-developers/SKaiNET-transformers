package sk.ainet.models.laya

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

/**
 * Serialises a [JsonElement] the way CPython's `json.dumps(value, ensure_ascii=False)` does.
 *
 * Laya feeds this text straight into the tokenizer: a structured state and a structured
 * criterion both become JSON text before they become token ids (REQ-PRE-001, REQ-PRE-002), so
 * every character here changes the sequence. The rules reproduced are CPython's:
 *
 * - `", "` between members and `": "` after a key (the default separators), members in
 *   insertion order;
 * - strings escape only `"`, `\`, and the control characters below U+0020 (`\n`, `\r`, `\t`,
 *   `\b`, `\f` by letter, the rest as `\u00XX`); every other character, non-ASCII included,
 *   is written as itself;
 * - integers of any size are written as given; floats use `float.__repr__`: the shortest
 *   decimal string that round-trips, in fixed notation while the decimal exponent is in
 *   `[-4, 16)` and in scientific notation (`1e+16`, `1e-05`, exponent sign and at least two
 *   digits) outside it; whole floats keep `.0`; negative zero is `-0.0`;
 * - `NaN`, `Infinity` and `-Infinity` are written as CPython writes them by default.
 */
public object PythonJson {

    /** `json.dumps(value, ensure_ascii=False)`. */
    public fun dumps(value: JsonElement): String = buildString { write(value) }

    /** `repr(float)` for a double, see the class comment for the rules. */
    public fun floatRepr(value: Double): String {
        if (value.isNaN()) return "NaN"
        if (value == Double.POSITIVE_INFINITY) return "Infinity"
        if (value == Double.NEGATIVE_INFINITY) return "-Infinity"
        if (value == 0.0) return if (1.0 / value < 0) "-0.0" else "0.0"
        val negative = value < 0
        val (digits, exponent) = shortestDigits(kotlin.math.abs(value))
        val body = if (exponent in -4 until 16) fixed(digits, exponent) else scientific(digits, exponent)
        return if (negative) "-$body" else body
    }

    private fun StringBuilder.write(value: JsonElement) {
        when (value) {
            is JsonNull -> append("null")
            is JsonPrimitive -> if (value.isString) writeString(value.content) else writeScalar(value.content)
            is JsonArray -> {
                append('[')
                value.forEachIndexed { i, element ->
                    if (i > 0) append(", ")
                    write(element)
                }
                append(']')
            }
            is JsonObject -> {
                append('{')
                var first = true
                for ((key, element) in value) {
                    if (!first) append(", ")
                    first = false
                    writeString(key)
                    append(": ")
                    write(element)
                }
                append('}')
            }
        }
    }

    private fun StringBuilder.writeScalar(content: String) {
        when {
            content == "true" || content == "false" || content == "null" -> append(content)
            INTEGER.matches(content) -> append(content)
            else -> append(floatRepr(content.toDouble()))
        }
    }

    private fun StringBuilder.writeString(text: String) {
        append('"')
        for (ch in text) {
            when (ch) {
                '"' -> append("\\\"")
                '\\' -> append("\\\\")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                '\b' -> append("\\b")
                '\u000C' -> append("\\f")
                else -> if (ch < ' ') {
                    append("\\u")
                    append(ch.code.toString(16).padStart(4, '0'))
                } else {
                    append(ch)
                }
            }
        }
        append('"')
    }

    private val INTEGER = Regex("-?(0|[1-9][0-9]*)")

    /**
     * The shortest significant-digit string that parses back to [value] (positive, finite,
     * non-zero) and its decimal exponent: `value = d.ddd × 10^exponent`.
     *
     * Kotlin's `toString` yields a round-tripping decimal on every target; on the JVM it is the
     * shortest one except that it never has fewer than two significant digits (`4.9E-324` where
     * CPython writes `5e-324`). The loop below tries every shorter rounding and keeps the first
     * one that still parses to the same double, which is CPython's choice.
     */
    private fun shortestDigits(value: Double): Pair<String, Int> {
        val text = value.toString()
        val ePos = text.indexOfFirst { it == 'E' || it == 'e' }
        val mantissa = if (ePos >= 0) text.substring(0, ePos) else text
        val exp10 = if (ePos >= 0) text.substring(ePos + 1).toInt() else 0
        val dot = mantissa.indexOf('.')
        val intPart = if (dot >= 0) mantissa.substring(0, dot) else mantissa
        val fracPart = if (dot >= 0) mantissa.substring(dot + 1) else ""
        var raw = intPart + fracPart
        var exponent = exp10 + intPart.length - 1
        val leading = raw.indexOfFirst { it != '0' }
        raw = raw.substring(leading)
        exponent -= leading
        val digits = raw.trimEnd('0').ifEmpty { "0" }
        for (n in 1 until digits.length) {
            val (candidate, carried) = roundDigits(digits, n)
            val candidateExponent = exponent + carried
            if (parse(candidate, candidateExponent) == value) return candidate to candidateExponent
        }
        return digits to exponent
    }

    /** Round [digits] to [n] significant digits, half up; returns the digits and 1 if a carry added a digit. */
    private fun roundDigits(digits: String, n: Int): Pair<String, Int> {
        val kept = digits.substring(0, n).toCharArray()
        val roundUp = digits[n] >= '5'
        if (!roundUp) return kept.concatToString().trimEnd('0').ifEmpty { "0" } to 0
        var i = n - 1
        while (i >= 0) {
            if (kept[i] == '9') {
                kept[i] = '0'
                i--
            } else {
                kept[i] = kept[i] + 1
                break
            }
        }
        return if (i < 0) ("1" + kept.concatToString()).trimEnd('0') to 1
        else kept.concatToString().trimEnd('0') to 0
    }

    private fun parse(digits: String, exponent: Int): Double {
        val mantissa = if (digits.length > 1) digits[0] + "." + digits.substring(1) else digits
        return "${mantissa}e$exponent".toDouble()
    }

    private fun fixed(digits: String, exponent: Int): String {
        if (exponent < 0) return "0." + "0".repeat(-exponent - 1) + digits
        val intLen = exponent + 1
        val intPart = if (digits.length >= intLen) digits.substring(0, intLen) else digits + "0".repeat(intLen - digits.length)
        val fracPart = if (digits.length > intLen) digits.substring(intLen) else "0"
        return "$intPart.$fracPart"
    }

    private fun scientific(digits: String, exponent: Int): String {
        val mantissa = if (digits.length > 1) digits[0] + "." + digits.substring(1) else digits
        val sign = if (exponent < 0) "-" else "+"
        return mantissa + "e" + sign + kotlin.math.abs(exponent).toString().padStart(2, '0')
    }
}
