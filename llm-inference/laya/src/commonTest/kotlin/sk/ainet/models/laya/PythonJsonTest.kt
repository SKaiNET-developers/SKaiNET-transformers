package sk.ainet.models.laya

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlin.test.Test
import kotlin.test.assertEquals

class PythonJsonTest {

    @Test
    fun floats_spell_like_cpython_repr() {
        val cases = mapOf(
            1e16 to "1e+16", 1e-5 to "1e-05", 1e-4 to "0.0001", 1e23 to "1e+23", 2.0 to "2.0", -0.0 to "-0.0",
            1e308 to "1e+308", Double.MIN_VALUE to "5e-324", 1.0 / 3.0 to "0.3333333333333333", 1.5 to "1.5",
            100.0 to "100.0", 1e15 to "1000000000000000.0", 0.1 + 0.2 to "0.30000000000000004", 123456.789 to "123456.789",
            -2.5 to "-2.5", 9999999999999998.0 to "9999999999999998.0", 1.7976931348623157e308 to "1.7976931348623157e+308",
            0.001 to "0.001", 0.00012345 to "0.00012345", 1.25e-7 to "1.25e-07",
        )
        for ((value, expected) in cases) assertEquals(expected, PythonJson.floatRepr(value), "repr($value)")
        assertEquals("NaN", PythonJson.floatRepr(Double.NaN))
        assertEquals("-Infinity", PythonJson.floatRepr(Double.NEGATIVE_INFINITY))
    }

    @Test
    fun objects_use_default_separators_and_insertion_order() {
        val state = buildJsonObject {
            put("order", 1)
            put("items", buildJsonArray { add(JsonPrimitive("a")); add(JsonPrimitive("b")) })
            put("n", 2.5)
            put("ok", true)
            put("missing", JsonPrimitive(null as String?))
        }
        assertEquals("""{"order": 1, "items": ["a", "b"], "n": 2.5, "ok": true, "missing": null}""", PythonJson.dumps(state))
    }

    @Test
    fun parsed_numbers_are_renormalised_like_python() {
        val text = """{"big": 1e+16, "small": 1e-05, "edge": 0.0001, "whole": 2.0, "negzero": -0.0, "denormal": 5e-324, "bigint": 123456789012345678901234567890, "negint": -7, "sci": 1E2}"""
        val element = Json.parseToJsonElement(text)
        assertEquals(
            """{"big": 1e+16, "small": 1e-05, "edge": 0.0001, "whole": 2.0, "negzero": -0.0, "denormal": 5e-324, "bigint": 123456789012345678901234567890, "negint": -7, "sci": 100.0}""",
            PythonJson.dumps(element),
        )
    }

    @Test
    fun strings_escape_only_what_cpython_escapes() {
        val s = JsonPrimitive("quote\" back\\ nl\n tab\t bell\u0007 del\u007f ümlaut 日本 😀")
        assertEquals("\"quote\\\" back\\\\ nl\\n tab\\t bell\\u0007 del\u007f ümlaut 日本 😀\"", PythonJson.dumps(s))
    }
}
