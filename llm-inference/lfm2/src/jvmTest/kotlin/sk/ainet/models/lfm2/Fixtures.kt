package sk.ainet.models.lfm2

import kotlinx.serialization.json.*
import kotlin.math.abs

data class ConvFixture(
    val schemaVersion: Int,
    val provenance: JsonObject,
    val config: ConvConfig,
    val tolerance: Tolerance,
    val weights: ConvWeights,
    val cases: List<ConvCase>,
)
data class ConvConfig(val hiddenSize: Int, val kernelSize: Int, val convBias: Boolean)
data class Tolerance(val atol: Float, val rtol: Float)
data class ConvWeights(val inProjection: List<List<Float>>, val conv: List<List<Float>>, val outProjection: List<List<Float>>)
data class ConvCase(val id: String, val chunks: List<ConvChunk>)
data class ConvChunk(
    val resetBefore: Boolean,
    val input: List<List<Float>>,
    val expectedOutput: List<List<Float>>,
    val expectedState: List<List<Float>>,
)

data class FixtureResult(val cases: Int, val chunks: Int, val comparedValues: Int, val maxAbsoluteError: Float, val provider: String)

fun readFixture(text: String): ConvFixture {
    val root = Json.parseToJsonElement(text).jsonObject
    val config = root.getValue("config").jsonObject
    val tolerance = root.getValue("tolerance").jsonObject
    val weights = root.getValue("weights").jsonObject
    fun matrix(value: JsonElement) = value.jsonArray.map { row -> row.jsonArray.map { it.jsonPrimitive.float } }
    return ConvFixture(root.getValue("schema_version").jsonPrimitive.int, root.getValue("provenance").jsonObject,
        ConvConfig(config.getValue("hidden_size").jsonPrimitive.int, config.getValue("kernel_size").jsonPrimitive.int, config.getValue("conv_bias").jsonPrimitive.boolean),
        Tolerance(tolerance.getValue("atol").jsonPrimitive.float, tolerance.getValue("rtol").jsonPrimitive.float),
        ConvWeights(matrix(weights.getValue("in_proj")), matrix(weights.getValue("conv")), matrix(weights.getValue("out_proj"))),
        root.getValue("cases").jsonArray.map { element ->
            val case = element.jsonObject
            ConvCase(case.getValue("id").jsonPrimitive.content, case.getValue("chunks").jsonArray.map {
                val chunk = it.jsonObject
                ConvChunk(chunk.getValue("reset_before").jsonPrimitive.boolean, matrix(chunk.getValue("input")), matrix(chunk.getValue("expected_output")), matrix(chunk.getValue("expected_state")))
            })
        })
}

fun evaluateFixture(fixture: ConvFixture): FixtureResult {
    require(fixture.schemaVersion == 1) { "Unsupported fixture schema" }
    require(!fixture.config.convBias) { "Biased convolution is unsupported" }
    require(fixture.cases.isNotEmpty()) { "No fixture cases" }
    require(fixture.tolerance.atol.isFinite() && fixture.tolerance.atol >= 0f && fixture.tolerance.rtol.isFinite() && fixture.tolerance.rtol >= 0f) { "Invalid tolerance" }
    var maximum = 0f
    var compared = 0
    var chunks = 0
    val model = ShortConv(fixture.config.hiddenSize, fixture.config.kernelSize, fixture.weights.inProjection, fixture.weights.conv, fixture.weights.outProjection)
    fun compare(actual: List<List<Float>>, expected: List<List<Float>>, label: String) {
        require(actual.size == expected.size && actual.indices.all { actual[it].size == expected[it].size }) { "$label shape mismatch" }
        for (row in actual.indices) for (col in actual[row].indices) {
            val a = actual[row][col]
            val e = expected[row][col]
            val error = abs(a - e)
            require(a.isFinite() && e.isFinite() && error <= fixture.tolerance.atol + fixture.tolerance.rtol * abs(e)) {
                "$label [$row,$col]: actual=$a expected=$e error=$error"
            }
            maximum = maxOf(maximum, error)
            compared++
        }
    }
    for (case in fixture.cases) {
        model.reset()
        require(case.chunks.isNotEmpty()) { "Empty case ${case.id}" }
        for ((index, chunk) in case.chunks.withIndex()) {
            if (chunk.resetBefore) model.reset()
            compare(model.forward(chunk.input), chunk.expectedOutput, "${case.id}/$index/output")
            compare(model.state(), chunk.expectedState, "${case.id}/$index/state")
            chunks++
        }
    }
    return FixtureResult(fixture.cases.size, chunks, compared, maximum, model.provider())
}

