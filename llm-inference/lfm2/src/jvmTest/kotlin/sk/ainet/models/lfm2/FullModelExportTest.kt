package sk.ainet.models.lfm2

import sk.ainet.models.lfm2.export.*

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import kotlin.test.*
import kotlinx.serialization.json.*

class FullModelExportTest {
    private val ctx = DirectCpuExecutionContext.create()
    private val dimensions = ModelDimensions(8, 12, 4, 2, 2, 4, 3, 1e-5f, 1e6, listOf("conv", "full_attention"))
    private fun weights(): Map<String, Tensor<FP32, Float>> = HybridLfm2Model.expectedWeights(dimensions).mapValues { (name, shape) ->
        val count = shape.dimensions.reduce(Int::times)
        val values = if (name.endsWith("norm.weight")) FloatArray(count) { 0.7f + it % 3 * 0.1f }
            else FloatArray(count) { (((it * 11 + name.hashCode().mod(13)) % 23) - 11) * 0.025f }
        ctx.fromFloatArray(shape, FP32::class, values)
    }

    private fun close(a: FloatArray, b: FloatArray) {
        assertEquals(a.size, b.size)
        for (i in a.indices) assertTrue(abs(a[i] - b[i]) < 2e-6f, "Value $i differs: ${a[i]} vs ${b[i]}")
    }

    @Test
    fun purePrefillNetworkPreservesCompleteEagerLogitsAndAllStatePorts() {
        val w = weights(); val d = dimensions; val ids = intArrayOf(1, 2, 3); val length = ids.size
        val embedding = w.getValue("model.embed_tokens.weight")
        val values = FloatArray(length * d.width) { embedding.data[ids[it / d.width], it % d.width] }
        val cosine = FloatArray(length * d.headDim); val sine = FloatArray(cosine.size)
        for (t in 0 until length) for (i in 0 until d.headDim / 2) for (lane in intArrayOf(i, i + d.headDim / 2)) {
            val angle = t * d.theta.pow(-2.0 * i / d.headDim)
            cosine[t * d.headDim + lane] = cos(angle).toFloat(); sine[t * d.headDim + lane] = sin(angle).toFloat()
        }
        fun tensor(shape: Shape, values: FloatArray) = ctx.fromFloatArray<FP32, Float>(shape, FP32::class, values)
        val mask = FloatArray(length * length) { if (it % length <= it / length) 0f else -Float.MAX_VALUE }
        val result = Lfm2PrefillNetwork(d, w).forward(tensor(Shape(length, d.width), values), tensor(Shape(length, d.headDim), cosine),
            tensor(Shape(length, d.headDim), sine), tensor(Shape(length, length), mask), mapOf(0 to tensor(Shape(d.width, d.kernel), FloatArray(d.width * d.kernel))), ctx)
        val state = HybridModelState()
        val expected = HybridLfm2Model(d, ctx, w, RmsReductionPolicy.BALANCED_FP32).forward(ids, state)
        close(result.logits.data.copyToFloatArray(), expected.logits.data.copyToFloatArray())
        assertEquals(listOf("history_0", "key_1", "value_1"), result.states.keys.toList())
        close(result.states.getValue("history_0").data.copyToFloatArray(), state.convolution.getValue(0).data.copyToFloatArray())
        close(result.states.getValue("key_1").data.copyToFloatArray(), state.keys.getValue(1).data.copyToFloatArray())
        close(result.states.getValue("value_1").data.copyToFloatArray(), state.values.getValue(1).data.copyToFloatArray())
    }

    @Test
    fun completeGraphKeepsRuntimeArgumentsAndOneExternalCopyPerWeight() {
        val w = weights()
        val trace = traceLfm2Prefill(Lfm2PrefillNetwork(dimensions, w), 3)
        assertEquals(listOf("embeddings", "cos", "sin", "mask", "history_0"), trace.inputNames)
        assertEquals(listOf("logits", "history_0", "key_1", "value_1"), trace.outputNames)
        assertEquals(trace.inputNames.size, trace.module.inputSpecs.size)
        assertEquals(trace.outputNames.size, trace.module.outputSpecs.size)
        assertEquals(w.size, trace.module.externalParameters.size)
        assertTrue(trace.module.content.contains("stablehlo.convolution"))
        assertTrue(trace.module.content.contains("across dimensions = [2]"))
        assertFalse(trace.module.content.contains("custom_call"))
        assertTrue(trace.operationCount > 100)
    }

    @Test
    fun streamedParameterPayloadsMatchWeightsAndFailedIntegrityPreservesPriorFile() {
        val w = weights()
        val trace = traceLfm2Prefill(Lfm2PrefillNetwork(dimensions, w), 2)
        val expected = w.values.map { sha256(floatBytes(it.data.copyToFloatArray())) }
        val directory = kotlin.io.path.createTempDirectory("full-parameter-test").toFile()
        try {
            val file = File(directory, "model.safetensors")
            val record = writeLfm2ParameterArchive(trace.module, file, expected)
            assertEquals(expected.sorted(), record.getValue("entries").jsonArray.map { it.jsonObject.getValue("sha256").jsonPrimitive.content }.sorted())
            val bytes = file.readBytes()
            val size = ByteBuffer.wrap(bytes, 0, 8).order(ByteOrder.LITTLE_ENDIAN).long
            assertEquals(0, size % 8)
            val header = Json.parseToJsonElement(bytes.copyOfRange(8, (8 + size).toInt()).toString(Charsets.UTF_8)).jsonObject
            assertEquals(w.size, header.size)
            val original = bytes.copyOf()
            assertFailsWith<IllegalArgumentException> { writeLfm2ParameterArchive(trace.module, file, expected.map { "wrong-$it" }) }
            assertContentEquals(original, file.readBytes())
            assertFalse(File(directory, "model.safetensors.partial").exists())
        } finally { directory.deleteRecursively() }
    }

    @Test
    fun exportBundlePinsPortableSourcesAbiAndPayloadAndCannotOverwrite() {
        val w = weights()
        val trace = traceLfm2Prefill(Lfm2PrefillNetwork(dimensions, w), 2)
        val hashes = w.values.map { sha256(floatBytes(it.data.copyToFloatArray())) }
        val parent = kotlin.io.path.createTempDirectory("lfm2-bundle-test").toFile()
        try {
            val output = File(parent, "bundle")
            val report = writeLfm2ExportBundle(trace, output, hashes, buildJsonObject { put("kind", "synthetic") })
            assertEquals("exported_not_yet_compiled", report.getValue("status").jsonPrimitive.content)
            assertEquals(trace.inputNames, report.getValue("input_order").jsonArray.map { it.jsonPrimitive.content })
            val sources = report.getValue("implementation_sources").jsonObject.getValue("files").jsonArray
            assertTrue(sources.any { it.jsonObject.getValue("file").jsonPrimitive.content.endsWith("Lfm2PrefillNetwork.kt") })
            assertTrue(sources.all { !File(it.jsonObject.getValue("file").jsonPrimitive.content).isAbsolute })
            val archive = File(output, "model.safetensors")
            assertEquals(sha256(archive), report.getValue("parameters").jsonObject.getValue("sha256").jsonPrimitive.content)
            assertFails { writeLfm2ExportBundle(trace, output, hashes, buildJsonObject {}) }
            val invalid = File(parent, "invalid.safetensors")
            assertFailsWith<IllegalArgumentException> { writeLfm2ParameterArchive(trace.module, invalid, hashes.map { "0".repeat(64) }) }
            assertFalse(invalid.exists())
            assertTrue(parent.listFiles()!!.none { it.name.endsWith(".partial") })
        } finally { parent.deleteRecursively() }
    }

    private fun floatBytes(values: FloatArray): ByteArray = ByteBuffer.allocate(values.size * 4)
        .order(ByteOrder.LITTLE_ENDIAN).apply { values.forEach { putFloat(it) } }.array()

}
