package sk.ainet.models.lfm2

import kotlinx.serialization.json.*
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.security.MessageDigest
import kotlin.test.*

class Lfm2WeightsTest {
    private val config = Lfm2Config.parse(tinyConfigJson().toString())

    private fun checkpoint(directory: File, dtype: String = "BF16", nonFinite: Boolean = false,
        changeHeader: (MutableMap<String, JsonElement>) -> Unit = {},
    ): File {
        val bytesPerValue = if (dtype == "F32") 4 else 2
        val expected = HybridLfm2Model.expectedWeights(config.toDimensions())
        val payload = ByteBuffer.allocate(expected.values.sumOf { it.dimensions.reduce(Int::times) * bytesPerValue }).order(ByteOrder.LITTLE_ENDIAN)
        val header = linkedMapOf<String, JsonElement>()
        for ((name, shape) in expected) {
            val start = payload.position()
            repeat(shape.dimensions.reduce(Int::times)) {
                when (dtype) {
                    "F32" -> payload.putFloat(if (nonFinite) Float.NaN else 0.25f)
                    "F16" -> payload.putShort(if (nonFinite) 0x7e00.toShort() else 0x3400.toShort())
                    else -> payload.putShort(if (nonFinite) 0x7fc0.toShort() else 0x3e80.toShort())
                }
            }
            header[name] = buildJsonObject {
                put("dtype", dtype); put("shape", JsonArray(shape.dimensions.map(::JsonPrimitive)))
                put("data_offsets", buildJsonArray { add(start); add(payload.position()) })
            }
        }
        changeHeader(header)
        val text = JsonObject(header).toString().toByteArray()
        val padded = text.size + (8 - text.size % 8) % 8
        return File(directory, "model.safetensors").apply {
            outputStream().use { stream ->
                stream.write(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(padded.toLong()).array())
                stream.write(text); repeat(padded - text.size) { stream.write(32) }; stream.write(payload.array())
            }
        }
    }

    private inline fun temporary(test: (File) -> Unit) {
        val directory = Files.createTempDirectory("lfm2-loader-test-").toFile()
        try { test(directory) } finally { directory.deleteRecursively() }
    }

    @Test
    fun loadsSupportedFloatingWeightsWithExactConvertedHashesAndUsableModel() = temporary { directory ->
        for (dtype in listOf("BF16", "F16", "F32")) {
            val file = checkpoint(directory, dtype)
            val loaded = loadLfm2Model(config, file)
            assertEquals(HybridLfm2Model.expectedWeights(config.toDimensions()).keys, loaded.weights.keys)
            assertEquals(21, loaded.records.size)
            assertEquals(RmsReductionPolicy.BALANCED_FP32, loaded.model.rmsReduction)
            for (record in loaded.records) {
                assertEquals(dtype, record.sourceDtype)
                val count = record.shape.reduce(Int::times)
                val bytes = ByteBuffer.allocate(count * 4).order(ByteOrder.LITTLE_ENDIAN)
                repeat(count) { bytes.putFloat(0.25f) }
                val hash = MessageDigest.getInstance("SHA-256").digest(bytes.array()).joinToString("") { "%02x".format(it) }
                assertEquals(hash, record.fp32Sha256)
                assertEquals(count * 4L, record.fp32Bytes)
            }
            val state = HybridModelState()
            val output = loaded.model.forward(intArrayOf(1, 2), state)
            assertEquals(2, state.position)
            assertTrue(output.logits.data.copyToFloatArray().all { it.isFinite() })
            assertFalse("lm_head.weight" in loaded.weights)
        }
    }

    @Test
    fun checkpointIdentityIsOptionalButEnforcedWhenSupplied() = temporary { directory ->
        val file = checkpoint(directory)
        val sha = MessageDigest.getInstance("SHA-256").digest(file.readBytes()).joinToString("") { "%02x".format(it) }
        assertEquals(sha, loadLfm2Model(config, file, expectedCheckpointSha256 = sha.uppercase()).checkpointSha256)
        assertFailsWith<IllegalArgumentException> { loadLfm2Model(config, file, expectedCheckpointSha256 = "0".repeat(64)) }
        assertFailsWith<IllegalArgumentException> { loadLfm2Model(config, file, expectedCheckpointSha256 = "invalid") }
    }

    @Test
    fun rejectsMissingExtraOrIncorrectShapeWeights() = temporary { directory ->
        assertFailsWith<IllegalArgumentException> { loadLfm2Model(config, checkpoint(directory) { it.remove("model.embedding_norm.weight") }) }
        assertFailsWith<IllegalArgumentException> { loadLfm2Model(config, checkpoint(directory) { it["lm_head.weight"] = it.getValue("model.embed_tokens.weight") }) }
        assertFailsWith<IllegalArgumentException> {
            loadLfm2Model(config, checkpoint(directory) { header ->
                val name = "model.embedding_norm.weight"
                header[name] = JsonObject(header.getValue(name).jsonObject + ("shape" to buildJsonArray { add(2); add(2) }))
            })
        }
    }

    @Test
    fun rejectsUnsupportedDtypeOverlappingExtentsAndNonFiniteWeights() = temporary { directory ->
        assertFailsWith<IllegalArgumentException> { loadLfm2Model(config, checkpoint(directory, "I16")) }
        assertFailsWith<IllegalArgumentException> {
            loadLfm2Model(config, checkpoint(directory) { header ->
                val name = "model.embedding_norm.weight"
                header[name] = JsonObject(header.getValue(name).jsonObject + ("data_offsets" to buildJsonArray { add(0); add(8) }))
            })
        }
        assertFailsWith<IllegalArgumentException> { loadLfm2Model(config, checkpoint(directory, nonFinite = true)) }
        val truncated = checkpoint(directory)
        truncated.writeBytes(truncated.readBytes().dropLast(2).toByteArray())
        assertFailsWith<IllegalArgumentException> { loadLfm2Model(config, truncated) }
    }
}
