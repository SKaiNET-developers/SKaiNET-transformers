package sk.ainet.models.llama

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.RandomAccessSource
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Bf16TensorData
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.Fp16DenseTensorData
import sk.ainet.lang.tensor.data.NarrowFloatTensorData
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DTypePolicy
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Fp16Codec
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Pins the narrow-float KEEP_NATIVE behaviour of [DecoderSafeTensorsLoader] — the
 * transformer-repo counterpart of the engine's `SafeTensorsParametersLoaderFp16PolicyTest`.
 *
 * The BF16 arm has existed since 0.25.0; the F16 arm landed with engine 0.38.0's
 * `Fp16DenseTensorData`. What matters here is that the loader resolves the two formats
 * **independently**: both are 2 bytes per element, so routing F16 bytes through the BF16 decode
 * would not throw — it would quietly produce wrong numbers. These tests assert the packed bytes
 * survive verbatim, decode bit-identically to the widening path, and that neither policy leaks
 * into the other format.
 *
 * Files are synthesized in-test; no model downloads are involved.
 */
class DecoderSafeTensorsLoaderNarrowFloatTest {

    /** A canonical-mappable HF weight name, so [HfTensorNameMapper] doesn't skip the tensor. */
    private val hfName = "model.layers.0.self_attn.q_proj.weight"
    private val canonical = LlamaTensorNames.attnQ(0)

    private val metadata = LlamaModelMetadata(
        architecture = "llama",
        embeddingLength = 4,
        contextLength = 8,
        blockCount = 1,
        headCount = 1,
        kvHeadCount = 1,
        feedForwardLength = 4,
        ropeDimensionCount = 4,
        vocabSize = 4,
    )

    private fun fp32ToFp16Bytes(values: FloatArray): ByteArray {
        val out = ByteArray(values.size * 2)
        for (i in values.indices) {
            val bits = Fp16Codec.encode(values[i])
            out[i * 2] = (bits and 0xFF).toByte()
            out[i * 2 + 1] = ((bits ushr 8) and 0xFF).toByte()
        }
        return out
    }

    private fun fp32ToBf16Bytes(values: FloatArray): ByteArray {
        val out = ByteArray(values.size * 2)
        for (i in values.indices) {
            val bits = (values[i].toRawBits() ushr 16) and 0xFFFF
            out[i * 2] = (bits and 0xFF).toByte()
            out[i * 2 + 1] = ((bits ushr 8) and 0xFF).toByte()
        }
        return out
    }

    /** Write a single-tensor SafeTensors file: 8-byte LE header length, JSON header, then data. */
    private fun writeSafeTensors(entries: List<Triple<String, String, ByteArray>>, rows: Int, cols: Int): File {
        val header = StringBuilder("{")
        var offset = 0L
        entries.forEachIndexed { i, (name, dtype, bytes) ->
            if (i > 0) header.append(",")
            header.append(
                "\"$name\": {\"dtype\": \"$dtype\", \"shape\": [$rows, $cols], " +
                    "\"data_offsets\": [$offset, ${offset + bytes.size}]}",
            )
            offset += bytes.size
        }
        header.append("}")
        val headerBytes = header.toString().toByteArray(Charsets.UTF_8)

        val file = Files.createTempFile("decoder_st_narrow", ".safetensors").toFile()
        file.deleteOnExit()
        file.outputStream().use { out ->
            out.write(
                ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
                    .putLong(headerBytes.size.toLong()).array(),
            )
            out.write(headerBytes)
            entries.forEach { (_, _, bytes) -> out.write(bytes) }
        }
        return file
    }

    private fun load(file: File, policy: DTypePolicy): Map<String, Tensor<FP32, Float>> {
        val ctx = DirectCpuExecutionContext()
        val loader = DecoderSafeTensorsLoader(
            ctx = ctx,
            dtype = FP32::class,
            metadata = metadata,
            tiedEmbeddings = false,
            dtypePolicy = policy,
        )
        val provider: () -> RandomAccessSource = { JvmRandomAccessSource.open(file) }
        return loader.loadToMap(provider).tensors
    }

    /** 2x4, all exactly representable in binary16 AND bfloat16 so decode comparisons are exact. */
    private val values = floatArrayOf(0.0f, 1.0f, -1.0f, 0.5f, 2.0f, -4.0f, 0.25f, 8.0f)

    @Test
    fun `default policy widens F16 to an FP32 float array`() {
        val file = writeSafeTensors(listOf(Triple(hfName, "F16", fp32ToFp16Bytes(values))), 2, 4)

        val weight = load(file, DTypePolicy.Any)[canonical] ?: error("missing $canonical")
        assertTrue(
            weight.data is FloatArrayTensorData<*>,
            "default policy must widen, got ${weight.data::class.simpleName}",
        )
        assertContentEquals(values, weight.data.copyToFloatArray())
    }

    @Test
    fun `Require(FP16) keeps the on-disk F16 bytes verbatim`() {
        val onDisk = fp32ToFp16Bytes(values)
        val file = writeSafeTensors(listOf(Triple(hfName, "F16", onDisk)), 2, 4)

        val weight = load(file, DTypePolicy.Require(FP16))[canonical] ?: error("missing $canonical")

        assertTrue(
            weight.data is Fp16DenseTensorData,
            "KEEP_NATIVE must produce Fp16DenseTensorData, got ${weight.data::class.simpleName}",
        )
        assertTrue(weight.data is NarrowFloatTensorData, "must be recognizable to narrow dispatch")
        assertTrue(
            weight.data !is Bf16TensorData,
            "an F16 tensor must never be mistaken for BF16 — the bit layouts differ",
        )
        // Byte-for-byte identity proves no widening pass ran.
        assertContentEquals(
            onDisk, (weight.data as Fp16DenseTensorData).packedData,
            "KEEP_NATIVE must preserve on-disk F16 bytes verbatim",
        )
        assertEquals(values.size * 2, (weight.data as Fp16DenseTensorData).packedData.size)
    }

    @Test
    fun `KEEP_NATIVE decodes bit-identically to the widening path`() {
        // Both paths apply the same binary16 decode; only the timing differs. Values here are
        // deliberately not all exact in binary16, so a rounding difference would show up.
        val wide = FloatArray(64) { (it - 32) * 0.1f }
        val file = writeSafeTensors(listOf(Triple(hfName, "F16", fp32ToFp16Bytes(wide))), 8, 8)

        val widened = load(file, DTypePolicy.Any)[canonical]!!.data.copyToFloatArray()
        val native = load(file, DTypePolicy.Require(FP16))[canonical]!!.data.copyToFloatArray()

        assertEquals(widened.size, native.size)
        for (i in widened.indices) {
            assertEquals(
                widened[i].toRawBits(), native[i].toRawBits(),
                "bit-identity expected at $i: widened=${widened[i]} native=${native[i]}",
            )
        }
    }

    @Test
    fun `a policy naming one narrow format widens the other`() {
        val f16Name = "model.layers.0.self_attn.q_proj.weight"
        val bf16Name = "model.layers.0.self_attn.k_proj.weight"
        val file = writeSafeTensors(
            listOf(
                Triple(f16Name, "F16", fp32ToFp16Bytes(values)),
                Triple(bf16Name, "BF16", fp32ToBf16Bytes(values)),
            ),
            2, 4,
        )
        val f16Canonical = LlamaTensorNames.attnQ(0)
        val bf16Canonical = LlamaTensorNames.attnK(0)

        // Require(FP16): F16 stays packed, BF16 widens — it cannot be re-encoded as F16.
        val a = load(file, DTypePolicy.Require(FP16))
        assertTrue(a[f16Canonical]!!.data is Fp16DenseTensorData, "F16 should be packed")
        assertTrue(a[bf16Canonical]!!.data is FloatArrayTensorData<*>, "BF16 should be widened")

        // ...and the mirror image.
        val b = load(file, DTypePolicy.Require(BF16))
        assertTrue(b[f16Canonical]!!.data is FloatArrayTensorData<*>, "F16 should be widened")
        assertTrue(b[bf16Canonical]!!.data is Bf16TensorData, "BF16 should be packed")
    }

    @Test
    fun `Prefer and OneOf reach the same KEEP_NATIVE path as Require`() {
        val file = writeSafeTensors(listOf(Triple(hfName, "F16", fp32ToFp16Bytes(values))), 2, 4)

        assertTrue(load(file, DTypePolicy.Prefer(FP16))[canonical]!!.data is Fp16DenseTensorData)
        assertTrue(
            load(file, DTypePolicy.OneOf(setOf(FP32, FP16)))[canonical]!!.data is Fp16DenseTensorData,
        )
        // A soft policy naming neither narrow format leaves the widening default in place.
        assertTrue(load(file, DTypePolicy.Prefer(FP32))[canonical]!!.data is FloatArrayTensorData<*>)
    }
}
