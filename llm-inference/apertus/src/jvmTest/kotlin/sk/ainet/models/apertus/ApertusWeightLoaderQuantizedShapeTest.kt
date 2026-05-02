package sk.ainet.models.apertus

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.io.gguf.export.GGUFWriter
import sk.ainet.io.gguf.export.GgufTensorEntry
import sk.ainet.io.gguf.export.GgufWriteRequest
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int8
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * Regression test for the NATIVE_OPTIMIZED branch of
 * [ApertusWeightLoader.streamingTensorToTensor] for quantized tensor types.
 *
 * Before the fix, this branch shared a body with RAW_BYTES and used the GGUF
 * tensor's *logical* shape when wrapping the raw byte buffer. For block-quantized
 * formats (Q4_K, Q8_0, …) the byte count differs from the logical element count
 * (e.g. Q8_0 stores 34 bytes per 32 elements), so passing the logical shape to
 * `ctx.fromByteArray` would size-mismatch.
 *
 * The fix makes NATIVE_OPTIMIZED wrap the bytes with `Shape(bytes.size)`
 * (mirroring the LlamaWeightLoader pattern). This test pins that contract.
 */
class ApertusWeightLoaderQuantizedShapeTest {

    @Test
    fun nativeOptimized_quantized_tensor_uses_byte_level_shape() {
        // Q8_0: blockSize=32, typeSize=34. Logical shape [32] → 34 bytes.
        val logicalShape = listOf(32)
        val payload = ByteArray(34) { (it - 17).toByte() } // arbitrary distinguishable bytes

        val ggufBytes = buildSingleTensorGgufBytes(
            tensorName = "test.weight",
            quantization = GGMLQuantizationType.Q8_0,
            logicalShape = logicalShape,
            payload = payload
        )

        val tempFile = Files.createTempFile("apertus-q8-0", ".gguf").toFile()
        tempFile.deleteOnExit()
        tempFile.writeBytes(ggufBytes)

        val ctx = DirectCpuExecutionContext.create()
        val source = JvmRandomAccessSource.open(tempFile)
        StreamingGGUFReader.open(source).use { reader ->
            val st = reader.tensors.single { it.name == "test.weight" }
            assertEquals(GGMLQuantizationType.Q8_0, st.tensorType)
            assertEquals(payload.size.toLong(), st.nBytes)

            val loader = ApertusWeightLoader.fromRandomAccess(
                randomAccessProvider = { source },
                quantPolicy = QuantPolicy.NATIVE_OPTIMIZED
            )

            val tensor = loader.streamingTensorToTensor<FP32, Float>(
                ctx, FP32::class, reader, st
            )

            // The fix: shape is byte-level, not the GGUF logical shape.
            assertEquals(
                Shape(payload.size),
                tensor.shape,
                "NATIVE_OPTIMIZED quantized tensor must use Shape(bytes.size)"
            )
        }
    }

    @Test
    fun nativeOptimized_q4k_tensor_uses_byte_level_shape() {
        // Q4_K: blockSize=256, typeSize=144. Logical shape [256] → 144 bytes.
        val logicalShape = listOf(256)
        val payload = ByteArray(144) { (it % 7).toByte() }

        val ggufBytes = buildSingleTensorGgufBytes(
            tensorName = "test.weight",
            quantization = GGMLQuantizationType.Q4_K,
            logicalShape = logicalShape,
            payload = payload
        )

        val tempFile = Files.createTempFile("apertus-q4-k", ".gguf").toFile()
        tempFile.deleteOnExit()
        tempFile.writeBytes(ggufBytes)

        val ctx = DirectCpuExecutionContext.create()
        val source = JvmRandomAccessSource.open(tempFile)
        StreamingGGUFReader.open(source).use { reader ->
            val st = reader.tensors.single { it.name == "test.weight" }
            assertEquals(GGMLQuantizationType.Q4_K, st.tensorType)

            val loader = ApertusWeightLoader.fromRandomAccess(
                randomAccessProvider = { source },
                quantPolicy = QuantPolicy.NATIVE_OPTIMIZED
            )

            val tensor = loader.streamingTensorToTensor<FP32, Float>(
                ctx, FP32::class, reader, st
            )

            assertEquals(Shape(payload.size), tensor.shape)
        }
    }

    /**
     * Build a minimal GGUF byte stream containing exactly one tensor, with
     * caller-controlled quantization tag, logical shape, and raw payload bytes.
     * The payload is written verbatim — no quantization is performed by the
     * writer — which lets us drive the loader's branches with synthetic data.
     */
    private fun buildSingleTensorGgufBytes(
        tensorName: String,
        quantization: GGMLQuantizationType,
        logicalShape: List<Int>,
        payload: ByteArray
    ): ByteArray {
        val ctx = DirectCpuExecutionContext.create()
        // Tensor data is materialized via TensorFlatten.flattenBytes, which iterates
        // 1-byte-per-element for an Int8 tensor — so we shape it as Shape(payload.size).
        val rawTensor = ctx.fromByteArray<Int8, Byte>(
            Shape(payload.size),
            Int8::class,
            payload
        )
        val request = GgufWriteRequest(
            metadata = mapOf("general.architecture" to "apertus"),
            tensors = listOf(
                GgufTensorEntry(
                    ggufName = tensorName,
                    tensor = rawTensor,
                    quantization = quantization,
                    shape = logicalShape
                )
            ),
            tensorMap = mapOf(tensorName to tensorName)
        )
        return GGUFWriter.writeToByteArray(request).second
    }
}
