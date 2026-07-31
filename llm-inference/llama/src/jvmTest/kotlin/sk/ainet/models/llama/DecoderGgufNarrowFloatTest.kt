package sk.ainet.models.llama

import kotlinx.io.Source
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.dequant.DequantOps
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Bf16DenseTensorData
import sk.ainet.lang.tensor.data.Bf16TensorData
import sk.ainet.lang.tensor.data.Fp16DenseTensorData
import sk.ainet.lang.tensor.data.NarrowFloatTensorData
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DTypePolicy
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Fp16Codec
import sk.ainet.lang.types.Int8
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * Covers the GGUF narrow-float KEEP_NATIVE path added for engine 0.38.0 — the policy decision
 * ([DecoderGgufWeightLoader.keepsNarrowNative]) and the tensor construction
 * ([DecoderGgufWeightLoader.createNarrowTensor]) — without synthesizing a GGUF file. Neither the
 * engine nor this repo ships a GGUF writer, and the surrounding parse/read machinery is already
 * exercised by the dequant path; what is new and worth pinning is these two decisions.
 *
 * The construction test matters most. GGUF header dims are reversed relative to the logical
 * row-major shape, so the FP32 path swaps the `Shape` and moves no bytes
 * (`DequantOps.transposeColumnMajorToRowMajor` returns its input unchanged). The packed path has
 * to do exactly the same: an actual element transpose here would hand the matmul kernel a
 * silently transposed weight matrix — wrong numbers, no exception.
 */
class DecoderGgufNarrowFloatTest {

    private val ctx = DirectCpuExecutionContext()
    private val noopSource: () -> Source = { error("source not used in these tests") }

    private fun loaderWith(policy: DTypePolicy) =
        DecoderGgufWeightLoader(sourceProvider = noopSource, dtypePolicy = policy)

    private fun fp16Bytes(values: FloatArray): ByteArray {
        val out = ByteArray(values.size * 2)
        for (i in values.indices) {
            val bits = Fp16Codec.encode(values[i])
            out[i * 2] = (bits and 0xFF).toByte()
            out[i * 2 + 1] = ((bits ushr 8) and 0xFF).toByte()
        }
        return out
    }

    private fun bf16Bytes(values: FloatArray): ByteArray {
        val out = ByteArray(values.size * 2)
        for (i in values.indices) {
            val bits = (values[i].toRawBits() ushr 16) and 0xFFFF
            out[i * 2] = (bits and 0xFF).toByte()
            out[i * 2 + 1] = ((bits ushr 8) and 0xFF).toByte()
        }
        return out
    }

    // ---------- policy resolution ----------

    @Test
    fun `default policy keeps nothing native`() {
        val loader = loaderWith(DTypePolicy.Any)
        assertFalse(loader.keepsNarrowNative(GGMLQuantizationType.F16, FP32::class))
        assertFalse(loader.keepsNarrowNative(GGMLQuantizationType.BF16, FP32::class))
    }

    @Test
    fun `a policy naming one narrow format leaves the other on the widening path`() {
        val f16 = loaderWith(DTypePolicy.Require(FP16))
        assertTrue(f16.keepsNarrowNative(GGMLQuantizationType.F16, FP32::class))
        assertFalse(
            f16.keepsNarrowNative(GGMLQuantizationType.BF16, FP32::class),
            "BF16 cannot be re-encoded as F16 — it must widen",
        )

        val bf16 = loaderWith(DTypePolicy.Require(BF16))
        assertTrue(bf16.keepsNarrowNative(GGMLQuantizationType.BF16, FP32::class))
        assertFalse(bf16.keepsNarrowNative(GGMLQuantizationType.F16, FP32::class))
    }

    @Test
    fun `soft policies reach the same KEEP_NATIVE decision`() {
        assertTrue(
            loaderWith(DTypePolicy.Prefer(BF16)).keepsNarrowNative(GGMLQuantizationType.BF16, FP32::class),
        )
        assertTrue(
            loaderWith(DTypePolicy.OneOf(setOf(FP32, FP16)))
                .keepsNarrowNative(GGMLQuantizationType.F16, FP32::class),
        )
    }

    @Test
    fun `quantized and F32 source types are never treated as narrow`() {
        val loader = loaderWith(DTypePolicy.OneOf(setOf(BF16, FP16)))
        assertFalse(loader.keepsNarrowNative(GGMLQuantizationType.F32, FP32::class))
        assertFalse(loader.keepsNarrowNative(GGMLQuantizationType.Q4_K, FP32::class))
        assertFalse(loader.keepsNarrowNative(GGMLQuantizationType.Q8_0, FP32::class))
    }

    @Test
    fun `KEEP_NATIVE only applies to an FP32 element type`() {
        val loader = loaderWith(DTypePolicy.Require(FP16))
        // FP16::class here is a request for the FP32-array storage path, not a packing request;
        // Int8 is the RAW_BYTES path. Neither may be silently reinterpreted as packed storage.
        assertFalse(loader.keepsNarrowNative(GGMLQuantizationType.F16, FP16::class))
        assertFalse(loader.keepsNarrowNative(GGMLQuantizationType.F16, Int8::class))
    }

    // ---------- tensor construction ----------

    @Test
    fun `rank-2 construction swaps the shape and moves no bytes`() {
        // GGUF header dims [rows=2, cols=4] describe a logical [4, 2] row-major tensor.
        val values = floatArrayOf(1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f)
        val bytes = fp16Bytes(values)
        val loader = loaderWith(DTypePolicy.Require(FP16))

        val tensor = loader.createNarrowTensor<FP32, Float>(
            ctx, FP32::class, Shape(2, 4), bytes, GGMLQuantizationType.F16,
        )

        assertContentEquals(
            intArrayOf(4, 2), tensor.shape.dimensions,
            "GGUF [rows, cols] must be reinterpreted as [cols, rows]",
        )
        val data = tensor.data as Fp16DenseTensorData
        assertSame(
            bytes, data.packedData,
            "the on-disk buffer must become the tensor's storage — no copy, no transpose",
        )
        // Element order is untouched, so a flat decode matches the source values in file order.
        assertContentEquals(values, data.copyToFloatArray())
    }

    @Test
    fun `rank-1 construction passes the shape through`() {
        val values = floatArrayOf(0.5f, -0.5f, 3.0f, -7.0f)
        val loader = loaderWith(DTypePolicy.Require(BF16))

        val tensor = loader.createNarrowTensor<FP32, Float>(
            ctx, FP32::class, Shape(4), bf16Bytes(values), GGMLQuantizationType.BF16,
        )

        assertContentEquals(intArrayOf(4), tensor.shape.dimensions)
        assertContentEquals(values, tensor.data.copyToFloatArray())
    }

    @Test
    fun `the source type picks the codec, not the byte width`() {
        // Both formats are 2 bytes per element, so a mix-up cannot fail loudly. Pin it from both
        // sides: the right wrapper type, and a decode that visibly differs from the other codec's.
        val values = floatArrayOf(1.0f, 2.0f, 4.0f, 8.0f)
        val loader = loaderWith(DTypePolicy.OneOf(setOf(BF16, FP16)))

        val asF16 = loader.createNarrowTensor<FP32, Float>(
            ctx, FP32::class, Shape(4), fp16Bytes(values), GGMLQuantizationType.F16,
        )
        assertTrue(asF16.data is Fp16DenseTensorData)
        assertFalse(asF16.data is Bf16TensorData, "F16 must never be mistaken for BF16")
        assertContentEquals(values, asF16.data.copyToFloatArray())

        val asBf16 = loader.createNarrowTensor<FP32, Float>(
            ctx, FP32::class, Shape(4), bf16Bytes(values), GGMLQuantizationType.BF16,
        )
        assertTrue(asBf16.data is Bf16DenseTensorData)
        assertContentEquals(values, asBf16.data.copyToFloatArray())

        // The same bytes read through the wrong codec do NOT coincide — so a dispatch that
        // confused the two could not pass these assertions by luck.
        val f16BytesOfValues = fp16Bytes(values)
        val misread = Bf16DenseTensorData.fromRawBytes(Shape(4), f16BytesOfValues).copyToFloatArray()
        assertTrue(
            misread.indices.any { kotlin.math.abs(misread[it] - values[it]) > 1e-3f },
            "test is vacuous if the two codecs agree on these bytes",
        )
    }

    @Test
    fun `KEEP_NATIVE decoding matches the widening path bit for bit`() {
        // The widening path for GGUF F16 is DequantOps.dequantF16FromBytes; KEEP_NATIVE defers
        // the identical decode to read time. Values are chosen not to be exact in binary16.
        val values = FloatArray(32) { (it - 16) * 0.1f }
        val bytes = fp16Bytes(values)
        val loader = loaderWith(DTypePolicy.Require(FP16))

        val widened = DequantOps.dequantF16FromBytes(bytes)
        val native = loader.createNarrowTensor<FP32, Float>(
            ctx, FP32::class, Shape(32), bytes, GGMLQuantizationType.F16,
        ).data.copyToFloatArray()

        assertEquals(widened.size, native.size)
        for (i in widened.indices) {
            assertEquals(
                widened[i].toRawBits(), native[i].toRawBits(),
                "bit-identity expected at $i: widened=${widened[i]} native=${native[i]}",
            )
        }
    }

    @Test
    fun `a short buffer is rejected rather than read out of bounds`() {
        val loader = loaderWith(DTypePolicy.Require(FP16))
        val toosmall = ByteArray(6) // 3 elements' worth for a 2x4 = 8-element tensor
        val error = kotlin.runCatching {
            loader.createNarrowTensor<FP32, Float>(
                ctx, FP32::class, Shape(2, 4), toosmall, GGMLQuantizationType.F16,
            )
        }.exceptionOrNull()
        assertTrue(error is IllegalArgumentException, "expected IllegalArgumentException, got $error")
    }

    @Test
    fun `the wrapper reports itself to narrow-float dispatch`() {
        val loader = loaderWith(DTypePolicy.Require(BF16))
        val tensor = loader.createNarrowTensor<FP32, Float>(
            ctx, FP32::class, Shape(4), bf16Bytes(floatArrayOf(1f, 2f, 3f, 4f)), GGMLQuantizationType.BF16,
        )
        assertTrue(
            tensor.data is NarrowFloatTensorData,
            "DefaultCpuOpsJvm.chooseQuantizedMatmul matches on NarrowFloatTensorData",
        )
    }
}
