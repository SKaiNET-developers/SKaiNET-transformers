package sk.ainet.models.gemma

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.lang.nn.quant.PreTransposedWeight
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q5_KBlockTensorData
import sk.ainet.lang.tensor.data.Q5_KTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.Q8_0TensorData
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int8

/**
 * Unit tests for the commonMain (board-shareable) Gemma quant layout helpers.
 * These run on every target (JVM + Kotlin/Native), proving the K/N board path's
 * relayout + packing logic without needing the full model.
 */
class GemmaQuantLayoutTest {

    @Test
    fun relayout_is_block_level_transpose() {
        // [outDim=2, inDim=512] -> blocksPerRow=2, 4 Q5_K blocks of 176 B.
        val bpb = 176
        val outDim = 2
        val inDim = 512
        val blocksPerRow = inDim / 256
        val bytes = ByteArray(outDim * blocksPerRow * bpb)
        // Tag each source block with its row-major index in its first byte.
        for (i in 0 until outDim * blocksPerRow) bytes[i * bpb] = i.toByte()

        val relaid = relayoutKSeriesRowMajorToBlockMajor(bytes, Shape(outDim, inDim), bpb)

        // dst block (b*outDim + r) must hold src block (r*blocksPerRow + b).
        for (r in 0 until outDim) {
            for (b in 0 until blocksPerRow) {
                val srcIdx = r * blocksPerRow + b
                val dstIdx = b * outDim + r
                assertEquals(srcIdx.toByte(), relaid[dstIdx * bpb], "block ($r,$b) misplaced")
            }
        }
    }

    @Test
    fun pack_q5k_defaults_to_pre_transposed_with_relaid_bytes() {
        // #184 (3)/#170, engine 0.40.0 closure train: packGemmaKQuant now
        // defaults to the pre-transposed marked path.
        val shape = Shape(2, 512)
        val bytes = ByteArray(2 * 2 * 176)
        for (i in 0 until 4) bytes[i * 176] = (i + 1).toByte()

        val td = packGemmaKQuant<FP32>(bytes, GGMLQuantizationType.Q5_K, shape)
        assertTrue(td is PreTransposedWeight, "Q5_K should default to the pre-transposed marked path")
        assertTrue(td is Q5_KTensorData, "the marked wrapper still satisfies Q5_KTensorData dispatch checks")
        assertEquals(Shape(512, 2), td.shape, "pre-transposed result carries the swapped [in, out] shape")
        // packedData is still the block-major relayout of the input — only the
        // logical shape + marker changed, not the bytes.
        val expected = relayoutKSeriesRowMajorToBlockMajor(bytes, shape, 176)
        assertTrue(td is PackedBlockStorage)
        assertTrue(expected.contentEquals((td as PackedBlockStorage).packedData))
    }

    @Test
    fun pack_q5k_preTransposed_false_keeps_classic_block_tensor_reachable() {
        // Deprecate-don't-delete: the non-transposed packer path stays reachable
        // for fallback / parity comparison against the pre-transposed default.
        val shape = Shape(2, 512)
        val bytes = ByteArray(2 * 2 * 176)
        for (i in 0 until 4) bytes[i * 176] = (i + 1).toByte()

        val td = packGemmaKQuant<FP32>(bytes, GGMLQuantizationType.Q5_K, shape, preTransposed = false)
        assertTrue(td is Q5_KBlockTensorData, "Q5_K should pack to the classic Q5_KBlockTensorData when opted out")
        assertEquals(shape, td.shape, "classic path keeps the checkpoint's [out, in] shape")
        val expected = relayoutKSeriesRowMajorToBlockMajor(bytes, shape, 176)
        assertTrue(expected.contentEquals(td.packedData))
    }

    @Test
    fun pack_q8_0_defaults_to_pre_transposed_block_tensor() {
        // Q8_0 is packed (32 elems / 34 B per block) so a tied Q8_0 lm_head
        // stays packed and runs on the Q8_0 kernel instead of dequanting to FP32;
        // as of the 0.40.0 closure train it packs pre-transposed by default.
        val td = packGemmaKQuant<FP32>(ByteArray(34), GGMLQuantizationType.Q8_0, Shape(1, 32))
        assertTrue(td is PreTransposedWeight, "Q8_0 should default to the pre-transposed marked path")
        assertTrue(td is Q8_0TensorData, "the marked wrapper still satisfies Q8_0TensorData dispatch checks")
    }

    @Test
    fun pack_q8_0_preTransposed_false_keeps_classic_block_tensor_reachable() {
        val td = packGemmaKQuant<FP32>(ByteArray(34), GGMLQuantizationType.Q8_0, Shape(1, 32), preTransposed = false)
        assertTrue(td is Q8_0BlockTensorData, "Q8_0 should pack to the classic Q8_0BlockTensorData when opted out")
    }

    @Test
    fun pack_unsupported_quant_returns_null() {
        // A quant type with no packed kernel (e.g. Q4_1) falls back to FP32 dequant.
        assertNull(packGemmaKQuant<FP32>(ByteArray(20), GGMLQuantizationType.Q4_1, Shape(1, 32)))
    }

    @Test
    fun extract_raw_bytes_roundtrips_on_every_platform() {
        // The NATIVE_OPTIMIZED loader wraps quant bytes via ctx.fromByteArray<Int8,Byte>;
        // extractRawBytes must read them back regardless of the platform backing
        // (JVM IntArrayTensorData vs native Byte-typed). Runs on jvm + linuxX64.
        val ctx = DirectCpuExecutionContext.create()
        val bytes = ByteArray(176 * 3) { ((it * 31 + 7) and 0xFF).toByte() }
        val t = ctx.fromByteArray<Int8, Byte>(Shape(bytes.size), Int8::class, bytes)
        val got = extractRawBytes(t.data)
        assertTrue(bytes.contentEquals(got), "extractRawBytes round-trip mismatch")
    }
}
