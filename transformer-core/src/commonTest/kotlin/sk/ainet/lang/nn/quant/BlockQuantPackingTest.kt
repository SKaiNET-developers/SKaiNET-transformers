package sk.ainet.lang.nn.quant

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q4_0BlockTensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q5_0BlockTensorData
import sk.ainet.lang.tensor.data.Q5_1BlockTensorData
import sk.ainet.lang.tensor.data.Q5_KBlockTensorData
import sk.ainet.lang.tensor.data.Q6_KBlockTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32

/**
 * Round-trip tests for the shared GGUF-block → `*BlockTensorData` packer
 * (#184 hoist 2) — the logic previously duplicated by gemma's
 * `GemmaQuantLayout`, llama's `LlamaQuantLayout` and apertus' JVM converter.
 * commonTest: runs on JVM and Kotlin/Native alike.
 */
class BlockQuantPackingTest {

    /** All seven packed-kernel encodings with their `(blockElems, bytesPerBlock)`. */
    private val encodings: List<Triple<TensorEncoding, Int, Int>> = listOf(
        Triple(TensorEncoding.Q4_K, 256, 144),
        Triple(TensorEncoding.Q5_K, 256, 176),
        Triple(TensorEncoding.Q6_K, 256, 210),
        Triple(TensorEncoding.Q8_0, 32, 34),
        Triple(TensorEncoding.Q4_0, 32, 18),
        Triple(TensorEncoding.Q5_0, 32, 22),
        Triple(TensorEncoding.Q5_1, 32, 24),
    )

    @Test
    fun block_layout_matches_encoding_constants() {
        for ((enc, blockElems, bpb) in encodings) {
            val layout = BlockQuantPacking.blockLayoutFor(enc)
            assertEquals(blockElems to bpb, layout, "layout for ${enc.name}")
            // Geometry must agree with the encoding's own physicalBytes.
            assertEquals(
                bpb.toLong(),
                enc.physicalBytes(blockElems.toLong()),
                "physicalBytes(${enc.name}) disagrees with packer geometry",
            )
        }
        assertNull(BlockQuantPacking.blockLayoutFor(TensorEncoding.Dense(4)))
        assertNull(BlockQuantPacking.blockLayoutFor(TensorEncoding.TernaryPacked))
    }

    @Test
    fun relayout_is_block_level_transpose_for_every_geometry() {
        for ((enc, blockElems, bpb) in encodings) {
            val outDim = 3
            val blocksPerRow = 4
            val inDim = blocksPerRow * blockElems
            val bytes = ByteArray(outDim * blocksPerRow * bpb)
            // Tag each source block with its row-major index in its first byte.
            for (i in 0 until outDim * blocksPerRow) bytes[i * bpb] = i.toByte()

            val relaid = BlockQuantPacking.relayoutRowMajorToBlockMajor(
                bytes, Shape(outDim, inDim), bpb, blockElems,
            )

            // dst block (b*outDim + r) must hold src block (r*blocksPerRow + b).
            for (r in 0 until outDim) {
                for (b in 0 until blocksPerRow) {
                    val srcIdx = r * blocksPerRow + b
                    val dstIdx = b * outDim + r
                    assertEquals(
                        srcIdx.toByte(), relaid[dstIdx * bpb],
                        "${enc.name}: block ($r,$b) misplaced",
                    )
                }
            }
        }
    }

    @Test
    fun pack_produces_the_matching_block_tensor_data_with_canonical_bytes_verbatim() {
        for ((enc, blockElems, bpb) in encodings) {
            val outDim = 2
            val blocksPerRow = 2
            val shape = Shape(outDim, blocksPerRow * blockElems)
            val bytes = ByteArray(outDim * blocksPerRow * bpb)
            for (i in 0 until outDim * blocksPerRow) bytes[i * bpb] = (i + 1).toByte()

            val td = BlockQuantPacking.pack<FP32>(bytes, enc, shape)
                ?: error("${enc.name}: pack unexpectedly returned null")
            val packedData = when (td) {
                is Q4_KBlockTensorData -> { assertEquals(TensorEncoding.Q4_K, enc); td.packedData }
                is Q5_KBlockTensorData -> { assertEquals(TensorEncoding.Q5_K, enc); td.packedData }
                is Q6_KBlockTensorData -> { assertEquals(TensorEncoding.Q6_K, enc); td.packedData }
                is Q8_0BlockTensorData -> { assertEquals(TensorEncoding.Q8_0, enc); td.packedData }
                is Q4_0BlockTensorData -> { assertEquals(TensorEncoding.Q4_0, enc); td.packedData }
                is Q5_0BlockTensorData -> { assertEquals(TensorEncoding.Q5_0, enc); td.packedData }
                is Q5_1BlockTensorData -> { assertEquals(TensorEncoding.Q5_1, enc); td.packedData }
                else -> error("${enc.name}: unexpected packed type ${td::class.simpleName}")
            }
            // Canonical row-major order is kept verbatim: the engine's packed
            // ops.transpose (>= 0.40.1) performs the physical block-grid
            // permutation itself and requires this order as its input.
            assertTrue(
                bytes.contentEquals(packedData),
                "${enc.name}: packedData must be the checkpoint bytes verbatim (canonical order)",
            )
            assertEquals(shape, td.shape, "${enc.name}: logical shape must be preserved")
        }
    }

    @Test
    fun pack_rejects_non_2d_and_misaligned_shapes() {
        assertFailsWith<IllegalArgumentException> {
            BlockQuantPacking.pack<FP32>(ByteArray(64), TensorEncoding.Q4_0, Shape(64))
        }
        assertFailsWith<IllegalArgumentException> {
            // inDim 33 not a multiple of block size 32.
            BlockQuantPacking.pack<FP32>(ByteArray(64), TensorEncoding.Q4_0, Shape(2, 33))
        }
        assertFailsWith<IllegalArgumentException> {
            // Buffer too small for [2, 64] @ 18 B/block (= 4 blocks = 72 B).
            BlockQuantPacking.pack<FP32>(ByteArray(71), TensorEncoding.Q4_0, Shape(2, 64))
        }
    }

    @Test
    fun packPreTransposed_swaps_shape_and_carries_the_marker() {
        for ((enc, blockElems, bpb) in encodings) {
            val outDim = 3
            val blocksPerRow = 2
            val shape = Shape(outDim, blocksPerRow * blockElems)
            val bytes = ByteArray(outDim * blocksPerRow * bpb)
            for (i in 0 until outDim * blocksPerRow) bytes[i * bpb] = (i + 1).toByte()

            val td = BlockQuantPacking.packPreTransposed<FP32>(bytes, enc, shape)
                ?: error("${enc.name}: packPreTransposed unexpectedly returned null")
            assertTrue(td is PreTransposedWeight, "${enc.name}: missing PreTransposedWeight marker")
            assertEquals(
                Shape(shape[1], shape[0]), td.shape,
                "${enc.name}: logical shape must be the transposed [in, out]",
            )
            // Relaid block-major bytes — the load-time equivalent of the
            // engine's physical packed ops.transpose (>= 0.40.1) applied to
            // pack()'s canonical result.
            val storage = td as PackedBlockStorage
            val expectedRelaid = BlockQuantPacking.relayoutRowMajorToBlockMajor(bytes, shape, bpb, blockElems)
            assertTrue(
                expectedRelaid.contentEquals(storage.packedData),
                "${enc.name}: pre-transposed packedData must be the block-major relayout",
            )
            assertEquals(enc, storage.encoding, "${enc.name}: encoding must survive the marker wrapper")
        }
        assertNull(
            BlockQuantPacking.packPreTransposed<FP32>(ByteArray(2048), TensorEncoding.Dense(4), Shape(2, 256)),
        )
    }

    @Test
    fun pack_returns_null_for_unpackable_encodings() {
        val shape = Shape(2, 256)
        assertNull(BlockQuantPacking.pack<FP32>(ByteArray(2048), TensorEncoding.Dense(4), shape))
        assertNull(BlockQuantPacking.pack<FP32>(ByteArray(2048), TensorEncoding.TernaryPacked, shape))
    }

    @Test
    fun relayout_rejects_non_2d_and_misaligned_shapes() {
        assertFailsWith<IllegalArgumentException> {
            BlockQuantPacking.relayoutRowMajorToBlockMajor(ByteArray(0), Shape(32), 18, 32)
        }
        assertFailsWith<IllegalArgumentException> {
            // inDim 33 not a multiple of block size 32.
            BlockQuantPacking.relayoutRowMajorToBlockMajor(ByteArray(64), Shape(2, 33), 18, 32)
        }
        assertFailsWith<IllegalArgumentException> {
            // Buffer too small for [2, 64] @ 18 B/block (= 4 blocks = 72 B).
            BlockQuantPacking.relayoutRowMajorToBlockMajor(ByteArray(71), Shape(2, 64), 18, 32)
        }
    }
}
