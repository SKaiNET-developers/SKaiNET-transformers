package sk.ainet.models.gemma

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q5_KBlockTensorData
import sk.ainet.lang.types.FP32

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
    fun pack_q5k_produces_block_tensor_with_relaid_bytes() {
        val shape = Shape(2, 512)
        val bytes = ByteArray(2 * 2 * 176)
        for (i in 0 until 4) bytes[i * 176] = (i + 1).toByte()

        val td = packGemmaKQuant<FP32>(bytes, GGMLQuantizationType.Q5_K, shape)
        assertTrue(td is Q5_KBlockTensorData, "Q5_K should pack to Q5_KBlockTensorData")
        // packedData is the block-major relayout of the input.
        val expected = relayoutKSeriesRowMajorToBlockMajor(bytes, shape, 176)
        assertTrue(expected.contentEquals(td.packedData))
    }

    @Test
    fun pack_non_kquant_returns_null() {
        assertNull(packGemmaKQuant<FP32>(ByteArray(34), GGMLQuantizationType.Q8_0, Shape(1, 32)))
    }
}
