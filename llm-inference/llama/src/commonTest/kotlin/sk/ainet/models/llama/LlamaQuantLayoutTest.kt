package sk.ainet.models.llama

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.lang.nn.quant.BlockQuantPacking
import sk.ainet.lang.nn.quant.PreTransposedWeight
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q4_KTensorData
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.types.FP32

/**
 * Unit tests for the commonMain Llama quant layout helpers — the Llama mirror
 * of `GemmaQuantLayoutTest` (which existed since #184; this one was missing
 * until the SKaiNET#968/0.40.1 layout-contract regression showed every
 * converter needs its own packing coverage). Runs on every target.
 */
class LlamaQuantLayoutTest {

    private val q4kBpb = 144
    private val kBlock = 256

    @Test
    fun pack_q4k_defaults_to_pre_transposed_with_relaid_bytes() {
        // [outDim=2, inDim=512] -> blocksPerRow=2, multi-block both ways.
        val shape = Shape(2, 512)
        val bytes = ByteArray(2 * 2 * q4kBpb)
        for (i in 0 until 4) bytes[i * q4kBpb] = (i + 1).toByte()

        val td = packLlamaKQuant<FP32>(bytes, GGMLQuantizationType.Q4_K, shape)
            ?: error("packLlamaKQuant returned null for Q4_K")
        assertTrue(td is PreTransposedWeight, "Q4_K should default to the pre-transposed marked path")
        assertTrue(td is Q4_KTensorData, "the marked wrapper still satisfies Q4_KTensorData dispatch checks")
        assertEquals(Shape(512, 2), td.shape, "pre-transposed result carries the swapped [in, out] shape")
        // Bytes are the input-block-major relayout — kernel feed order,
        // computed once at load time.
        val expected = BlockQuantPacking.relayoutRowMajorToBlockMajor(bytes, shape, q4kBpb, kBlock)
        assertTrue(td is PackedBlockStorage)
        assertTrue(expected.contentEquals((td as PackedBlockStorage).packedData))
    }

    @Test
    fun pack_q4k_preTransposed_false_keeps_canonical_bytes_verbatim() {
        val shape = Shape(2, 512)
        val bytes = ByteArray(2 * 2 * q4kBpb)
        for (i in 0 until 4) bytes[i * q4kBpb] = (i + 1).toByte()

        val td = packLlamaKQuant<FP32>(bytes, GGMLQuantizationType.Q4_K, shape, preTransposed = false)
        assertTrue(td is Q4_KBlockTensorData, "Q4_K should pack to the classic Q4_KBlockTensorData when opted out")
        assertEquals(shape, td.shape, "classic path keeps the checkpoint's [out, in] shape")
        // Canonical checkpoint bytes verbatim: the classic path defers the
        // block-grid permutation to the engine's physical packed ops.transpose
        // (>= 0.40.1) inside linearProject.
        assertTrue(bytes.contentEquals(td.packedData))
    }

    @Test
    fun pack_unsupported_quant_returns_null() {
        assertNull(packLlamaKQuant<FP32>(ByteArray(20), GGMLQuantizationType.Q4_1, Shape(1, 32)))
    }

    @Test
    fun logicalShapeFor_maps_the_2d_matmul_weights() {
        val md = LlamaModelMetadata(
            architecture = "llama",
            embeddingLength = 64,
            contextLength = 128,
            blockCount = 2,
            headCount = 4,
            kvHeadCount = 2,
            feedForwardLength = 256,
            ropeDimensionCount = null,
            vocabSize = 1000,
        )
        assertEquals(Shape(1000, 64), logicalShapeFor(LlamaTensorNames.TOKEN_EMBEDDINGS, md))
        assertEquals(Shape(1000, 64), logicalShapeFor(LlamaTensorNames.OUTPUT_WEIGHT, md))
        assertEquals(Shape(64, 64), logicalShapeFor("blk.0.attn_q.weight", md))
        assertEquals(Shape(32, 64), logicalShapeFor("blk.0.attn_k.weight", md))
        assertEquals(Shape(32, 64), logicalShapeFor("blk.1.attn_v.weight", md))
        assertEquals(Shape(64, 64), logicalShapeFor("blk.0.attn_output.weight", md))
        assertEquals(Shape(256, 64), logicalShapeFor("blk.0.ffn_gate.weight", md))
        assertEquals(Shape(256, 64), logicalShapeFor("blk.1.ffn_up.weight", md))
        assertEquals(Shape(64, 256), logicalShapeFor("blk.0.ffn_down.weight", md))
        assertNull(logicalShapeFor("blk.0.attn_norm.weight", md))
        assertNull(logicalShapeFor("output_norm.weight", md))
    }
}
