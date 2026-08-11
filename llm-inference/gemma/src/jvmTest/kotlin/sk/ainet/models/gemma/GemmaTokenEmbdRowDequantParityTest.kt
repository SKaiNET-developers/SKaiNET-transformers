package sk.ainet.models.gemma

import java.lang.foreign.Arena
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.dequant.DequantOps
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Locks the remaining half of #178 / #184 item (1): the MAIN `token_embd` table stays packed and its
 * gather dequantises rows on demand — with parity against the dense FP32 dequant.
 *
 * Three locks over a synthetic Q8_0 table:
 * 1. [GemmaPerLayerTokenEmbedTensorData.dequantRow] is bit-identical to the corresponding rows of a
 *    full dense [DequantOps.dequantFromBytes] pass (exact — same dequant kernel either way).
 * 2. The ENGINE `ops.gather` (SKaiNET >= 0.39.0) recognises the wrapper as a
 *    `sk.ainet.lang.tensor.data.RowDequantSource` and gathers via `dequantRow` — this only holds now
 *    that the gemma wrappers implement the engine interface (previously they implemented a
 *    transformer-core-local duplicate the engine could not see, and a plain `ops.gather` would have
 *    thrown via the wrapper's poisoned `get`).
 * 3. [convertGemmaWeightsToMemSeg] (the JVM eager path) keeps a row-sliceable quant `token_embd`
 *    PACKED instead of inflating it to dense FP32 (~0.67 GB for FunctionGemma's 262153x640 table).
 */
class GemmaTokenEmbdRowDequantParityTest {

    private val vocab = 8
    private val dim = 64 // 2 Q8_0 blocks per row
    private val blockElems = 32
    private val bytesPerBlock = 34

    /** fp16 bit patterns for the block scales we cycle through: 0.5, 1.0, 2.0. */
    private val scaleBits = intArrayOf(0x3800, 0x3C00, 0x4000)

    /** Synthetic Q8_0 payload: per block a 2-byte LE fp16 scale + 32 int8 values, row-major. */
    private fun buildQ8_0Bytes(): ByteArray {
        val blocksPerRow = dim / blockElems
        val bytes = ByteArray(vocab * blocksPerRow * bytesPerBlock)
        var off = 0
        var blockIdx = 0
        for (row in 0 until vocab) {
            for (b in 0 until blocksPerRow) {
                val bits = scaleBits[blockIdx % scaleBits.size]
                bytes[off] = (bits and 0xFF).toByte()
                bytes[off + 1] = ((bits ushr 8) and 0xFF).toByte()
                for (j in 0 until blockElems) {
                    // deterministic, row- and position-distinct int8 values in [-128, 127]
                    bytes[off + 2 + j] = ((row * 37 + b * 11 + j * 3) % 256 - 128).toByte()
                }
                off += bytesPerBlock
                blockIdx++
            }
        }
        return bytes
    }

    private fun denseBaseline(bytes: ByteArray): FloatArray =
        DequantOps.dequantFromBytes(bytes, GGMLQuantizationType.Q8_0, vocab * dim)

    @Test
    fun dequantRowMatchesDenseDequant() {
        val bytes = buildQ8_0Bytes()
        val packed = GemmaPerLayerTokenEmbedTensorData(Shape(vocab, dim), GGMLQuantizationType.Q8_0, bytes)
        val dense = denseBaseline(bytes)
        for (row in intArrayOf(0, 1, 3, vocab - 1)) {
            val got = packed.dequantRow(row)
            assertEquals(dim, got.size, "row $row width")
            for (c in 0 until dim) {
                assertEquals(dense[row * dim + c], got[c], "row $row col $c") // exact: same kernel
            }
        }
    }

    @Test
    fun engineGatherConsumesRowDequantSource() {
        val ctx = DirectCpuExecutionContext()
        val bytes = buildQ8_0Bytes()
        val packed = GemmaPerLayerTokenEmbedTensorData(Shape(vocab, dim), GGMLQuantizationType.Q8_0, bytes)
        @Suppress("UNCHECKED_CAST")
        val weight = ctx.fromData<FP32, Float>(packed as TensorData<FP32, Float>, FP32::class)
        val lookup = intArrayOf(2, 5, 2) // repeat included
        val idx: Tensor<Int32, Float> = ctx.fromIntArray(Shape(lookup.size), Int32::class, lookup)

        // Plain engine gather — NOT the Embedding layer's manual row-dequant path. Before the
        // re-parent to sk.ainet.lang.tensor.data.RowDequantSource this threw the wrapper's
        // "get is unsupported" error; now the engine's row-dequant branch handles it.
        @Suppress("UNCHECKED_CAST")
        val out = ctx.ops.gather(weight, idx as Tensor<DType, *>, dim = 0)

        val dense = denseBaseline(bytes)
        val got = out.data.copyToFloatArray()
        assertEquals(lookup.size * dim, got.size)
        for ((r, token) in lookup.withIndex()) {
            for (c in 0 until dim) {
                assertEquals(dense[token * dim + c], got[r * dim + c], "gathered row $r (token $token) col $c")
            }
        }
    }

    @Test
    fun memSegConverterKeepsTokenEmbdPacked() {
        val ctx = DirectCpuExecutionContext()
        val bytes = buildQ8_0Bytes()

        /** Raw-byte carrier mimicking a NATIVE_OPTIMIZED-loaded quant tensor (1-D byte shape). */
        class RawBytes(private val raw: ByteArray) : TensorData<Int32, Int> {
            override val shape: Shape = Shape(raw.size)
            override fun get(vararg indices: Int): Int = raw[indices[0]].toInt()
            override fun set(vararg indices: Int, value: Int): Unit = error("read-only")
        }

        val rawTensor = ctx.fromData(RawBytes(bytes), Int32::class)
        val metadata = Gemma4ModelMetadata(
            architecture = "gemma4",
            embeddingLength = dim,
            contextLength = 32,
            blockCount = 1,
            headCount = 1,
            kvHeadCount = 1,
            intermediateSize = dim,
            headDim = dim,
            globalHeadDim = dim,
            vocabSize = vocab,
            slidingWindow = 32,
            kvSharedLayers = 0,
            layerTypes = listOf("full_attention"),
            ropeParametersFull = Gemma4RopeConfig(base = 10000f),
            ropeParametersSliding = Gemma4RopeConfig(base = 10000f),
            maxPositionEmbeddings = 32,
        )
        val weights = Gemma4Weights(
            metadata = metadata,
            tensors = mapOf(Gemma4TensorNames.TOKEN_EMBEDDINGS to rawTensor),
            quantTypes = mapOf(Gemma4TensorNames.TOKEN_EMBEDDINGS to GGMLQuantizationType.Q8_0),
            logicalShapes = mapOf(Gemma4TensorNames.TOKEN_EMBEDDINGS to Shape(vocab, dim)),
        )

        Arena.ofConfined().use { arena ->
            val converted = convertGemmaWeightsToMemSeg(weights, ctx, arena)
            val embd = converted.tensors.getValue(Gemma4TensorNames.TOKEN_EMBEDDINGS)
            val data = embd.data
            assertTrue(
                data is GemmaPerLayerTokenEmbedTensorData,
                "row-sliceable Q8_0 token_embd must stay packed, was ${data::class.simpleName}",
            )
            assertEquals(Shape(vocab, dim), data.shape, "wrapper must report the logical 2-D shape")
            val dense = denseBaseline(bytes)
            for (row in intArrayOf(0, vocab - 1)) {
                val got = data.dequantRow(row)
                for (c in 0 until dim) {
                    assertEquals(dense[row * dim + c], got[c], "converted row $row col $c")
                }
            }
        }
    }
}
