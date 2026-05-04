package sk.ainet.models.llama

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Q4MemorySegmentMarker
import sk.ainet.lang.tensor.data.Q8MemorySegmentMarker
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int8
import java.lang.foreign.Arena

class DecoderGgufMemSegConverterTest {

    private val ctx = DirectCpuExecutionContext()
    private val dim = 32   // multiple of block size (32)
    private val ffn = 64
    private val vocab = 4

    private val metadata = LlamaModelMetadata(
        architecture = "qwen3",
        embeddingLength = dim,
        contextLength = 8,
        blockCount = 1,
        headCount = 1,
        kvHeadCount = 1,
        feedForwardLength = ffn,
        ropeDimensionCount = dim,
        vocabSize = vocab,
    )

    @Test
    fun `empty quantTypes returns weights unchanged`() {
        val weights = DecoderGgufWeights<FP32, Float>(
            metadata = metadata,
            tensors = mapOf(
                "blk.0.attn_norm.weight" to fpTensor(Shape(dim)),
                "output_norm.weight" to fpTensor(Shape(dim)),
            ),
            quantTypes = emptyMap(),
        )

        Arena.ofConfined().use { arena ->
            val out = DecoderGgufMemSegConverter.convert(weights, ctx, arena)
            // Same instance — no-op short-circuit.
            assertSame(weights, out)
        }
    }

    @Test
    fun `Q4_0 tensor is wrapped as Q4 MemSegment`() {
        val rawQ4 = rawQ4Tensor(rows = dim, cols = dim)
        val weights = DecoderGgufWeights<FP32, Float>(
            metadata = metadata,
            tensors = mapOf(
                "blk.0.attn_q.weight" to rawQ4,
                "blk.0.attn_norm.weight" to fpTensor(Shape(dim)), // FP32 pass-through
            ),
            quantTypes = mapOf("blk.0.attn_q.weight" to GGMLQuantizationType.Q4_0),
        )

        Arena.ofConfined().use { arena ->
            val out = DecoderGgufMemSegConverter.convert(weights, ctx, arena)

            val q = out.tensors.getValue("blk.0.attn_q.weight")
            assertTrue(
                q.data is Q4MemorySegmentMarker,
                "Q4_0 tensor must be wrapped as Q4MemorySegmentTensorData; got ${q.data::class.simpleName}",
            )

            val norm = out.tensors.getValue("blk.0.attn_norm.weight")
            assertSame(
                weights.tensors.getValue("blk.0.attn_norm.weight"), norm,
                "FP32 tensor without an entry in quantTypes must pass through unchanged",
            )

            // quantTypes is dropped after conversion — packed tensors carry
            // their own marker, dequantized FP32 tensors have no quant identity.
            assertTrue(out.quantTypes.isEmpty(), "quantTypes should be cleared post-convert")
        }
    }

    @Test
    fun `Q8_0 tensor is wrapped as Q8 MemSegment`() {
        val rawQ8 = rawQ8Tensor(rows = ffn, cols = dim)
        val weights = DecoderGgufWeights<FP32, Float>(
            metadata = metadata,
            tensors = mapOf("blk.0.ffn_gate.weight" to rawQ8),
            quantTypes = mapOf("blk.0.ffn_gate.weight" to GGMLQuantizationType.Q8_0),
        )

        Arena.ofConfined().use { arena ->
            val out = DecoderGgufMemSegConverter.convert(weights, ctx, arena)

            val gate = out.tensors.getValue("blk.0.ffn_gate.weight")
            assertTrue(
                gate.data is Q8MemorySegmentMarker,
                "Q8_0 tensor must be wrapped as Q8MemorySegmentTensorData; got ${gate.data::class.simpleName}",
            )
        }
    }

    @Test
    fun `tensor count and key set are preserved`() {
        val q4 = rawQ4Tensor(dim, dim)
        val q8 = rawQ8Tensor(ffn, dim)
        val weights = DecoderGgufWeights<FP32, Float>(
            metadata = metadata,
            tensors = linkedMapOf(
                "blk.0.attn_q.weight" to q4,
                "blk.0.ffn_gate.weight" to q8,
                "output_norm.weight" to fpTensor(Shape(dim)),
            ),
            quantTypes = mapOf(
                "blk.0.attn_q.weight" to GGMLQuantizationType.Q4_0,
                "blk.0.ffn_gate.weight" to GGMLQuantizationType.Q8_0,
            ),
        )

        Arena.ofConfined().use { arena ->
            val out = DecoderGgufMemSegConverter.convert(weights, ctx, arena)
            assertEquals(weights.tensors.keys, out.tensors.keys)
            assertEquals(weights.tensors.size, out.tensors.size)
        }
    }

    private fun fpTensor(shape: Shape, value: Float = 1f): Tensor<FP32, Float> =
        ctx.full(shape, FP32::class, value)

    /** Build a raw-byte tensor that simulates a NATIVE_OPTIMIZED Q4_0 load. */
    private fun rawQ4Tensor(rows: Int, cols: Int): Tensor<FP32, Float> {
        val nElements = rows * cols
        val blockSize = 32
        val bytesPerBlock = 18
        val nBlocks = nElements / blockSize
        val nBytes = nBlocks * bytesPerBlock

        val bytes = ByteArray(nBytes)
        for (block in 0 until nBlocks) {
            val off = block * bytesPerBlock
            // f16 scale = 0.5 → encode as half-float
            val halfBits = floatToHalf(0.5f)
            bytes[off] = (halfBits and 0xFF).toByte()
            bytes[off + 1] = ((halfBits shr 8) and 0xFF).toByte()
            // Nibble codes: 8 (zero offset) on both halves for simplicity
            for (i in 0 until 16) {
                bytes[off + 2 + i] = 0x88.toByte()
            }
        }

        val tensor = ctx.fromByteArray<Int8, Byte>(Shape(nBytes), Int8::class, bytes)
        @Suppress("UNCHECKED_CAST")
        return tensor as Tensor<FP32, Float>
    }

    /** Build a raw-byte tensor that simulates a NATIVE_OPTIMIZED Q8_0 load. */
    private fun rawQ8Tensor(rows: Int, cols: Int): Tensor<FP32, Float> {
        val nElements = rows * cols
        val blockSize = 32
        val bytesPerBlock = 34
        val nBlocks = nElements / blockSize
        val nBytes = nBlocks * bytesPerBlock

        val bytes = ByteArray(nBytes)
        for (block in 0 until nBlocks) {
            val off = block * bytesPerBlock
            val halfBits = floatToHalf(0.25f)
            bytes[off] = (halfBits and 0xFF).toByte()
            bytes[off + 1] = ((halfBits shr 8) and 0xFF).toByte()
            for (i in 0 until 32) {
                bytes[off + 2 + i] = 0
            }
        }

        val tensor = ctx.fromByteArray<Int8, Byte>(Shape(nBytes), Int8::class, bytes)
        @Suppress("UNCHECKED_CAST")
        return tensor as Tensor<FP32, Float>
    }

    private fun floatToHalf(f: Float): Int {
        val bits = f.toRawBits()
        val sign = (bits ushr 16) and 0x8000
        val exp = ((bits ushr 23) and 0xFF) - 127 + 15
        val mant = bits and 0x7FFFFF
        if (exp <= 0) return sign
        if (exp >= 31) return sign or 0x7C00
        return sign or (exp shl 10) or (mant ushr 13)
    }
}
