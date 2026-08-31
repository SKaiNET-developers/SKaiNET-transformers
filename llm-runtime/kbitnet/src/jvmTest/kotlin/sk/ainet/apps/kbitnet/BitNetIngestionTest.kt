package sk.ainet.apps.kbitnet

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlinx.coroutines.runBlocking
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.models.bitnet.BitNetTensorNames

/**
 * transformers#359: the `BitNetIngestion` facade round-trip — a tied synthetic BitNet I2_S GGUF
 * (compact sibling of the writer in `BitNetWeightLoaderTest`) loads through the facade into a
 * `BitNetRuntimeWeights` whose planes head gates the two-stage decode and whose `toModule()`
 * binds a runnable network.
 */
class BitNetIngestionTest {

    private val ctx = DirectCpuExecutionContext()

    private val dim = 16
    private val ffDim = 32
    private val vocabSize = 32

    private class T(val name: String, val type: Int, val dims: LongArray, val data: ByteArray)

    /** BitNet.cpp `quantize_i2_s` GROUP_128 packing + 32-byte trailer with the scale ×8. */
    private fun i2sTensor(name: String, out: Int, inDim: Int, scale: Float, seed: Int): T {
        val rng = Random(seed)
        val elements = out * inDim
        check(elements % 128 == 0)
        val payload = ByteArray(elements / 4)
        for (j in 0 until elements) {
            val jb = j % 128
            val idx = (j / 128) * 32 + jb % 32
            payload[idx] = (payload[idx].toInt() or (rng.nextInt(3) shl (6 - 2 * (jb / 32)))).toByte()
        }
        val trailer = ByteBuffer.allocate(32).order(ByteOrder.LITTLE_ENDIAN)
        repeat(8) { trailer.putFloat(scale) }
        return T(name, 36, longArrayOf(inDim.toLong(), out.toLong()), payload + trailer.array())
    }

    private fun f32Tensor(name: String, values: FloatArray, vararg neDims: Long): T {
        val buf = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        values.forEach { buf.putFloat(it) }
        return T(name, 0, neDims, buf.array())
    }

    private fun norm(n: Int, seed: Int): FloatArray {
        val rng = Random(seed)
        return FloatArray(n) { 0.8f + rng.nextFloat() * 0.4f }
    }

    /** Tied 2B4T-style file: no `output.weight`. */
    private fun writeTiedGguf(): File {
        val arch = "bitnet-b1.58"
        val rng = Random(42)
        val embd = FloatArray(vocabSize * dim) { (rng.nextFloat() - 0.5f) * 0.6f }
        val tensors = listOf(
            f32Tensor("token_embd.weight", embd, dim.toLong(), vocabSize.toLong()),
            f32Tensor("output_norm.weight", norm(dim, 20), dim.toLong()),
            f32Tensor("blk.0.attn_norm.weight", norm(dim, 21), dim.toLong()),
            i2sTensor("blk.0.attn_q.weight", dim, dim, scale = 0.11f, seed = 1),
            i2sTensor("blk.0.attn_k.weight", dim, dim, scale = 0.12f, seed = 2),
            i2sTensor("blk.0.attn_v.weight", dim, dim, scale = 0.13f, seed = 3),
            f32Tensor("blk.0.attn_sub_norm.weight", norm(dim, 22), dim.toLong()),
            i2sTensor("blk.0.attn_output.weight", dim, dim, scale = 0.14f, seed = 4),
            f32Tensor("blk.0.ffn_norm.weight", norm(dim, 23), dim.toLong()),
            i2sTensor("blk.0.ffn_gate.weight", ffDim, dim, scale = 0.15f, seed = 5),
            i2sTensor("blk.0.ffn_up.weight", ffDim, dim, scale = 0.16f, seed = 6),
            f32Tensor("blk.0.ffn_sub_norm.weight", norm(ffDim, 24), ffDim.toLong()),
            i2sTensor("blk.0.ffn_down.weight", dim, ffDim, scale = 0.17f, seed = 7),
        )

        fun u32(v: Int): (ByteBuffer) -> Unit = { it.putInt(4); it.putInt(v) }
        fun f32v(v: Float): (ByteBuffer) -> Unit = { it.putInt(6); it.putFloat(v) }
        fun str(v: String): (ByteBuffer) -> Unit = { b ->
            b.putInt(8); val e = v.encodeToByteArray(); b.putLong(e.size.toLong()); b.put(e)
        }
        val kvs: List<Pair<String, (ByteBuffer) -> Unit>> = listOf(
            "general.architecture" to str(arch),
            "$arch.embedding_length" to u32(dim),
            "$arch.context_length" to u32(16),
            "$arch.block_count" to u32(1),
            "$arch.attention.head_count" to u32(2),
            "$arch.attention.head_count_kv" to u32(2),
            "$arch.feed_forward_length" to u32(ffDim),
            "$arch.rope.dimension_count" to u32(dim / 2),
            "$arch.vocab_size" to u32(vocabSize),
            "$arch.attention.layer_norm_rms_epsilon" to f32v(1e-5f),
        )

        val head = ByteBuffer.allocate(64 * 1024).order(ByteOrder.LITTLE_ENDIAN)
        head.putInt(0x46554747)
        head.putInt(3)
        head.putLong(tensors.size.toLong())
        head.putLong(kvs.size.toLong())
        for ((key, write) in kvs) {
            val k = key.encodeToByteArray()
            head.putLong(k.size.toLong()); head.put(k)
            write(head)
        }
        fun padded(size: Int): Int = ((size + 31) / 32) * 32
        var dataOffset = 0L
        for (t in tensors) {
            val n = t.name.encodeToByteArray()
            head.putLong(n.size.toLong()); head.put(n)
            head.putInt(t.dims.size)
            for (d in t.dims) head.putLong(d)
            head.putInt(t.type)
            head.putLong(dataOffset)
            dataOffset += padded(t.data.size).toLong()
        }
        repeat((32 - (head.position() % 32)) % 32) { head.put(0) }

        val file = File.createTempFile("kbitnet_i2s_", ".gguf")
        file.deleteOnExit()
        RandomAccessFile(file, "rw").use { raf ->
            raf.write(head.array(), 0, head.position())
            for (t in tensors) {
                raf.write(t.data)
                repeat(padded(t.data.size) - t.data.size) { raf.write(0) }
            }
        }
        return file
    }

    @Test
    fun streamingFacadeLoadsPackedWeightsWithAPlanesHead() {
        val file = writeTiedGguf()
        try {
            val weights = runBlocking {
                BitNetIngestion(ctx).loadStreaming({ JvmRandomAccessSource.open(file.path) })
            }
            assertEquals("bitnet-b1.58", weights.metadata.architecture)
            assertTrue(BitNetTensorNames.TOKEN_EMBEDDING in weights.tensors)
            assertNotNull(
                weights.planesHead,
                "the tied lm_head must be materialized as BITNET_PLANES (#357)",
            )
            // The container binds into a runnable network.
            val model = weights.toModule()
            assertTrue(model.modules.isNotEmpty())

            // Opting out of the planes head keeps the dense tied path.
            val dense = runBlocking {
                BitNetIngestion(ctx).loadStreaming(
                    { JvmRandomAccessSource.open(file.path) },
                    planesLmHead = false,
                )
            }
            assertNull(dense.planesHead)
        } finally {
            file.delete()
        }
    }
}
