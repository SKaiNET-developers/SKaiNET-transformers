package sk.ainet.lang.nn.dsl.decoder

import kotlinx.io.Buffer
import kotlinx.io.readByteArray
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.safetensors.SafeTensorsWriter
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.NarrowFloatInputMajorTensorData
import sk.ainet.lang.tensor.data.NarrowFloatTensorData
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.Bf16Codec
import sk.ainet.lang.types.DTypePolicy
import sk.ainet.lang.types.FP32
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * End-to-end fixture for the engine-backed [DecoderSafeTensorsLoader] (SKaiNET#1246): a
 * synthetic single-file LLaMA checkpoint written with the engine's `SafeTensorsWriter`, run
 * through the engine `SafeTensorsParametersLoader` and this loader's family-side policy —
 * HF → GGUF renaming, `[1, dim]` norm normalization, tied embeddings, keep-native relayout.
 * The legacy `Q4` + `.qb` export is pinned separately because it cannot ride the engine.
 */
class DecoderSafeTensorsLoaderFixtureTest {

    private val dim = 4
    private val ffn = 4
    private val vocab = 4

    private val metadata = GgufDecoderMetadata(
        architecture = "llama",
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
    fun `engine path maps every slot, normalizes norms, ties the output and drops unmapped tensors`() {
        val file = writeHfCheckpoint()
        val t = load(file, DTypePolicy.Any, tied = true)

        val expected = listOf(
            DecoderTensorNames.TOKEN_EMBEDDINGS, DecoderTensorNames.OUTPUT_NORM, DecoderTensorNames.OUTPUT_WEIGHT,
            DecoderTensorNames.attnNorm(0), DecoderTensorNames.attnQ(0), DecoderTensorNames.attnK(0),
            DecoderTensorNames.attnV(0), DecoderTensorNames.attnOut(0), DecoderTensorNames.ffnNorm(0),
            DecoderTensorNames.ffnGate(0), DecoderTensorNames.ffnUp(0), DecoderTensorNames.ffnDown(0),
        )
        for (slot in expected) assertNotNull(t[slot], "slot $slot must be populated; got ${t.keys}")
        assertEquals(expected.toSet(), t.keys, "no extra slots — the rotary decoy must not be delivered")
        assertFalse(t.keys.any { "rotary" in it })

        // [1, dim] norms are normalized to [dim]; matmul weights keep [out, in].
        assertEquals(Shape(dim), t[DecoderTensorNames.OUTPUT_NORM]!!.shape)
        assertEquals(Shape(dim), t[DecoderTensorNames.attnNorm(0)]!!.shape)
        assertEquals(Shape(dim, dim), t[DecoderTensorNames.attnQ(0)]!!.shape)
        assertEquals(Shape(vocab, dim), t[DecoderTensorNames.TOKEN_EMBEDDINGS]!!.shape)

        // Tied: output.weight is the very same tensor as token_embd.
        assertSame(t[DecoderTensorNames.TOKEN_EMBEDDINGS], t[DecoderTensorNames.OUTPUT_WEIGHT])

        // Values round-trip exactly (bf16-representable fixtures) and Any widens to FP32.
        assertContentEquals(values(dim * dim, Q_BASE), floats(t[DecoderTensorNames.attnQ(0)]!!))
        assertContentEquals(values(dim, NORM_BASE), floats(t[DecoderTensorNames.OUTPUT_NORM]!!))
        assertContentEquals(values(vocab * dim, EMBED_BASE), floats(t[DecoderTensorNames.TOKEN_EMBEDDINGS]!!))
        assertTrue(t[DecoderTensorNames.attnQ(0)]!!.data is FloatArrayTensorData<*>, "Any must widen BF16")
    }

    @Test
    fun `Require BF16 keeps matmul weights packed input-major and leaves F32 norms dense`() {
        val file = writeHfCheckpoint()
        val t = load(file, DTypePolicy.Require(BF16), tied = false)

        val q = t[DecoderTensorNames.attnQ(0)]!!.data
        assertTrue(q is NarrowFloatInputMajorTensorData, "a matmul weight must be relaid input-major, got ${q::class.simpleName}")
        assertEquals(Bf16Codec, (q as NarrowFloatTensorData).codec)
        assertContentEquals(values(dim * dim, Q_BASE), t[DecoderTensorNames.attnQ(0)]!!.data.copyToFloatArray())

        // The gathered embedding stays row-major (still packed), norms are dense FP32.
        val embed = t[DecoderTensorNames.TOKEN_EMBEDDINGS]!!.data
        assertTrue(embed is NarrowFloatTensorData && embed !is NarrowFloatInputMajorTensorData)
        assertTrue(t[DecoderTensorNames.OUTPUT_NORM]!!.data is FloatArrayTensorData<*>)
        assertEquals(Shape(dim), t[DecoderTensorNames.OUTPUT_NORM]!!.shape)
    }

    @Test
    fun `legacy Q4 plus qb export still loads through the retained reader`() {
        // 4x4 q_proj: nibbles 1..4 per row, one FP32 scale per row → q * scale.
        val rows = dim; val cols = dim
        val q4 = ByteArray(rows * cols / 2)
        for (r in 0 until rows) for (c in 0 until cols) {
            val flat = r * cols + c
            val nib = (c + 1) and 0x0F
            val idx = flat / 2
            q4[idx] = if (flat % 2 == 0) (q4[idx].toInt() or nib).toByte() else (q4[idx].toInt() or (nib shl 4)).toByte()
        }
        val scales = floatArrayOf(0.5f, 0.25f, 1.0f, 2.0f)
        val file = writeRaw(
            listOf(
                Entry("model.layers.0.self_attn.q_proj.weight", "Q4", listOf(rows, cols), q4),
                Entry("model.layers.0.self_attn.q_proj.weight.qb", "F32", listOf(rows, 1), f32Bytes(scales)),
            )
        )
        val t = load(file, DTypePolicy.Any, tied = false)
        val expected = FloatArray(rows * cols) { (it % cols + 1).toFloat() * scales[it / cols] }
        assertContentEquals(expected, floats(t[DecoderTensorNames.attnQ(0)]!!))
        assertFalse(t.containsKey("${DecoderTensorNames.attnQ(0)}.qb"))
    }

    // ---- fixtures ----

    private fun load(file: File, policy: DTypePolicy, tied: Boolean): Map<String, Tensor<FP32, Float>> {
        val loader = DecoderSafeTensorsLoader(
            ctx = DirectCpuExecutionContext(),
            dtype = FP32::class,
            metadata = metadata,
            tiedEmbeddings = tied,
            dtypePolicy = policy,
        )
        val provider: () -> RandomAccessSource = { JvmRandomAccessSource.open(file) }
        return loader.loadToMap(provider).tensors
    }

    /** A 1-layer LLaMA-style HF checkpoint: BF16 projections + embedding, F32 norms stored `[1, dim]`. */
    private fun writeHfCheckpoint(): File {
        val buffer = Buffer()
        SafeTensorsWriter.write(buffer) {
            tensorBF16("model.embed_tokens.weight", listOf(vocab.toLong(), dim.toLong()), values(vocab * dim, EMBED_BASE))
            tensorF32("model.norm.weight", listOf(1L, dim.toLong()), values(dim, NORM_BASE))
            val l = "model.layers.0"
            tensorF32("$l.input_layernorm.weight", listOf(1L, dim.toLong()), values(dim, LN_BASE))
            tensorF32("$l.post_attention_layernorm.weight", listOf(1L, dim.toLong()), values(dim, 2.5f))
            tensorBF16("$l.self_attn.q_proj.weight", listOf(dim.toLong(), dim.toLong()), values(dim * dim, Q_BASE))
            tensorBF16("$l.self_attn.k_proj.weight", listOf(dim.toLong(), dim.toLong()), values(dim * dim, 0.125f))
            tensorBF16("$l.self_attn.v_proj.weight", listOf(dim.toLong(), dim.toLong()), values(dim * dim, 0.25f))
            tensorBF16("$l.self_attn.o_proj.weight", listOf(dim.toLong(), dim.toLong()), values(dim * dim, 0.375f))
            tensorBF16("$l.mlp.gate_proj.weight", listOf(ffn.toLong(), dim.toLong()), values(ffn * dim, 0.5f))
            tensorBF16("$l.mlp.up_proj.weight", listOf(ffn.toLong(), dim.toLong()), values(ffn * dim, 0.625f))
            tensorBF16("$l.mlp.down_proj.weight", listOf(dim.toLong(), ffn.toLong()), values(dim * ffn, 0.75f))
            // Unmapped float tensor: materialized by the engine, dropped by the family mapping.
            tensorF32("model.rotary_emb.inv_freq", listOf(2L), floatArrayOf(1.0f, 0.5f))
        }
        val file = Files.createTempFile("decoder_st_fixture", ".safetensors").toFile().also { it.deleteOnExit() }
        file.writeBytes(buffer.readByteArray())
        return file
    }

    private data class Entry(val name: String, val dtype: String, val shape: List<Int>, val bytes: ByteArray)

    /** Hand-written header for dtypes the writer has no typed helper for (the legacy Q4 export). */
    private fun writeRaw(entries: List<Entry>): File {
        val header = StringBuilder("{")
        var offset = 0L
        entries.forEachIndexed { i, e ->
            if (i > 0) header.append(",")
            header.append("\"${e.name}\":{\"dtype\":\"${e.dtype}\",\"shape\":[${e.shape.joinToString(",")}],\"data_offsets\":[$offset,${offset + e.bytes.size}]}")
            offset += e.bytes.size
        }
        header.append("}")
        val headerBytes = header.toString().toByteArray(Charsets.UTF_8)
        val file = Files.createTempFile("decoder_st_q4", ".safetensors").toFile().also { it.deleteOnExit() }
        file.outputStream().use { out ->
            out.write(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(headerBytes.size.toLong()).array())
            out.write(headerBytes)
            entries.forEach { out.write(it.bytes) }
        }
        return file
    }

    private fun f32Bytes(values: FloatArray): ByteArray {
        val bb = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        values.forEach { bb.putFloat(it) }
        return bb.array()
    }

    /** bf16-exact values: base + i/8 stays within 7 mantissa bits for these tiny extents. */
    private fun values(n: Int, base: Float) = FloatArray(n) { base + (it % 16) * 0.125f }
    private fun floats(t: Tensor<FP32, Float>): FloatArray = t.data.copyToFloatArray()

    private companion object {
        const val EMBED_BASE = 1.0f
        const val NORM_BASE = 2.0f
        const val Q_BASE = -1.0f
        const val LN_BASE = 0.5f
    }
}
