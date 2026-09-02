package sk.ainet.models.llama

import kotlinx.io.Buffer
import kotlinx.io.readByteArray
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.safetensors.SafeTensorsWriter
import sk.ainet.lang.nn.dsl.decoder.DecoderSafeTensorsLoader
import sk.ainet.lang.nn.dsl.decoder.GgufDecoderMetadata
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.NarrowFloatInputMajorTensorData
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DTypePolicy
import sk.ainet.lang.types.FP32
import java.io.File
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * The llama half of the engine-backed SafeTensors load (SKaiNET#1246): a synthetic HF
 * checkpoint through [DecoderSafeTensorsLoader.load] into structured [LlamaRuntimeWeights],
 * i.e. the engine loader → family renaming → [LlamaWeightMapper] shape contract, end to end.
 */
class DecoderSafeTensorsLoaderLlamaFixtureTest {

    private val dim = 4
    private val ffn = 8
    private val vocab = 6

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
    fun `runtime weights come out of the engine path with the mapper's shapes and values`() {
        val file = writeHfCheckpoint()
        val weights = loader(DTypePolicy.Any).load { JvmRandomAccessSource.open(file) }

        assertEquals(1, weights.layers.size)
        assertEquals(Shape(vocab, dim), weights.tokenEmbedding.shape)
        assertEquals(Shape(dim), weights.outputNorm.shape, "[1, dim] norm must be normalized before mapping")
        assertSame(weights.tokenEmbedding, weights.outputWeight, "tied embeddings")
        val layer = weights.layers[0]
        assertEquals(Shape(dim, dim), layer.wq.shape)
        assertEquals(Shape(ffn, dim), layer.ffnGate.shape)
        assertEquals(Shape(dim, ffn), layer.ffnDown.shape)
        assertContentEquals(values(dim * dim, Q_BASE), layer.wq.data.copyToFloatArray())
        assertContentEquals(values(dim, LN_BASE), layer.attnNorm.data.copyToFloatArray())
        assertTrue(layer.wq.data is FloatArrayTensorData<*>, "Any widens BF16 to FP32")
    }

    @Test
    fun `Require BF16 reaches the runtime weights as input-major packed projections`() {
        val file = writeHfCheckpoint()
        val weights = loader(DTypePolicy.Require(BF16)).load { JvmRandomAccessSource.open(file) }
        val layer = weights.layers[0]
        assertTrue(layer.wq.data is NarrowFloatInputMajorTensorData, "got ${layer.wq.data::class.simpleName}")
        assertTrue(layer.ffnDown.data is NarrowFloatInputMajorTensorData)
        assertContentEquals(values(dim * dim, Q_BASE), layer.wq.data.copyToFloatArray())
        assertTrue(layer.attnNorm.data is FloatArrayTensorData<*>, "F32 norms stay dense")
    }

    private fun loader(policy: DTypePolicy) = DecoderSafeTensorsLoader(
        ctx = DirectCpuExecutionContext(),
        dtype = FP32::class,
        metadata = metadata,
        tiedEmbeddings = true,
        dtypePolicy = policy,
    )

    private fun writeHfCheckpoint(): File {
        val buffer = Buffer()
        SafeTensorsWriter.write(buffer) {
            tensorBF16("model.embed_tokens.weight", listOf(vocab.toLong(), dim.toLong()), values(vocab * dim, 1.0f))
            tensorF32("model.norm.weight", listOf(1L, dim.toLong()), values(dim, 2.0f))
            val l = "model.layers.0"
            tensorF32("$l.input_layernorm.weight", listOf(dim.toLong()), values(dim, LN_BASE))
            tensorF32("$l.post_attention_layernorm.weight", listOf(dim.toLong()), values(dim, 2.5f))
            tensorBF16("$l.self_attn.q_proj.weight", listOf(dim.toLong(), dim.toLong()), values(dim * dim, Q_BASE))
            tensorBF16("$l.self_attn.k_proj.weight", listOf(dim.toLong(), dim.toLong()), values(dim * dim, 0.125f))
            tensorBF16("$l.self_attn.v_proj.weight", listOf(dim.toLong(), dim.toLong()), values(dim * dim, 0.25f))
            tensorBF16("$l.self_attn.o_proj.weight", listOf(dim.toLong(), dim.toLong()), values(dim * dim, 0.375f))
            tensorBF16("$l.mlp.gate_proj.weight", listOf(ffn.toLong(), dim.toLong()), values(ffn * dim, 0.5f))
            tensorBF16("$l.mlp.up_proj.weight", listOf(ffn.toLong(), dim.toLong()), values(ffn * dim, 0.625f))
            tensorBF16("$l.mlp.down_proj.weight", listOf(dim.toLong(), ffn.toLong()), values(dim * ffn, 0.75f))
        }
        val file = Files.createTempFile("llama_st_fixture", ".safetensors").toFile().also { it.deleteOnExit() }
        file.writeBytes(buffer.readByteArray())
        return file
    }

    private fun values(n: Int, base: Float) = FloatArray(n) { base + (it % 16) * 0.125f }

    private companion object {
        const val Q_BASE = -1.0f
        const val LN_BASE = 0.5f
    }
}
