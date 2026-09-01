package sk.ainet.models.llama

import sk.ainet.lang.nn.dsl.decoder.DecoderGgufWeights
import sk.ainet.lang.nn.dsl.decoder.DecoderTensorNames
import sk.ainet.lang.nn.dsl.decoder.GgufDecoderMetadata

import org.junit.Test
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

class LlamaWeightMapperTest {

    @Test
    fun `maps loader tensors into runtime weights with shape checks`() {
        // dim=4, ff_dim=8, vocab=8, ctx=4
        val metadata = GgufDecoderMetadata(
            architecture = "llama",
            embeddingLength = 4,
            contextLength = 4,
            blockCount = 1,
            headCount = 1,
            kvHeadCount = 1,
            feedForwardLength = 8,
            ropeDimensionCount = 4,
            vocabSize = 8
        )

        val ctx = DefaultDataExecutionContext()
        fun tensor(shape: Shape, size: Int, start: Float): sk.ainet.lang.tensor.Tensor<FP32, Float> {
            val values = FloatArray(size) { i -> start + i }
            return ctx.fromFloatArray(shape, FP32::class, values)
        }

        // GGUF format shapes:
        // - token embeddings: [dim, vocab]
        // - attention weights: [dim, dim] or [dim, kv_dim]
        // - ffn gate/up: [dim, ff_dim]
        // - ffn down: [ff_dim, dim]
        // - output weight: [dim, vocab]
        val tensors = linkedMapOf(
            DecoderTensorNames.TOKEN_EMBEDDINGS to tensor(Shape(8, 4), 32, 0f),    // [vocab, dim]
            DecoderTensorNames.OUTPUT_NORM to tensor(Shape(4), 4, 100f),
            DecoderTensorNames.OUTPUT_WEIGHT to tensor(Shape(8, 4), 32, 200f),     // [vocab, dim]
            DecoderTensorNames.ROPE_FREQS_REAL to tensor(Shape(4, 2), 8, 300f),
            DecoderTensorNames.ROPE_FREQS_IMAG to tensor(Shape(4, 2), 8, 400f),
            DecoderTensorNames.attnNorm(0) to tensor(Shape(4), 4, 10f),
            DecoderTensorNames.attnQ(0) to tensor(Shape(4, 4), 16, 20f),           // [dim, dim]
            DecoderTensorNames.attnK(0) to tensor(Shape(4, 4), 16, 30f),           // [kv_dim, dim]
            DecoderTensorNames.attnV(0) to tensor(Shape(4, 4), 16, 40f),           // [kv_dim, dim]
            DecoderTensorNames.attnOut(0) to tensor(Shape(4, 4), 16, 50f),         // [dim, dim]
            DecoderTensorNames.ffnNorm(0) to tensor(Shape(4), 4, 60f),
            DecoderTensorNames.ffnGate(0) to tensor(Shape(8, 4), 32, 70f),         // [ff_dim, dim]
            DecoderTensorNames.ffnDown(0) to tensor(Shape(4, 8), 32, 80f),         // [dim, ff_dim]
            DecoderTensorNames.ffnUp(0) to tensor(Shape(8, 4), 32, 90f)            // [ff_dim, dim]
        )

        val runtime = LlamaWeightMapper.map(DecoderGgufWeights(metadata, tensors))
        assertEquals(metadata, runtime.metadata)
        assertEquals(1, runtime.layers.size)

        val layer = runtime.layers.first()
        assertEquals(Shape(4), layer.attnNorm.shape)
        assertEquals(Shape(4, 4), layer.wq.shape)
        assertEquals(Shape(4, 8), layer.ffnDown.shape)
        assertNotNull(runtime.ropeFreqReal)
        assertEquals(Shape(4, 2), runtime.ropeFreqReal!!.shape)
    }

    @Test
    fun `maps loader tensors into FP16 runtime weights`() {
        val metadata = GgufDecoderMetadata(
            architecture = "llama",
            embeddingLength = 4,
            contextLength = 4,
            blockCount = 1,
            headCount = 1,
            kvHeadCount = 1,
            feedForwardLength = 8,
            ropeDimensionCount = 4,
            vocabSize = 8
        )

        val ctx = DefaultDataExecutionContext()
        fun tensor(shape: Shape, size: Int, start: Float): sk.ainet.lang.tensor.Tensor<FP16, Float> {
            val values = FloatArray(size) { i -> start + i }
            return ctx.fromFloatArray(shape, FP16::class, values)
        }

        val tensors = linkedMapOf(
            DecoderTensorNames.TOKEN_EMBEDDINGS to tensor(Shape(8, 4), 32, 0f),
            DecoderTensorNames.OUTPUT_NORM to tensor(Shape(4), 4, 100f),
            DecoderTensorNames.OUTPUT_WEIGHT to tensor(Shape(8, 4), 32, 200f),
            DecoderTensorNames.ROPE_FREQS_REAL to tensor(Shape(4, 2), 8, 300f),
            DecoderTensorNames.ROPE_FREQS_IMAG to tensor(Shape(4, 2), 8, 400f),
            DecoderTensorNames.attnNorm(0) to tensor(Shape(4), 4, 10f),
            DecoderTensorNames.attnQ(0) to tensor(Shape(4, 4), 16, 20f),
            DecoderTensorNames.attnK(0) to tensor(Shape(4, 4), 16, 30f),
            DecoderTensorNames.attnV(0) to tensor(Shape(4, 4), 16, 40f),
            DecoderTensorNames.attnOut(0) to tensor(Shape(4, 4), 16, 50f),
            DecoderTensorNames.ffnNorm(0) to tensor(Shape(4), 4, 60f),
            DecoderTensorNames.ffnGate(0) to tensor(Shape(8, 4), 32, 70f),
            DecoderTensorNames.ffnDown(0) to tensor(Shape(4, 8), 32, 80f),
            DecoderTensorNames.ffnUp(0) to tensor(Shape(8, 4), 32, 90f)
        )

        val runtime: LlamaRuntimeWeights<FP16> = LlamaWeightMapper.map(DecoderGgufWeights(metadata, tensors))
        assertEquals(metadata, runtime.metadata)
        assertEquals(1, runtime.layers.size)

        val layer = runtime.layers.first()
        assertEquals(Shape(4), layer.attnNorm.shape)
        assertEquals(Shape(4, 4), layer.wq.shape)
        assertEquals(Shape(4, 8), layer.ffnDown.shape)
        assertNotNull(runtime.ropeFreqReal)
    }
}
