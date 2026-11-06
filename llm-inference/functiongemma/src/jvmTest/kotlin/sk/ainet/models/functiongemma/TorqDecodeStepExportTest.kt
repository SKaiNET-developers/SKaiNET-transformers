package sk.ainet.models.functiongemma

import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Assumptions.assumeTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.lang.nn.dsl.decoder.DecodeStepExport
import sk.ainet.lang.nn.dsl.decoder.DecodeStepSpec
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q4_0Quantizer
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.FP32
import sk.ainet.models.gemma.GemmaModel
import sk.ainet.models.gemma.GemmaNetworkLoader
import sk.ainet.models.gemma.GemmaWeightLoader
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * One FunctionGemma decode step for an NPU with block-quantized weights (Torq `mixed_quant_matmul`),
 * through [DecodeStepExport]: Q8_0 projections kept packed and folded, the tied vocabulary projection
 * as a Q4_0 copy of the embedding, the embedding lookup left to the caller, and a static past of
 * [PAST] positions. Writes `decode_step.mlir` and `io.txt` (which argument and result is which tensor).
 *
 * Runs only with `TORQ_DECODE_OUT` set (the output directory) and the Q8_0 checkpoint in `GEMMA_GGUF`.
 */
class TorqDecodeStepExportTest {

    @Test
    fun exportDecodeStepWithPackedWeights() = runBlocking {
        val out = System.getenv("TORQ_DECODE_OUT")
        assumeTrue(out != null, "set TORQ_DECODE_OUT to export the Torq decode step")
        FunctionGemmaFixture.assumeRealCheckpointRunnable()

        val ctx = DirectCpuExecutionContext.create()
        val weights = GemmaWeightLoader(randomAccessProvider = { JvmRandomAccessSource.open(FunctionGemmaFixture.gguf) })
            .loadToMapStreaming<FP32, Float>(ctx, FP32::class)
        @Suppress("UNCHECKED_CAST")
        val model = GemmaNetworkLoader.fromWeights(ctx, weights, FP32::class) as GemmaModel<FP32, Float>
        val md = weights.metadata
        val nLayers = md.blockCount

        val embedding = weights.tensors.getValue("token_embd.weight")
        val vocab = embedding.shape[0]
        val hidden = embedding.shape[1]
        // The tied table as stored (Q8_0), decoded once and re-encoded as Q4_0: the Torq compiler
        // hangs on the Q8 vocabulary projection.
        val rawEmbedding = StreamingGGUFReader.open(JvmRandomAccessSource.open(FunctionGemmaFixture.gguf)).use { it.loadTensor("token_embd.weight") }
        val dense = Q8_0BlockTensorData(Shape(vocab, hidden), rawEmbedding).toFloatArray()
        @Suppress("UNCHECKED_CAST")
        val lmHeadQ4 = ctx.fromData(Q4_0Quantizer.quantize(dense, Shape(vocab, hidden)) as TensorData<FP32, Float>, FP32::class)

        val spec = DecodeStepSpec(
            hidden = hidden, headDim = md.getHeadDim(0), nKVHeads = md.kvHeadCount, nLayers = nLayers,
            past = PAST, slidingRopeBase = true,
        )
        val step = DecodeStepExport.export<FP32, Float>(spec, FP32::class) { inputs, ectx ->
            model.forwardEmbeddedWithPast(inputs.embedded, inputs.rope, inputs.selfK, inputs.selfV, ectx, lmHeadWeight = lmHeadQ4)
        }
        step.diagnostics.filter { it.startsWith("BLOCK_QUANT_SKIPPED") }.forEach(::println)
        assertEquals(nLayers * 7 + 1, step.foldedBlockQuantMatmuls, "every projection and the vocabulary projection folded")

        File(out).mkdirs()
        File(out, "decode_step.mlir").writeText(step.mlir)
        File(out, "io.txt").writeText(step.ioTable())
    }

    private companion object {
        const val PAST = 16
    }
}
