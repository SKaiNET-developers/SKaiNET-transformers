package sk.ainet.models.functiongemma

import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Assumptions.assumeTrue
import sk.ainet.compile.hlo.toStableHlo
import sk.ainet.compile.opt.passes.BlockQuantMatmulFusionPass
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.lang.tensor.data.Q4_0Quantizer
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.operators.bind
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.trace.PackedConstantHandling
import sk.ainet.lang.types.FP32
import sk.ainet.models.gemma.GemmaModel
import sk.ainet.models.gemma.GemmaNetworkLoader
import sk.ainet.models.gemma.GemmaWeightLoader
import sk.ainet.tape.Execution
import java.io.File
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * One FunctionGemma decode step for an NPU with block-quantized weights (Torq `mixed_quant_matmul`):
 * Q8_0 projections kept packed and folded by [BlockQuantMatmulFusionPass], the tied vocabulary
 * projection as a Q4_0 copy of the embedding, the embedding lookup left to the caller, and a static
 * past of [PAST] positions.
 *
 * Inputs, in order: the scaled embedding `[1, 640]`, cos/sin `[1, 256]` for the global and the
 * sliding RoPE base, then K and V `[1, 1, PAST, 256]` per layer. Outputs: logits `[1, vocab]`, then
 * the extended K and V per layer.
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
        val headDim = md.getHeadDim(0)
        val nKV = md.kvHeadCount

        val embedding = weights.tensors.getValue("token_embd.weight")
        println("token_embd data: ${embedding.data::class.qualifiedName}, q proj: ${weights.tensors.getValue("blk.0.attn_q.weight").data::class.qualifiedName}")
        val vocab = embedding.shape[0]
        val hidden = embedding.shape[1]
        // The tied table as stored (Q8_0), decoded once and re-encoded as Q4_0: the Torq compiler
        // hangs on the Q8 vocabulary projection.
        val rawEmbedding = StreamingGGUFReader.open(JvmRandomAccessSource.open(FunctionGemmaFixture.gguf)).use { it.loadTensor("token_embd.weight") }
        val dense = Q8_0BlockTensorData(Shape(vocab, hidden), rawEmbedding).toFloatArray()
        @Suppress("UNCHECKED_CAST")
        val lmHeadQ4 = ctx.fromData(Q4_0Quantizer.quantize(dense, Shape(vocab, hidden)) as TensorData<FP32, Float>, FP32::class)

        // Inputs are bound to the tape context: a bare VoidOpsTensor runs operator overloads
        // (`x * x` in RMSNorm) on its own untraced ops, and the result is captured as a constant.
        val tapeCtx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        fun input(shape: Shape) = voidF32(shape).bind(tapeCtx)
        val embedded = input(Shape(1, hidden))
        val cosG = input(Shape(1, headDim)); val sinG = input(Shape(1, headDim))
        val cosS = input(Shape(1, headDim)); val sinS = input(Shape(1, headDim))
        val kIn = List(nLayers) { input(Shape(1, nKV, PAST, headDim)) }
        val vIn = List(nLayers) { input(Shape(1, nKV, PAST, headDim)) }

        var outputs: GemmaModel.GemmaWithPastOutput<FP32, Float>? = null
        val tape = tapeCtx.record {
            val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
            Execution.tapeStack.pushTape(ct)
            try {
                outputs = model.forwardEmbeddedWithPast(
                    embedded, GemmaModel.RopeCosSin(cosG, sinG, cosS, sinS), kIn, vIn, this as ExecutionContext, lmHeadWeight = lmHeadQ4,
                )
            } finally {
                Execution.tapeStack.popTape()
            }
        }.first
        val graph = (tape as DefaultExecutionTape).toComputeGraph(
            synthesizeExternalInputs = true, packedConstants = PackedConstantHandling.KEEP_PACKED,
        )
        val fused = BlockQuantMatmulFusionPass().apply(graph)
        fused.diagnostics.filter { it.startsWith("BLOCK_QUANT_SKIPPED") }.forEach(::println)
        val folded = fused.diagnostics.count { it.startsWith("BLOCK_QUANT_FUSED") }
        println("folded $folded block-quant matmuls")
        assertTrue(folded == nLayers * 7 + 1, "expected every projection and the vocabulary projection folded, got $folded")

        val module = toStableHlo(fused.graph, "decode_step")
        File(out).mkdirs()
        File(out, "decode_step.mlir").writeText(module.content)
        // Which function argument / result is which tensor: the emitter orders them by graph topology.
        fun id(t: Tensor<FP32, Float>) = tapeCtx.session.refOf(t as Tensor<*, *>).id
        val inRoles = mutableMapOf(id(embedded) to "embedded", id(cosG) to "cosG", id(sinG) to "sinG", id(cosS) to "cosS", id(sinS) to "sinS")
        kIn.forEachIndexed { i, t -> inRoles[id(t)] = "k$i" }
        vIn.forEachIndexed { i, t -> inRoles[id(t)] = "v$i" }
        val o = outputs!!
        val outRoles = mutableMapOf(id(o.logits) to "logits")
        o.selfK.forEachIndexed { i, t -> outRoles[id(t)] = "k$i" }
        o.selfV.forEachIndexed { i, t -> outRoles[id(t)] = "v$i" }
        File(out, "io.txt").writeText(
            module.inputSpecs.mapIndexed { i, s -> "in $i ${inRoles[s.name] ?: "?"} ${s.name} ${s.shape}" }.joinToString("\n") + "\n" +
                module.outputSpecs.mapIndexed { i, s -> "out $i ${outRoles[s.name] ?: "?"} ${s.name} ${s.shape}" }.joinToString("\n") + "\n"
        )
    }

    private fun voidF32(shape: Shape): Tensor<FP32, Float> = VoidOpsTensor(
        object : TensorData<FP32, Float> {
            override val shape: Shape = shape
            // Unreadable, like the shape-only results of VoidTensorOps: an eager shortcut that
            // reads its input (RMSNormalization's fused path) must not bake this input in as a constant.
            override fun get(vararg indices: Int): Float = throw UnsupportedOperationException("graph input has no values")
            override fun set(vararg indices: Int, value: Float) {}
        },
        FP32::class,
    )

    private companion object {
        const val PAST = 16
    }
}
