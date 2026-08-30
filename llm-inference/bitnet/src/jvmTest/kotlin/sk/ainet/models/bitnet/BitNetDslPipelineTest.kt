package sk.ainet.models.bitnet

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import sk.ainet.models.llama.DecoderGgufWeights
import sk.ainet.models.llama.LlamaModelMetadata

/**
 * transformers#336: the BitNet DSL pipeline end to end on synthetic weights — the module tree,
 * the [BitNetGGUFNameResolver] mapping (incl. the two BitNet-only sub-norm tensors), and a
 * DIRECT-mode forward producing finite logits. Mirrors `QwenDslPipelineTest`.
 */
class BitNetDslPipelineTest {

    private val ctx = DirectCpuExecutionContext()

    private val dim = 8
    private val nHeads = 2
    private val kvHeads = 2
    private val headDim = dim / nHeads
    private val ffDim = 16
    private val vocabSize = 32
    private val seqLen = 16

    private fun randn(shape: Shape, seed: Int): Tensor<FP32, Float> =
        ctx.fromFloatArray(
            shape, FP32::class,
            FloatArray(shape.volume) { (kotlin.math.sin((seed * 1000 + it).toFloat()) * 0.3f) },
        )

    private fun ones(shape: Shape): Tensor<FP32, Float> =
        ctx.fromFloatArray(shape, FP32::class, FloatArray(shape.volume) { 1f })

    private val metadata = LlamaModelMetadata(
        architecture = "bitnet-b1.58",
        embeddingLength = dim,
        contextLength = seqLen,
        blockCount = 1,
        headCount = nHeads,
        kvHeadCount = kvHeads,
        feedForwardLength = ffDim,
        ropeDimensionCount = headDim,
        vocabSize = vocabSize,
    )

    private fun buildWeightTensors(): Map<String, Tensor<FP32, Float>> = linkedMapOf(
        "token_embd.weight" to randn(Shape(vocabSize, dim), seed = 10),
        "output_norm.weight" to ones(Shape(dim)),
        "output.weight" to randn(Shape(vocabSize, dim), seed = 11),
        "blk.0.attn_norm.weight" to ones(Shape(dim)),
        "blk.0.attn_q.weight" to randn(Shape(dim, dim), seed = 1),
        "blk.0.attn_k.weight" to randn(Shape(dim, dim), seed = 2),
        "blk.0.attn_v.weight" to randn(Shape(dim, dim), seed = 3),
        "blk.0.attn_sub_norm.weight" to ones(Shape(dim)),
        "blk.0.attn_output.weight" to randn(Shape(dim, dim), seed = 4),
        "blk.0.ffn_norm.weight" to ones(Shape(dim)),
        "blk.0.ffn_gate.weight" to randn(Shape(ffDim, dim), seed = 5),
        "blk.0.ffn_up.weight" to randn(Shape(ffDim, dim), seed = 7),
        "blk.0.ffn_sub_norm.weight" to ones(Shape(ffDim)),
        "blk.0.ffn_down.weight" to randn(Shape(dim, ffDim), seed = 6),
    )

    @Test
    fun networkDefinitionBuildsTheBitNetModuleTree() {
        val model = bitnetNetwork<FP32, Float>(metadata)
        val topLevelNames = model.modules.map { it.name }
        assertTrue("token_embd" in topLevelNames, "Should have token_embd module")
        assertTrue("blk.0" in topLevelNames, "Should have blk.0 module")
        assertTrue("output_norm" in topLevelNames, "Should have output_norm module")
        assertTrue("output" in topLevelNames, "Should have output module")
    }

    @Test
    fun weightLoadingMapsEveryParameterIncludingTheSubNorms() {
        // BitNetNetworkLoader.applyWeightsToNetwork REQUIRES every non-bias parameter to map —
        // if attn_sub_norm / ffn_sub_norm failed to resolve, this throws with the missing list.
        val weights = DecoderGgufWeights<FP32, Float>(metadata, buildWeightTensors())
        val model = BitNetNetworkLoader.fromWeights(weights)
        assertTrue(model.modules.isNotEmpty(), "Model should have child modules after weight loading")
    }

    @Test
    fun directModeForwardProducesFiniteLogitsWithCorrectShape() {
        val weights = DecoderGgufWeights<FP32, Float>(metadata, buildWeightTensors())
        val model = BitNetNetworkLoader.fromWeights(weights)
        val runtime = OptimizedLLMRuntime(
            model = model,
            ctx = ctx,
            mode = OptimizedLLMMode.DIRECT,
            dtype = FP32::class,
        )
        for (tokenId in intArrayOf(1, 5, 3)) {
            val logits = runtime.forward(tokenId)
            assertEquals(vocabSize, logits.shape[logits.shape.rank - 1], "logits over the vocab")
            for (i in 0 until vocabSize) {
                val v = logits.data.get(0, i)
                assertTrue(v.isFinite(), "logit[$i] must be finite, was $v")
            }
        }
    }
}
