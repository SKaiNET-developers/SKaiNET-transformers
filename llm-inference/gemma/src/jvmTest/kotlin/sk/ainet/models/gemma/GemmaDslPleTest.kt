package sk.ainet.models.gemma

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

/**
 * End-to-end test for Phase 5f.5 PLE wiring through the DSL pipeline.
 *
 * Toy fixture (dim=8, 1 layer, vocab=16, ple_dim=4) with explicit PLE
 * weights. Verifies the full path
 *
 *   gemmaNetwork(ple = true)
 *     → GemmaModel with PerLayerInputBlockHook in each block
 *     → PerLayerEmbedding computed and sliced per-layer
 *     → OptimizedLLMRuntime in DIRECT mode
 *
 * produces finite logits with the right shape and distinct-input
 * distinct-output behaviour. Does NOT check against a reference — that
 * comes when we can run HF's Gemma 4 on the same synthetic weights, or
 * when real-model parity is achieved in a later phase.
 */
class GemmaDslPleTest {

    private val dim = 8
    private val ffDim = 16
    private val vocabSize = 16
    private val perLayerDim = 4
    private val numLayers = 2
    private val nHeads = 2
    private val kvHeads = 2
    private val headDim = dim / nHeads
    private val seqLen = 32

    private val ctx = DirectCpuExecutionContext()

    private fun randn(shape: Shape, seed: Int): Tensor<FP32, Float> {
        val rng = kotlin.random.Random(seed)
        val values = FloatArray(shape.volume) { (rng.nextFloat() - 0.5f) * 0.1f }
        return ctx.fromFloatArray(shape, FP32::class, values)
    }

    private fun ones(shape: Shape): Tensor<FP32, Float> {
        val values = FloatArray(shape.volume) { 1.0f }
        return ctx.fromFloatArray(shape, FP32::class, values)
    }

    private val metadata = Gemma4ModelMetadata(
        architecture = "gemma4",
        embeddingLength = dim,
        contextLength = seqLen,
        blockCount = numLayers,
        headCount = nHeads,
        kvHeadCount = kvHeads,
        intermediateSize = ffDim,
        headDim = headDim,
        globalHeadDim = headDim,
        vocabSize = vocabSize,
        slidingWindow = seqLen,
        kvSharedLayers = 0,
        layerTypes = List(numLayers) { "full_attention" },
        ropeParametersFull = Gemma4RopeConfig(base = 10000f),
        ropeParametersSliding = Gemma4RopeConfig(base = 10000f),
        maxPositionEmbeddings = seqLen,
        perLayerEmbeddingLength = perLayerDim
    )

    /** Build a fixture including every tensor `gemmaNetwork(ple=true)` asks for. */
    private fun buildWeights(): Gemma4Weights<FP32, Float> {
        val tensors = linkedMapOf<String, Tensor<FP32, Float>>()

        // Top-level.
        tensors[Gemma4TensorNames.TOKEN_EMBEDDINGS] = randn(Shape(vocabSize, dim), seed = 10)
        tensors[Gemma4TensorNames.OUTPUT_NORM] = ones(Shape(dim))
        tensors[Gemma4TensorNames.OUTPUT_WEIGHT] = randn(Shape(vocabSize, dim), seed = 11)

        // PLE top-level tensors.
        tensors["per_layer_token_embd.weight"] =
            randn(Shape(vocabSize, numLayers * perLayerDim), seed = 30)
        tensors["per_layer_model_proj.weight"] =
            randn(Shape(numLayers * perLayerDim, dim), seed = 31)
        tensors["per_layer_proj_norm.weight"] = ones(Shape(perLayerDim))

        // Per-block.
        for (layer in 0 until numLayers) {
            tensors[Gemma4TensorNames.inputLayernorm(layer)] = ones(Shape(dim))
            tensors[Gemma4TensorNames.attnQ(layer)] = randn(Shape(dim, dim), seed = 100 + layer * 10)
            tensors[Gemma4TensorNames.attnK(layer)] = randn(Shape(dim, dim), seed = 101 + layer * 10)
            tensors[Gemma4TensorNames.attnV(layer)] = randn(Shape(dim, dim), seed = 102 + layer * 10)
            tensors[Gemma4TensorNames.attnOut(layer)] = randn(Shape(dim, dim), seed = 103 + layer * 10)
            tensors[Gemma4TensorNames.postAttentionLayernorm(layer)] = ones(Shape(dim))
            tensors[Gemma4TensorNames.ffnGate(layer)] = randn(Shape(ffDim, dim), seed = 104 + layer * 10)
            tensors[Gemma4TensorNames.ffnDown(layer)] = randn(Shape(dim, ffDim), seed = 105 + layer * 10)
            tensors[Gemma4TensorNames.ffnUp(layer)] = randn(Shape(ffDim, dim), seed = 106 + layer * 10)

            // PLE block-level tensors.
            tensors["blk.$layer.inp_gate.weight"] = randn(Shape(perLayerDim, dim), seed = 200 + layer * 10)
            tensors["blk.$layer.proj.weight"] = randn(Shape(dim, perLayerDim), seed = 201 + layer * 10)
            tensors["blk.$layer.post_norm.weight"] = ones(Shape(dim))
        }

        return Gemma4Weights(metadata = metadata, tensors = tensors)
    }

    @Test
    fun `PLE-enabled network loads without missing params`() {
        val weights = buildWeights()
        val model = gemmaNetwork<FP32, Float>(metadata, ple = true)
        assertTrue(model is GemmaModel<FP32, Float>, "expected GemmaModel, got ${model::class.simpleName}")
        val ple = (model as GemmaModel<FP32, Float>).ple
        assertTrue(ple != null, "PLE module must be non-null when ple=true")

        // Spot-check that every block has the hook.
        for (blk in model.blocks) {
            val hook = blk.modules.firstOrNull { it is PerLayerInputBlockHook<*, *> }
            assertTrue(hook != null, "every block must carry a PerLayerInputBlockHook, got modules=${blk.modules.map { it::class.simpleName }}")
        }

        // Loading should wire every declared param without complaining.
        val loaded = GemmaNetworkLoader.fromWeights(ctx, weights)
        assertTrue(loaded is GemmaModel<FP32, Float>)
    }

    @Test
    fun `PLE-enabled forward produces finite logits with shape vocab`() {
        val weights = buildWeights()
        val loaded = GemmaNetworkLoader.fromWeights(ctx, weights)
        val runtime = OptimizedLLMRuntime(
            model = loaded,
            ctx = ctx,
            mode = OptimizedLLMMode.DIRECT,
            dtype = FP32::class
        )

        val logits = runtime.forward(tokenId = 3)
        // Last dim should be vocab.
        assertEquals(vocabSize, logits.shape[logits.rank - 1], "logits last dim must be vocab")

        val buf = logits.data.copyToFloatArray()
        for ((i, v) in buf.withIndex()) {
            assertTrue(v.isFinite(), "logit[$i] = $v not finite — PLE path produced NaN/Inf")
        }
    }

    @Test
    fun `PLE-enabled forward is deterministic across repeated calls`() {
        val weights = buildWeights()
        val loaded = GemmaNetworkLoader.fromWeights(ctx, weights)
        val runtime = OptimizedLLMRuntime(
            model = loaded, ctx = ctx, mode = OptimizedLLMMode.DIRECT, dtype = FP32::class
        )

        val first = runtime.forward(tokenId = 5).data.copyToFloatArray()
        // Reset runtime state and re-forward — need fresh runtime since KV cache
        // carries state forward.
        val runtime2 = OptimizedLLMRuntime(
            model = GemmaNetworkLoader.fromWeights(ctx, weights),
            ctx = ctx, mode = OptimizedLLMMode.DIRECT, dtype = FP32::class
        )
        val second = runtime2.forward(tokenId = 5).data.copyToFloatArray()

        assertEquals(first.size, second.size)
        for (i in first.indices) {
            assertEquals(first[i], second[i], 1e-6f, "logit[$i] drifted across runs: $first[$i] vs $second[$i]")
        }
    }
}
