package sk.ainet.models.gemma

import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertTrue
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

/**
 * Phase 5c.1: numerical parity between the hand-coded [Gemma4Runtime] and
 * the DSL-based path ([gemmaNetwork] + [OptimizedLLMRuntime] in DIRECT mode).
 *
 * Scope of this file: **one global-attention layer**, no sliding window, no
 * shared KV, no proportional RoPE. This isolates the core pipeline
 * (embedding → RMSNorm → Q/K/V projection → RoPE → causal attention → output
 * projection → residual → RMSNorm → GeGLU FFN → residual → output projection)
 * so any divergence points to a core primitive. Sliding-window and shared-KV
 * variants follow in 5c.2.
 *
 * Both paths consume the **same** [Tensor] objects for every weight, so any
 * output difference comes from implementation choices (op ordering, attention
 * kernel, etc.), not from weight initialisation drift.
 */
class GemmaRuntimeParityTest {

    private val ctx = DirectCpuExecutionContext()

    private val dim = 16
    private val nHeads = 2
    private val nKvHeads = 1
    private val headDim = dim / nHeads
    private val ffnDim = 32
    private val vocabSize = 32
    private val seqLen = 16
    private val nLayers = 1

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
        blockCount = nLayers,
        headCount = nHeads,
        kvHeadCount = nKvHeads,
        intermediateSize = ffnDim,
        headDim = headDim,
        globalHeadDim = headDim,
        vocabSize = vocabSize,
        slidingWindow = seqLen, // Effectively unlimited; not a sliding layer anyway.
        kvSharedLayers = 0,
        layerTypes = listOf("full_attention"),
        // Default RoPE (not proportional) so both paths compute frequencies identically.
        ropeParametersFull = Gemma4RopeConfig(
            base = 10000f,
            ropeType = "default",
            factor = 1.0f,
            partialRotaryFactor = 1.0f
        ),
        ropeParametersSliding = Gemma4RopeConfig(base = 10000f, ropeType = "default"),
        maxPositionEmbeddings = seqLen
    )

    /** Build one shared set of weight tensors used by both runtimes. */
    private fun buildSharedWeights(): SharedWeights {
        val tokenEmbedding = randn(Shape(vocabSize, dim), seed = 10)
        val finalNorm = ones(Shape(dim))
        val lmHead = randn(Shape(vocabSize, dim), seed = 11)

        val layer0Weights = Gemma4LayerWeights(
            inputLayernorm = ones(Shape(dim)),
            wq = randn(Shape(nHeads * headDim, dim), seed = 1),
            wk = randn(Shape(nKvHeads * headDim, dim), seed = 2),
            wv = randn(Shape(nKvHeads * headDim, dim), seed = 3),
            wo = randn(Shape(dim, nHeads * headDim), seed = 4),
            postAttentionLayernorm = ones(Shape(dim)),
            gateProj = randn(Shape(ffnDim, dim), seed = 5),
            upProj = randn(Shape(ffnDim, dim), seed = 6),
            downProj = randn(Shape(dim, ffnDim), seed = 7)
        )

        val runtimeWeights = Gemma4RuntimeWeights(
            metadata = metadata,
            tokenEmbedding = tokenEmbedding,
            ropeFreqReal = null,
            ropeFreqImag = null,
            layers = listOf(layer0Weights),
            finalNorm = finalNorm,
            lmHead = lmHead
        )

        val tensorsByGgufName = linkedMapOf<String, Tensor<FP32, Float>>(
            Gemma4TensorNames.TOKEN_EMBEDDINGS to tokenEmbedding,
            Gemma4TensorNames.OUTPUT_NORM to finalNorm,
            Gemma4TensorNames.OUTPUT_WEIGHT to lmHead,
            Gemma4TensorNames.inputLayernorm(0) to layer0Weights.inputLayernorm,
            Gemma4TensorNames.attnQ(0) to layer0Weights.wq,
            Gemma4TensorNames.attnK(0) to layer0Weights.wk,
            Gemma4TensorNames.attnV(0) to layer0Weights.wv,
            Gemma4TensorNames.attnOut(0) to layer0Weights.wo,
            Gemma4TensorNames.postAttentionLayernorm(0) to layer0Weights.postAttentionLayernorm,
            Gemma4TensorNames.ffnGate(0) to layer0Weights.gateProj,
            Gemma4TensorNames.ffnUp(0) to layer0Weights.upProj,
            Gemma4TensorNames.ffnDown(0) to layer0Weights.downProj
        )
        val gemmaWeights = Gemma4Weights<FP32, Float>(
            metadata = metadata,
            tensors = tensorsByGgufName
        )
        return SharedWeights(runtimeWeights, gemmaWeights)
    }

    private data class SharedWeights(
        val runtime: Gemma4RuntimeWeights<FP32>,
        val dsl: Gemma4Weights<FP32, Float>
    )

    @Test
    fun `single-layer global - Gemma4Runtime and gemmaNetwork agree on forward logits`() {
        val shared = buildSharedWeights()
        val config = Gemma4Config.fromMetadata(metadata)

        // --- Path A: hand-coded Gemma4Runtime + Gemma4AttentionBackend
        val handCodedKv = HeapGemma4KvCache.fromConfig(config, seqLen)
        val handCodedBackend = Gemma4AttentionBackend(ctx, shared.runtime, FP32::class, config, handCodedKv)
        val handCoded: Gemma4Runtime<FP32> = Gemma4Runtime(
            ctx = ctx,
            weights = shared.runtime,
            attentionBackend = handCodedBackend,
            dtype = FP32::class,
            config = config
        )

        // --- Path B: DSL via gemmaNetwork + OptimizedLLMRuntime
        val dslModel = GemmaNetworkLoader.fromWeights(ctx, shared.dsl)
        val dsl = OptimizedLLMRuntime(
            model = dslModel,
            ctx = ctx,
            mode = OptimizedLLMMode.DIRECT,
            dtype = FP32::class
        )

        // Run the same token sequence through both and compare logits.
        val tokens = intArrayOf(1, 5, 3, 0, 2)
        val tolerance = 1e-3f
        var maxDiff = 0f
        for ((step, tokenId) in tokens.withIndex()) {
            val a = handCoded.forward(tokenId).data.copyToFloatArray()
            val b = dsl.forward(tokenId).data.copyToFloatArray()
            require(a.size == b.size) { "step=$step: logit sizes differ: ${a.size} vs ${b.size}" }
            for (i in a.indices) {
                val d = abs(a[i] - b[i])
                maxDiff = max(maxDiff, d)
                assertTrue(
                    d < tolerance,
                    "step=$step logit[$i] diverged: hand=${a[i]} dsl=${b[i]} diff=$d (> $tolerance)"
                )
            }
        }
        println("Single-layer parity PASSED. Max |Δlogit| across ${tokens.size} steps: $maxDiff")
    }

    @Test
    fun `mixed global and sliding layers - Gemma4Runtime and gemmaNetwork agree within tolerance`() {
        val miniNLayers = 4
        val miniSlidingWindow = 3
        val miniMetadata = metadata.copy(
            blockCount = miniNLayers,
            slidingWindow = miniSlidingWindow,
            // Mix: 3 sliding then one global (Gemma 4's convention: last layer is always global).
            layerTypes = listOf("sliding_attention", "sliding_attention", "sliding_attention", "full_attention"),
            kvSharedLayers = 0, // Shared-KV parity deferred — see note below.
        )
        val config = Gemma4Config.fromMetadata(miniMetadata)

        val tokenEmbedding = randn(Shape(vocabSize, dim), seed = 100)
        val finalNorm = ones(Shape(dim))
        val lmHead = randn(Shape(vocabSize, dim), seed = 101)
        val layerWeights = (0 until miniNLayers).map { layer ->
            Gemma4LayerWeights(
                inputLayernorm = ones(Shape(dim)),
                wq = randn(Shape(nHeads * headDim, dim), seed = 200 + layer * 10 + 1),
                wk = randn(Shape(nKvHeads * headDim, dim), seed = 200 + layer * 10 + 2),
                wv = randn(Shape(nKvHeads * headDim, dim), seed = 200 + layer * 10 + 3),
                wo = randn(Shape(dim, nHeads * headDim), seed = 200 + layer * 10 + 4),
                postAttentionLayernorm = ones(Shape(dim)),
                gateProj = randn(Shape(ffnDim, dim), seed = 200 + layer * 10 + 5),
                upProj = randn(Shape(ffnDim, dim), seed = 200 + layer * 10 + 6),
                downProj = randn(Shape(dim, ffnDim), seed = 200 + layer * 10 + 7)
            )
        }

        val runtimeWeights = Gemma4RuntimeWeights(
            metadata = miniMetadata,
            tokenEmbedding = tokenEmbedding,
            ropeFreqReal = null,
            ropeFreqImag = null,
            layers = layerWeights,
            finalNorm = finalNorm,
            lmHead = lmHead
        )

        val dslTensors = linkedMapOf<String, Tensor<FP32, Float>>()
        dslTensors[Gemma4TensorNames.TOKEN_EMBEDDINGS] = tokenEmbedding
        dslTensors[Gemma4TensorNames.OUTPUT_NORM] = finalNorm
        dslTensors[Gemma4TensorNames.OUTPUT_WEIGHT] = lmHead
        layerWeights.forEachIndexed { layer, lw ->
            dslTensors[Gemma4TensorNames.inputLayernorm(layer)] = lw.inputLayernorm
            dslTensors[Gemma4TensorNames.attnQ(layer)] = lw.wq
            dslTensors[Gemma4TensorNames.attnK(layer)] = lw.wk
            dslTensors[Gemma4TensorNames.attnV(layer)] = lw.wv
            dslTensors[Gemma4TensorNames.attnOut(layer)] = lw.wo
            dslTensors[Gemma4TensorNames.postAttentionLayernorm(layer)] = lw.postAttentionLayernorm
            dslTensors[Gemma4TensorNames.ffnGate(layer)] = lw.gateProj
            dslTensors[Gemma4TensorNames.ffnUp(layer)] = lw.upProj
            dslTensors[Gemma4TensorNames.ffnDown(layer)] = lw.downProj
        }
        val dslWeights = Gemma4Weights<FP32, Float>(miniMetadata, dslTensors)

        // Path A: hand-coded
        val kvCache = HeapGemma4KvCache.fromConfig(config, seqLen)
        val backend = Gemma4AttentionBackend(ctx, runtimeWeights, FP32::class, config, kvCache)
        val handCoded = Gemma4Runtime(
            ctx = ctx,
            weights = runtimeWeights,
            attentionBackend = backend,
            dtype = FP32::class,
            config = config
        )

        // Path B: DSL
        val dslModel = GemmaNetworkLoader.fromWeights(ctx, dslWeights)
        val dsl = OptimizedLLMRuntime(
            model = dslModel,
            ctx = ctx,
            mode = OptimizedLLMMode.DIRECT,
            dtype = FP32::class
        )

        // Drive past the sliding-window boundary so both sliding and post-sliding
        // behaviour is exercised.
        val tokens = intArrayOf(1, 5, 3, 0, 2, 7, 4, 6)
        val tolerance = 1e-3f
        var maxDiff = 0f
        for ((step, tokenId) in tokens.withIndex()) {
            val a = handCoded.forward(tokenId).data.copyToFloatArray()
            val b = dsl.forward(tokenId).data.copyToFloatArray()
            for (i in a.indices) {
                val d = abs(a[i] - b[i])
                maxDiff = max(maxDiff, d)
                assertTrue(
                    d < tolerance,
                    "step=$step (window=$miniSlidingWindow) logit[$i] diverged: hand=${a[i]} dsl=${b[i]} diff=$d"
                )
            }
        }
        println("Mixed-layer parity PASSED. Max |Δlogit| across ${tokens.size} steps (window=$miniSlidingWindow): $maxDiff")
    }
}

/*
 * Known gap — shared-KV parity.
 *
 * Phase 5b's [SharedKVCache] forwards reads and writes to an owning
 * [AppendKVCache]. That's "append and share the accumulated history".
 *
 * The hand-coded [HeapGemma4KvCache] uses positional storage keyed by
 * (layer_slot, position); shared layers write to the same slot, so within a
 * single decode step the *last* shared layer's K/V overwrites the earlier
 * ones at that position. Across steps each slot accumulates whatever the
 * last layer wrote at that step.
 *
 * These semantics are different enough that closing parity requires a
 * positional KV cache in llm-core. That refactor belongs to a dedicated
 * follow-up; this parity test stays at `kvSharedLayers = 0` until then.
 */
