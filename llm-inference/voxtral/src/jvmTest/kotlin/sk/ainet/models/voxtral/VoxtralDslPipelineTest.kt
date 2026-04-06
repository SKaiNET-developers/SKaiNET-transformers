package sk.ainet.models.voxtral

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import sk.ainet.models.llama.LlamaModelMetadata
import sk.ainet.models.llama.LlamaTensorNames
import sk.ainet.models.llama.LlamaWeights

/**
 * Self-contained end-to-end test for the Voxtral backbone DSL pipeline:
 * VoxtralNetworkLoader → voxtralBackboneNetwork() → OptimizedLLMRuntime
 *
 * Uses a tiny model (dim=8, 1 layer, vocab=16) with deterministic
 * weights so no external model file is needed.
 *
 * The backbone is a standard LLaMA-architecture transformer, so these tests
 * validate the same forward-pass behavior as LLaMA/Qwen but through the
 * Voxtral entry points.
 */
class VoxtralDslPipelineTest {

    private val dim = 8
    private val ffDim = 16
    private val vocabSize = 16
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

    private val metadata = LlamaModelMetadata(
        architecture = "voxtral_tts",
        embeddingLength = dim,
        contextLength = seqLen,
        blockCount = 1,
        headCount = nHeads,
        kvHeadCount = kvHeads,
        feedForwardLength = ffDim,
        ropeDimensionCount = headDim,
        vocabSize = vocabSize
    )

    private fun buildWeightTensors(): Map<String, Tensor<FP32, Float>> = linkedMapOf(
        LlamaTensorNames.TOKEN_EMBEDDINGS to randn(Shape(vocabSize, dim), seed = 10),
        LlamaTensorNames.OUTPUT_NORM to ones(Shape(dim)),
        LlamaTensorNames.OUTPUT_WEIGHT to randn(Shape(vocabSize, dim), seed = 11),
        LlamaTensorNames.attnNorm(0) to ones(Shape(dim)),
        LlamaTensorNames.attnQ(0) to randn(Shape(dim, dim), seed = 1),
        LlamaTensorNames.attnK(0) to randn(Shape(dim, dim), seed = 2),
        LlamaTensorNames.attnV(0) to randn(Shape(dim, dim), seed = 3),
        LlamaTensorNames.attnOut(0) to randn(Shape(dim, dim), seed = 4),
        LlamaTensorNames.ffnNorm(0) to ones(Shape(dim)),
        LlamaTensorNames.ffnGate(0) to randn(Shape(ffDim, dim), seed = 5),
        LlamaTensorNames.ffnDown(0) to randn(Shape(dim, ffDim), seed = 6),
        LlamaTensorNames.ffnUp(0) to randn(Shape(ffDim, dim), seed = 7)
    )

    @Test
    fun `backbone DSL network definition builds correct module tree`() {
        val model = voxtralBackboneNetwork<FP32, Float>(metadata)

        val topLevelNames = model.modules.map { it.name }
        assertTrue("token_embd" in topLevelNames, "Should have token_embd module")
        assertTrue("blk.0" in topLevelNames, "Should have blk.0 module")
        assertTrue("output_norm" in topLevelNames, "Should have output_norm module")
        assertTrue("output" in topLevelNames, "Should have output module")
    }

    @Test
    fun `backbone weight loading maps all parameters`() {
        val tensors = buildWeightTensors()
        val weights = LlamaWeights<FP32, Float>(metadata, tensors)

        val model = VoxtralNetworkLoader.backboneFromWeights(weights)

        assertTrue(model.modules.isNotEmpty(), "Model should have child modules after weight loading")
    }

    @Test
    fun `backbone DIRECT mode forward produces finite logits with correct shape`() {
        val tensors = buildWeightTensors()
        val weights = LlamaWeights<FP32, Float>(metadata, tensors)
        val model = VoxtralNetworkLoader.backboneFromWeights(weights)

        val runtime = OptimizedLLMRuntime(
            model = model,
            ctx = ctx,
            mode = OptimizedLLMMode.DIRECT,
            dtype = FP32::class
        )

        val testTokens = intArrayOf(1, 5, 3)
        for (tokenId in testTokens) {
            val logits = runtime.forward(tokenId)

            assertEquals(vocabSize, logits.shape[logits.shape.rank - 1],
                "Last dim should be vocabSize for token $tokenId")

            val buf = logits.data.copyToFloatArray()
            for (i in buf.indices) {
                assertTrue(buf[i].isFinite(), "logit[$i] = ${buf[i]} is not finite for token $tokenId")
            }
        }
    }

    @Test
    fun `backbone DIRECT mode forward is deterministic`() {
        val tensors = buildWeightTensors()
        val weights = LlamaWeights<FP32, Float>(metadata, tensors)

        val model1 = VoxtralNetworkLoader.backboneFromWeights(weights)
        val runtime1 = OptimizedLLMRuntime(model1, ctx, OptimizedLLMMode.DIRECT, FP32::class)

        val model2 = VoxtralNetworkLoader.backboneFromWeights(weights)
        val runtime2 = OptimizedLLMRuntime(model2, ctx, OptimizedLLMMode.DIRECT, FP32::class)

        val logits1 = runtime1.forward(1).data.copyToFloatArray()
        val logits2 = runtime2.forward(1).data.copyToFloatArray()

        assertEquals(logits1.size, logits2.size)
        for (i in logits1.indices) {
            assertEquals(logits1[i], logits2[i], "Logit[$i] differs between runs")
        }
    }

    @Test
    fun `backbone generate produces valid token sequence`() {
        val tensors = buildWeightTensors()
        val weights = LlamaWeights<FP32, Float>(metadata, tensors)
        val model = VoxtralNetworkLoader.backboneFromWeights(weights)

        val runtime = OptimizedLLMRuntime(
            model = model,
            ctx = ctx,
            mode = OptimizedLLMMode.DIRECT,
            dtype = FP32::class
        )

        val generated = mutableListOf<Int>()
        runtime.generate(
            prompt = intArrayOf(1, 5),
            steps = 4,
            temperature = 1.0f
        ) { tokenId ->
            generated.add(tokenId)
        }

        assertEquals(4, generated.size, "Should generate exactly 4 tokens")
        for (tokenId in generated) {
            assertTrue(tokenId in 0 until vocabSize,
                "Generated token $tokenId should be in [0, $vocabSize)")
        }
    }

    @Test
    fun `backbone logits change with different input tokens`() {
        val tensors = buildWeightTensors()
        val weights = LlamaWeights<FP32, Float>(metadata, tensors)
        val model = VoxtralNetworkLoader.backboneFromWeights(weights)

        val runtime = OptimizedLLMRuntime(model, ctx, OptimizedLLMMode.DIRECT, FP32::class)

        val logits1 = runtime.forward(1).data.copyToFloatArray()
        val logits2 = runtime.forward(5).data.copyToFloatArray()

        var allSame = true
        for (i in logits1.indices) {
            if (abs(logits1[i] - logits2[i]) > 1e-6f) {
                allSame = false
                break
            }
        }
        assertTrue(!allSame, "Different tokens should produce different logits")
    }

    @Test
    fun `acoustic network definition builds correct module tree`() {
        val acousticMetadata = LlamaModelMetadata(
            architecture = "voxtral_tts_acoustic",
            embeddingLength = dim,
            contextLength = seqLen,
            blockCount = 2,
            headCount = nHeads,
            kvHeadCount = kvHeads,
            feedForwardLength = ffDim,
            ropeDimensionCount = headDim,
            vocabSize = vocabSize
        )

        val model = voxtralAcousticNetwork<FP32, Float>(acousticMetadata)

        val topLevelNames = model.modules.map { it.name }
        assertTrue("acoustic.blk.0" in topLevelNames, "Should have acoustic.blk.0 module")
        assertTrue("acoustic.blk.1" in topLevelNames, "Should have acoustic.blk.1 module")
        assertTrue("acoustic.output_norm" in topLevelNames, "Should have acoustic.output_norm module")
    }

    @Test
    fun `default metadata matches Voxtral-4B spec`() {
        val defaults = VoxtralDefaults.DEFAULT

        assertEquals(3072, defaults.backbone.embeddingLength)
        assertEquals(26, defaults.backbone.blockCount)
        assertEquals(32, defaults.backbone.headCount)
        assertEquals(8, defaults.backbone.kvHeadCount)
        assertEquals(9216, defaults.backbone.feedForwardLength)
        assertEquals(131072, defaults.backbone.vocabSize)
        assertEquals(128, defaults.backbone.ropeDimensionCount)

        assertEquals(3, defaults.acousticModel.blockCount)
        assertEquals(3072, defaults.acousticModel.embeddingLength)

        assertEquals(24000, defaults.codec.samplingRate)
        assertEquals(8192, defaults.codec.semanticCodebookSize)
        assertEquals(21, defaults.codec.acousticCodebookSize)
        assertEquals(36, defaults.audio.nAcousticCodebooks)
        assertEquals(37, defaults.audio.totalCodebooks)
    }

    // ========== Acoustic pipeline tests ==========

    private val acousticMetadataTiny = LlamaModelMetadata(
        architecture = "voxtral_tts_acoustic",
        embeddingLength = dim,
        contextLength = seqLen,
        blockCount = 1,
        headCount = nHeads,
        kvHeadCount = kvHeads,
        feedForwardLength = ffDim,
        ropeDimensionCount = headDim,
        vocabSize = vocabSize
    )

    private val tinyCodebooks = 4
    private val tinyLevels = 5
    private val tinyAcousticDim = tinyCodebooks * tinyLevels  // 20

    private fun buildAcousticWeightTensors(): Map<String, Tensor<FP32, Float>> {
        val tensors = linkedMapOf<String, Tensor<FP32, Float>>()

        // Acoustic transformer layer 0
        tensors[VoxtralTensorNames.acousticAttnNorm(0)] = ones(Shape(dim))
        tensors[VoxtralTensorNames.acousticAttnQ(0)] = randn(Shape(dim, dim), seed = 20)
        tensors[VoxtralTensorNames.acousticAttnK(0)] = randn(Shape(dim, dim), seed = 21)
        tensors[VoxtralTensorNames.acousticAttnV(0)] = randn(Shape(dim, dim), seed = 22)
        tensors[VoxtralTensorNames.acousticAttnOut(0)] = randn(Shape(dim, dim), seed = 23)
        tensors[VoxtralTensorNames.acousticFfnNorm(0)] = ones(Shape(dim))
        tensors[VoxtralTensorNames.acousticFfnGate(0)] = randn(Shape(ffDim, dim), seed = 24)
        tensors[VoxtralTensorNames.acousticFfnDown(0)] = randn(Shape(dim, ffDim), seed = 25)
        tensors[VoxtralTensorNames.acousticFfnUp(0)] = randn(Shape(ffDim, dim), seed = 26)
        tensors[VoxtralTensorNames.ACOUSTIC_NORM] = ones(Shape(dim))

        // Input/output projections
        tensors[VoxtralTensorNames.ACOUSTIC_INPUT_PROJ] = randn(Shape(dim, tinyAcousticDim), seed = 30)
        tensors[VoxtralTensorNames.ACOUSTIC_OUTPUT_PROJ] = randn(Shape(tinyAcousticDim, dim), seed = 31)

        return tensors
    }

    @Test
    fun `flow matching Euler produces finite output`() {
        val fm = VoxtralFlowMatching()
        val testSeqLen = 4
        val testDim = 8

        val result = fm.sampleEuler<FP32>(
            ctx = ctx,
            dtype = FP32::class,
            seqLen = testSeqLen,
            acousticDim = testDim,
            numSteps = 4,
            velocityFn = { xt, _ ->
                // Identity velocity: just return -xt (moves toward zero)
                ctx.ops.mulScalar(xt, -1.0f)
            },
            random = kotlin.random.Random(42)
        )

        assertEquals(2, result.shape.rank, "Output should be 2D")
        assertEquals(testSeqLen, result.shape[0], "First dim should be seqLen")
        assertEquals(testDim, result.shape[1], "Second dim should be acousticDim")

        val data = result.data.copyToFloatArray()
        for (i in data.indices) {
            assertTrue(data[i].isFinite(), "Output[$i] = ${data[i]} is not finite")
        }
    }

    @Test
    fun `flow matching midpoint produces finite output`() {
        val fm = VoxtralFlowMatching()

        val result = fm.sampleMidpoint<FP32>(
            ctx = ctx,
            dtype = FP32::class,
            seqLen = 4,
            acousticDim = 8,
            numSteps = 4,
            velocityFn = { xt, _ -> ctx.ops.mulScalar(xt, -1.0f) },
            random = kotlin.random.Random(42)
        )

        val data = result.data.copyToFloatArray()
        for (i in data.indices) {
            assertTrue(data[i].isFinite(), "Midpoint output[$i] = ${data[i]} is not finite")
        }
    }

    @Test
    fun `FSQ quantization produces valid codes`() {
        val fm = VoxtralFlowMatching()
        val nCb = 2
        val levels = 3
        // 1 frame, 2 codebooks × 3 levels = 6 values
        // cb0: [-1.0, 0.5, 2.0] → argmax = 2
        // cb1: [3.0, -1.0, 0.0] → argmax = 0
        val values = floatArrayOf(-1.0f, 0.5f, 2.0f, 3.0f, -1.0f, 0.0f)
        @Suppress("UNCHECKED_CAST")
        val tensor = ctx.fromFloatArray<FP32, Float>(Shape(1, values.size), FP32::class, values) as Tensor<FP32, Float>

        val codes = fm.quantizeFSQ(tensor, nCb, levels)

        assertEquals(nCb, codes.size, "Should produce nCodebooks codes per frame")
        for (code in codes) {
            assertTrue(code in 0 until levels,
                "Code $code should be in [0, $levels)")
        }
        assertEquals(2, codes[0], "First codebook argmax should be index 2")
        assertEquals(0, codes[1], "Second codebook argmax should be index 0")
    }

    @Test
    fun `acoustic runtime generates valid codes with tiny model`() {
        val tensors = buildAcousticWeightTensors()
        val weights = LlamaWeights<FP32, Float>(acousticMetadataTiny, tensors)

        val acousticRuntime = VoxtralNetworkLoader.acousticFromWeights(
            weights = weights,
            acousticMetadata = acousticMetadataTiny,
            ctx = ctx,
            nCodebooks = tinyCodebooks,
            codebookLevels = tinyLevels
        )

        // Simulate backbone hidden states: 3 frames of dim=8
        val backboneHidden = randn(Shape(3, dim), seed = 50)

        val codes = acousticRuntime.generate(
            backboneHidden = backboneHidden,
            numSteps = 4,
            method = "euler",
            random = kotlin.random.Random(42)
        )

        assertEquals(3 * tinyCodebooks, codes.size,
            "Should produce seqLen * nCodebooks codes")
        for (code in codes) {
            assertTrue(code in 0 until tinyLevels,
                "Code $code should be in [0, $tinyLevels)")
        }
    }

    @Test
    fun `acoustic runtime is deterministic with same seed`() {
        val tensors = buildAcousticWeightTensors()
        val weights = LlamaWeights<FP32, Float>(acousticMetadataTiny, tensors)
        val backboneHidden = randn(Shape(3, dim), seed = 50)

        val runtime1 = VoxtralNetworkLoader.acousticFromWeights(
            weights, acousticMetadataTiny, ctx, tinyCodebooks, tinyLevels
        )
        val codes1 = runtime1.generate(backboneHidden, numSteps = 4, random = kotlin.random.Random(42))

        val runtime2 = VoxtralNetworkLoader.acousticFromWeights(
            weights, acousticMetadataTiny, ctx, tinyCodebooks, tinyLevels
        )
        val codes2 = runtime2.generate(backboneHidden, numSteps = 4, random = kotlin.random.Random(42))

        assertTrue(codes1.contentEquals(codes2), "Same seed should produce identical codes")
    }

    @Test
    fun `acoustic runtime produces different codes for different conditioning`() {
        val tensors = buildAcousticWeightTensors()
        val weights = LlamaWeights<FP32, Float>(acousticMetadataTiny, tensors)

        val hidden1 = randn(Shape(3, dim), seed = 50)
        val hidden2 = randn(Shape(3, dim), seed = 99)

        val runtime1 = VoxtralNetworkLoader.acousticFromWeights(
            weights, acousticMetadataTiny, ctx, tinyCodebooks, tinyLevels
        )
        val codes1 = runtime1.generate(hidden1, numSteps = 4, random = kotlin.random.Random(42))

        val runtime2 = VoxtralNetworkLoader.acousticFromWeights(
            weights, acousticMetadataTiny, ctx, tinyCodebooks, tinyLevels
        )
        val codes2 = runtime2.generate(hidden2, numSteps = 4, random = kotlin.random.Random(42))

        var allSame = true
        for (i in codes1.indices) {
            if (codes1[i] != codes2[i]) { allSame = false; break }
        }
        assertTrue(!allSame, "Different conditioning should produce different codes")
    }

    @Test
    fun `config parser parses Mistral params json`() {
        val json = """
        {
            "dim": 3072,
            "n_layers": 26,
            "n_heads": 32,
            "n_kv_heads": 8,
            "hidden_dim": 9216,
            "vocab_size": 131072,
            "max_seq_len": 65536,
            "head_dim": 128,
            "rope_theta": 1000000.0,
            "norm_eps": 1e-5
        }
        """.trimIndent()

        val metadata = VoxtralConfigParser.parse(json)

        assertEquals(3072, metadata.backbone.embeddingLength)
        assertEquals(26, metadata.backbone.blockCount)
        assertEquals(32, metadata.backbone.headCount)
        assertEquals(8, metadata.backbone.kvHeadCount)
        assertEquals(9216, metadata.backbone.feedForwardLength)
        assertEquals(131072, metadata.backbone.vocabSize)
        assertEquals(65536, metadata.backbone.contextLength)
        assertEquals(128, metadata.backbone.ropeDimensionCount)
    }
}
