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
