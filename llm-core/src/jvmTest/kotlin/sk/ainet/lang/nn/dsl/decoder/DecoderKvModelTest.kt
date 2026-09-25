package sk.ainet.lang.nn.dsl.decoder

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.weights.MappingConfig
import sk.ainet.io.weights.WeightMapper
import sk.ainet.io.weights.WeightTensor
import sk.ainet.lang.nn.transformer.RoPEMode
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

/**
 * Correctness gate for [DecoderKvModel] (SKaiNET-transformers#411, the qwen-kv-v1 KV-cache path):
 * builds a tiny GQA network with real weights via [decoderTransformerNetwork] directly (no
 * dependency on the `qwen` module — this is llm-core, one layer below it), then checks that
 * [DecoderKvModel.forwardPrefillAt] + [DecoderKvModel.forwardWithPast] reproduce the SAME logits
 * the ordinary eager [OptimizedLLMRuntime] forward produces for the same prompt, token by token.
 * The KV-cache path is a refactor of attention (chunked/incremental instead of one growing-cache
 * eager pass), so it must agree with the reference eager path exactly — GQA (nKVHeads < nHeads)
 * included, since that is the one axis this class changes vs. GemmaModel's version (no `expandKV`,
 * K/V stay at their native head count; see the class doc for why that is still correct).
 *
 * The chunk path ([DecoderKvModel.forwardPrefillWithPast]) needs a host-built RoPE-range table and
 * causal mask that only the export harness (SKaiNET-transformers#411 Q1.3, not written yet) and
 * the native runtime build today — it gets its correctness gate from `QwenExportDumpTest` /
 * `QwenVmfbParityTest` against the real compiled graph instead of a unit test here.
 */
class DecoderKvModelTest {
    private val dim = 8
    private val ffDim = 16
    private val vocabSize = 16
    private val nHeads = 4
    private val kvHeads = 2       // GQA: nHeads % kvHeads == 0, nHeads > kvHeads
    private val headDim = dim / nHeads
    private val seqLen = 32
    private val nLayers = 2

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

    private data class TestMeta(
        override val embeddingLength: Int,
        override val contextLength: Int,
        override val blockCount: Int,
        override val headCount: Int,
        override val kvHeadCount: Int,
        override val feedForwardLength: Int,
        override val ropeDimensionCount: Int?,
        override val vocabSize: Int,
        override val ropeFreqBase: Float = 1_000_000f,
        override val rmsNormEps: Float = 1e-6f,
        override val bosTokenId: Int = 1,
        override val eosTokenId: Int = 2,
    ) : DecoderModelMetadata

    private fun metadata() = TestMeta(
        embeddingLength = dim, contextLength = seqLen, blockCount = nLayers,
        headCount = nHeads, kvHeadCount = kvHeads, feedForwardLength = ffDim,
        ropeDimensionCount = headDim, vocabSize = vocabSize,
    )

    /** Qwen-shaped tensor names (attn q/k/v/o, ffn gate/up/down, both norms) — the generic
     *  `blk.N.*` GGUF naming [decoderTransformerNetwork] and [WeightMapper] expect. */
    private fun buildWeightTensors(): Map<String, Tensor<FP32, Float>> {
        val qDim = headDim * nHeads
        val kvDim = headDim * kvHeads
        val m = linkedMapOf(
            DecoderTensorNames.TOKEN_EMBEDDINGS to randn(Shape(vocabSize, dim), seed = 10),
            DecoderTensorNames.OUTPUT_NORM to ones(Shape(dim)),
            DecoderTensorNames.OUTPUT_WEIGHT to randn(Shape(vocabSize, dim), seed = 11),
        )
        for (l in 0 until nLayers) {
            val s = l * 100
            m[DecoderTensorNames.attnNorm(l)] = ones(Shape(dim))
            m[DecoderTensorNames.attnQ(l)] = randn(Shape(qDim, dim), seed = s + 1)
            m[DecoderTensorNames.attnK(l)] = randn(Shape(kvDim, dim), seed = s + 2)
            m[DecoderTensorNames.attnV(l)] = randn(Shape(kvDim, dim), seed = s + 3)
            m[DecoderTensorNames.attnOut(l)] = randn(Shape(dim, qDim), seed = s + 4)
            m[DecoderTensorNames.ffnNorm(l)] = ones(Shape(dim))
            m[DecoderTensorNames.ffnGate(l)] = randn(Shape(ffDim, dim), seed = s + 5)
            m[DecoderTensorNames.ffnDown(l)] = randn(Shape(dim, ffDim), seed = s + 6)
            m[DecoderTensorNames.ffnUp(l)] = randn(Shape(ffDim, dim), seed = s + 7)
        }
        return m
    }

    private fun buildModel(): sk.ainet.lang.nn.Module<FP32, Float> {
        val model = decoderTransformerNetwork<FP32, Float>(
            metadata(), qkNorm = false, attnBias = false, ropeMode = RoPEMode.SPLIT_HALF,
        )
        val tensors = buildWeightTensors()
        val weightTensors = tensors.map { (name, tensor) -> WeightTensor(name = name, shape = tensor.shape.dimensions.toList(), tensor = tensor) }
        val result = WeightMapper.applyWeights(model, weightTensors, MappingConfig())
        // Zero-initialized bias params with no matching tensor are expected to stay unmapped
        // when the architecture has no bias (this test's network: attnBias = false, no lm-head
        // bias) -- the same tolerance QwenNetworkLoader.applyWeightsToNetwork applies.
        val unmappedNonBias = result.missingParams.filter { !it.contains(".bias") }
        assertTrue(unmappedNonBias.isEmpty(), "unmapped non-bias params: $unmappedNonBias")
        return model
    }

    @Test fun kvCachePathMatchesEagerForwardTokenByToken() {
        // Two independent model instances from the SAME weights: DecoderKvModel.forwardPrefillAt
        // calls stripCaches() (mha.kvCache = null on every block, like GemmaModel's own
        // stripCaches), which would corrupt a SHARED model's live in-place eager cache mid-test.
        val eagerModel = buildModel()
        val kvModel = buildModel()
        val kv = DecoderKvModel<FP32, Float>(kvModel, FP32::class)

        // Reference: ordinary eager forward, one token at a time (OptimizedLLMRuntime owns the
        // in-place KV cache each block's own MultiHeadAttention.kvCache uses).
        val eagerRuntime = OptimizedLLMRuntime(eagerModel, ctx, OptimizedLLMMode.DIRECT, FP32::class)
        val prompt = intArrayOf(3, 7, 1, 5)
        val eagerLogits = prompt.map { eagerRuntime.forward(it).data.copyToFloatArray() }

        // DecoderKvModel path: prefill-at the whole prompt (selecting the LAST position) — its
        // logits are the same causal-uncached computation the eager path did for that last
        // position, so they must match eagerLogits.last() exactly.
        val promptTensor: Tensor<FP32, Float> = ctx.fromFloatArray(Shape(prompt.size), FP32::class, FloatArray(prompt.size) { prompt[it].toFloat() })
        val selectLast: Tensor<FP32, Float> = ctx.fromFloatArray(Shape(1, prompt.size), FP32::class, FloatArray(prompt.size) { if (it == prompt.size - 1) 1f else 0f })
        val prefill = kv.forwardPrefillAt(promptTensor, selectLast, ctx)
        assertEquals(vocabSize, prefill.logits.shape[prefill.logits.rank - 1])
        assertEquals(nLayers, prefill.selfK.size)
        assertLogitsClose(eagerLogits.last(), prefill.logits.data.copyToFloatArray(), "prefillAt (last prompt token)")

        // Continue decoding with forwardWithPast, comparing against further eager steps.
        var selfK = prefill.selfK
        var selfV = prefill.selfV
        var position = prompt.size
        val extra = intArrayOf(2, 9)
        for (tok in extra) {
            val eager = eagerRuntime.forward(tok).data.copyToFloatArray()
            val rope = kv.buildRopeCosSin(position, ctx)
            val tokenTensor: Tensor<FP32, Float> = ctx.fromFloatArray(Shape(1), FP32::class, floatArrayOf(tok.toFloat()))
            val step = kv.forwardWithPast(tokenTensor, rope, selfK, selfV, ctx)
            assertEquals(vocabSize, step.logits.shape[step.logits.rank - 1])
            assertLogitsClose(eager, step.logits.data.copyToFloatArray(), "forwardWithPast @ pos=$position")
            selfK = step.selfK; selfV = step.selfV
            position += 1
            // Cache grows by exactly one row per step (dim 2 = seq/rows).
            assertEquals(position, selfK[0].shape[2], "selfK cache length should track position")
            assertEquals(kvHeads, selfK[0].shape[1], "selfK should stay at nKVHeads (no expandKV)")
        }
    }

    @Test fun buildRopeCosSinPicksSplitHalfForQwenMode() {
        val model = buildModel()
        val kv = DecoderKvModel<FP32, Float>(model, FP32::class)
        val rope = kv.buildRopeCosSin(0, ctx)
        assertEquals(Shape(1, headDim), rope.cos.shape)
        assertEquals(Shape(1, headDim), rope.sin.shape)
    }

    private fun assertLogitsClose(expected: FloatArray, actual: FloatArray, label: String) {
        assertEquals(expected.size, actual.size, "$label: size mismatch")
        for (i in expected.indices) {
            val d = kotlin.math.abs(expected[i] - actual[i])
            assertTrue(d < 1e-4f, "$label: logit[$i] expected=${expected[i]} actual=${actual[i]} diff=$d")
        }
    }
}
