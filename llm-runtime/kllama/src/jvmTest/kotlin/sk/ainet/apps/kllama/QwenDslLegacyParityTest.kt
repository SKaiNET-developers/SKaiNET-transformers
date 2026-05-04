package sk.ainet.apps.kllama

import java.lang.foreign.Arena
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertTrue
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import sk.ainet.models.llama.DecoderGgufWeights
import sk.ainet.models.llama.LlamaModelMetadata
import sk.ainet.models.llama.LlamaRuntime
import sk.ainet.models.llama.LlamaTensorNames
import sk.ainet.models.llama.LlamaWeightMapper
import sk.ainet.models.llama.MemSegWeightConverter
import sk.ainet.models.qwen.QwenNetworkLoader

/**
 * Pre-Phase-4 parity guard: DSL Qwen and legacy `LlamaRuntime` produce
 * bit-for-bit identical logits when fed equivalent inputs (the same
 * `DecoderGgufWeights`, same metadata, same RoPE base, same RMSNorm eps).
 * Closes #114.
 *
 * ## Setup detail (the gotcha that #114's earlier draft tripped over)
 *
 * `MemSegWeightConverter.convert` early-returns when `quantTypes` is empty
 * (`MemSegWeightConverter.kt:46`). In production, real Qwen3 GGUFs always
 * carry quantized tensors so the converter runs and pre-transposes FP32
 * tensors via `tensor.t()`. The legacy `LlamaRuntime.linearProject`
 * heuristic at `LlamaRuntime.kt:70-80` then correctly identifies the
 * pre-transposed `[in, out]` weight and skips runtime transpose.
 *
 * For an all-FP32 synthetic test the converter bypasses pre-transpose,
 * the heuristic misfires on square Q/O projections, and "DSL ↔ legacy"
 * logits diverge by ~50 — purely a test-setup artifact. Real GGUFs are
 * not affected. To prevent the same trap in the future this test injects
 * a dummy quant entry to force the converter to run.
 */
class QwenDslLegacyParityTest {

    private val ctx = DirectCpuExecutionContext()
    private val dim = 32
    private val nHeads = 2
    private val kvHeads = 2 // no GQA — keeps mapper shape constraints simple
    private val headDim = dim / nHeads
    private val ffnDim = 64
    private val vocabSize = 64
    private val seqLen = 16

    private val metadata = LlamaModelMetadata(
        architecture = "qwen3",
        embeddingLength = dim,
        contextLength = seqLen,
        blockCount = 1,
        headCount = nHeads,
        kvHeadCount = kvHeads,
        feedForwardLength = ffnDim,
        ropeDimensionCount = headDim,
        vocabSize = vocabSize,
        ropeFreqBase = 1_000_000f,
        rmsNormEps = 1e-6f,
    )

    @Test
    @Suppress("DEPRECATION") // intentionally exercises legacy LlamaRuntime
    fun `DSL Qwen and legacy LlamaRuntime produce identical logits on identical weights`() {
        val weights = buildWeights()

        // DSL path
        val dslModel = QwenNetworkLoader.fromWeights(weights)
        val dslRuntime = OptimizedLLMRuntime(dslModel, ctx, OptimizedLLMMode.DIRECT, FP32::class)

        // Legacy path — production CLI also runs MemSegWeightConverter.convert
        // after LlamaWeightMapper.map. The dummy quant entry in
        // `buildWeights()` ensures the converter actually runs (it
        // early-returns on empty quantTypes).
        Arena.ofConfined().use { arena ->
            val mapped = LlamaWeightMapper.map(weights)
            val converted = MemSegWeightConverter.convert(mapped, ctx, arena)
            val backend = CpuAttentionBackend(
                ctx, converted, FP32::class,
                ropeFreqBase = metadata.ropeFreqBase,
            )
            val legacy = LlamaRuntime(
                ctx, converted, backend, FP32::class,
                eps = metadata.rmsNormEps,
            )

            // Three tokens — exercises position 0 (empty cache), position 1
            // (1-element cache, RoPE non-trivial), position 2 (KV-cache
            // replay non-trivial).
            val tokens = intArrayOf(1, 5, 3)
            var maxDiff = 0f
            for ((step, tokenId) in tokens.withIndex()) {
                val dslLogits = dslRuntime.forward(tokenId).data.copyToFloatArray()
                val legacyLogits = legacy.forward(tokenId).data.copyToFloatArray()

                assertTrue(
                    dslLogits.size == legacyLogits.size,
                    "step=$step shape mismatch: dsl=${dslLogits.size} legacy=${legacyLogits.size}",
                )
                for (i in dslLogits.indices) {
                    val d = abs(dslLogits[i] - legacyLogits[i])
                    maxDiff = max(maxDiff, d)
                    assertTrue(
                        d < 1e-4f,
                        "step=$step logit[$i] divergence: dsl=${dslLogits[i]} " +
                            "legacy=${legacyLogits[i]} diff=$d",
                    )
                }
            }
            println(
                "Qwen DSL ↔ LlamaRuntime parity PASSED. " +
                    "Max |Δlogit| across ${tokens.size} tokens: $maxDiff",
            )
        }
    }

    private fun buildWeights(): DecoderGgufWeights<FP32, Float> {
        val tokenEmb = FloatArray(vocabSize * dim) { ((it % 5) - 2).toFloat() }
        val outputW = FloatArray(vocabSize * dim) { ((it % 5) - 2).toFloat() }
        val ones = FloatArray(dim) { 1f }
        val onesHead = FloatArray(headDim) { 1f }
        val q = FloatArray(dim * dim) { (((it * 3) % 7) - 3).toFloat() }
        val k = FloatArray(dim * dim) { (((it * 5) % 7) - 3).toFloat() }
        val v = FloatArray(dim * dim) { (((it * 7) % 7) - 3).toFloat() }
        val o = FloatArray(dim * dim) { (((it * 11) % 7) - 3).toFloat() }
        val gate = FloatArray(ffnDim * dim) { (((it * 3) % 5) - 2).toFloat() }
        val up = FloatArray(ffnDim * dim) { (((it * 5) % 5) - 2).toFloat() }
        val down = FloatArray(dim * ffnDim) { (((it * 7) % 5) - 2).toFloat() }

        // Dummy quant entry — name doesn't match any real tensor, so each
        // FP32 tensor's `quantTypes[name]` lookup returns null and goes
        // through the FP32 pre-transpose branch in
        // `MemSegWeightConverter.maybeConvert`. Forces the converter past
        // its `if (qt.isEmpty()) return weights` early-return.
        val dummyQuant = mapOf(
            "__force_converter_to_run__" to GGMLQuantizationType.Q4_0,
        )

        return DecoderGgufWeights(
            metadata = metadata,
            tensors = linkedMapOf(
                LlamaTensorNames.TOKEN_EMBEDDINGS to fp(Shape(vocabSize, dim), tokenEmb),
                LlamaTensorNames.OUTPUT_NORM to fp(Shape(dim), ones),
                LlamaTensorNames.OUTPUT_WEIGHT to fp(Shape(vocabSize, dim), outputW),
                LlamaTensorNames.attnNorm(0) to fp(Shape(dim), ones),
                LlamaTensorNames.attnQ(0) to fp(Shape(dim, dim), q),
                LlamaTensorNames.attnK(0) to fp(Shape(dim, dim), k),
                LlamaTensorNames.attnV(0) to fp(Shape(dim, dim), v),
                LlamaTensorNames.attnOut(0) to fp(Shape(dim, dim), o),
                // Per-head RMSNorm scales — trigger qkNorm on both runtimes.
                LlamaTensorNames.attnQNorm(0) to fp(Shape(headDim), onesHead),
                LlamaTensorNames.attnKNorm(0) to fp(Shape(headDim), onesHead),
                LlamaTensorNames.ffnNorm(0) to fp(Shape(dim), ones),
                LlamaTensorNames.ffnGate(0) to fp(Shape(ffnDim, dim), gate),
                LlamaTensorNames.ffnUp(0) to fp(Shape(ffnDim, dim), up),
                LlamaTensorNames.ffnDown(0) to fp(Shape(dim, ffnDim), down),
            ),
            quantTypes = dummyQuant,
        )
    }

    private fun fp(shape: Shape, values: FloatArray): Tensor<FP32, Float> =
        ctx.fromFloatArray(shape, FP32::class, values)
}
