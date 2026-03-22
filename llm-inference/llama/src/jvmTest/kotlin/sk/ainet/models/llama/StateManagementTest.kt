package sk.ainet.models.llama

import kotlin.math.abs
import kotlin.test.Ignore
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.assertNotEquals
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

/**
 * Tests for OptimizedLLMRuntime state management.
 *
 * Uses a tiny model (dim=8, 1 layer, vocab=16) with deterministic weights —
 * no external model file needed.
 *
 * ## DIRECT mode — position tracking via RoPE + KV cache
 *
 * Position tracking works correctly: RoPE applies different rotations per position,
 * KV cache accumulates context. However, when the SAME token is fed at every position,
 * all V vectors are identical (V has no RoPE), so attention output is position-independent.
 * With DIFFERENT tokens, the position effect is clearly visible.
 *
 * ## OPTIMIZED mode — graph compilation limitations
 *
 * The compiled graph has two separate issues:
 * 1. **Weight resolution**: Graph produces near-zero outputs even at position 0,
 *    indicating weights are not being propagated through the graph correctly.
 * 2. **Mutable state**: KV cache and position counter are mutable state that cannot
 *    be captured in a static computation graph. The graph is a snapshot of one
 *    forward pass. See docs/ARCHITECTURE.md for design options.
 *
 * OPTIMIZED tests are @Ignore'd until the graph weight resolution issue is fixed.
 */
class StateManagementTest {

    // ---- Tiny model setup (no external files) ----

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
        val values = FloatArray(shape.volume) { (rng.nextFloat() - 0.5f) * 2.0f }
        return ctx.fromFloatArray(shape, FP32::class, values)
    }

    private fun ones(shape: Shape): Tensor<FP32, Float> {
        val values = FloatArray(shape.volume) { 1.0f }
        return ctx.fromFloatArray(shape, FP32::class, values)
    }

    private val metadata = LlamaModelMetadata(
        architecture = "llama",
        embeddingLength = dim,
        contextLength = seqLen,
        blockCount = 1,
        headCount = nHeads,
        kvHeadCount = kvHeads,
        feedForwardLength = ffDim,
        ropeDimensionCount = headDim,
        vocabSize = vocabSize
    )

    private fun buildWeights(): Map<String, Tensor<FP32, Float>> = linkedMapOf(
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

    private fun createDirectRuntime(): OptimizedLLMRuntime<FP32> {
        val model = LlamaNetworkLoader.fromWeights(LlamaWeights(metadata, buildWeights()))
        return OptimizedLLMRuntime(model, ctx, OptimizedLLMMode.DIRECT, FP32::class)
    }

    private fun createOptimizedRuntime(): OptimizedLLMRuntime<FP32> {
        val model = LlamaNetworkLoader.fromWeights(LlamaWeights(metadata, buildWeights()))
        return OptimizedLLMRuntime(model, ctx, OptimizedLLMMode.OPTIMIZED, FP32::class)
    }

    // ---- Helpers ----

    private fun maxAbsDiff(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size)
        var max = 0f
        for (i in a.indices) {
            val d = abs(a[i] - b[i])
            if (d > max) max = d
        }
        return max
    }

    private fun mismatchFraction(a: FloatArray, b: FloatArray, tol: Float = 1e-3f): Float {
        require(a.size == b.size)
        var count = 0
        for (i in a.indices) {
            if (abs(a[i] - b[i]) > tol) count++
        }
        return count.toFloat() / a.size
    }

    // =====================================================================
    // Issue 1: DIRECT mode position/state tracking
    // =====================================================================

    @Test
    fun `DIRECT - same token at different positions produces identical logits (V has no RoPE)`() {
        // Same token at different positions produces identical logits because:
        // - RoPE rotates Q and K but NOT V
        // - With the same token, all V vectors in the KV cache are identical
        // - Attention output = weighted_sum(identical_V_vectors) = V, regardless of weights
        // - Therefore the output is position-independent when all tokens are the same
        //
        // This is mathematically correct behavior. Position-dependence manifests
        // when DIFFERENT tokens produce different V vectors (tested in multi-step test).
        val runtime = createDirectRuntime()

        val logits0 = runtime.forward(1).data.copyToFloatArray()  // position 0
        val logits1 = runtime.forward(1).data.copyToFloatArray()  // position 1

        val diff = maxAbsDiff(logits0, logits1)
        println("DIRECT same-token pos0 vs pos1: maxDiff=$diff")
        println("  pos0 first 5: ${logits0.take(5)}")
        println("  pos1 first 5: ${logits1.take(5)}")

        // Same token → same V → logits should be nearly identical (within FP precision)
        assertTrue(diff < 1e-3f,
            "Same token at different positions should produce identical logits " +
            "(V has no RoPE), but maxDiff=$diff")
    }

    @Test
    fun `DIRECT - different tokens at successive positions should produce different logits`() {
        // With different tokens, V vectors differ, so attention output changes with position.
        // This is the primary test for RoPE + KV cache position tracking.
        val runtime = createDirectRuntime()

        val logits0 = runtime.forward(1).data.copyToFloatArray()  // token=1 at position 0
        val logits1 = runtime.forward(5).data.copyToFloatArray()  // token=5 at position 1

        val diff = maxAbsDiff(logits0, logits1)
        println("DIRECT token=1@pos0 vs token=5@pos1: maxDiff=$diff")
        println("  pos0 first 5: ${logits0.take(5)}")
        println("  pos1 first 5: ${logits1.take(5)}")

        assertTrue(diff > 0.01f,
            "Different tokens at successive positions should produce different logits " +
            "(maxDiff=$diff). RoPE/KVCache may not be functioning.")
    }

    @Test
    fun `DIRECT - different tokens should produce very different logits`() {
        // Two independent runtimes, each seeing one token at position 0.
        // Different input embeddings should produce different outputs.
        val runtime1 = createDirectRuntime()
        val runtime2 = createDirectRuntime()

        val logits1 = runtime1.forward(1).data.copyToFloatArray()
        val logits5 = runtime2.forward(5).data.copyToFloatArray()

        val diff = maxAbsDiff(logits1, logits5)
        println("DIRECT token=1 vs token=5 (fresh runtimes): maxDiff=$diff")
        println("  token=1 first 5: ${logits1.take(5)}")
        println("  token=5 first 5: ${logits5.take(5)}")

        assertTrue(diff > 0.01f,
            "Different tokens should produce different logits (maxDiff=$diff)")
    }

    @Test
    fun `DIRECT - multi-step logits should show increasing divergence from step 0`() {
        val runtime = createDirectRuntime()
        val allLogits = mutableListOf<FloatArray>()

        // Feed 4 tokens
        for (token in intArrayOf(1, 5, 3, 7)) {
            allLogits.add(runtime.forward(token).data.copyToFloatArray())
        }

        // Each step should differ more from step 0 as context accumulates
        println("DIRECT multi-step divergence from step 0:")
        for (i in 1 until allLogits.size) {
            val diff = maxAbsDiff(allLogits[0], allLogits[i])
            val mismatch = mismatchFraction(allLogits[0], allLogits[i])
            println("  step $i: maxDiff=$diff, mismatch=${mismatch * 100}%")
        }

        // At minimum, step 1 should differ from step 0
        val step1Diff = maxAbsDiff(allLogits[0], allLogits[1])
        assertTrue(step1Diff > 0.01f,
            "Step 1 logits should differ from step 0 (maxDiff=$step1Diff)")
    }

    @Test
    fun `DIRECT - reset should restore to initial state`() {
        val runtime = createDirectRuntime()

        // Forward a few tokens to advance state
        runtime.forward(1)
        runtime.forward(5)
        runtime.forward(3)

        // Reset
        runtime.reset()

        // Forward same first token — should match a fresh runtime
        val afterReset = runtime.forward(1).data.copyToFloatArray()

        val freshRuntime = createDirectRuntime()
        val fresh = freshRuntime.forward(1).data.copyToFloatArray()

        val diff = maxAbsDiff(afterReset, fresh)
        println("DIRECT reset vs fresh: maxDiff=$diff")

        assertTrue(diff < 1e-6f,
            "After reset, logits should match a fresh runtime (maxDiff=$diff)")
    }

    // =====================================================================
    // Issue 2: OPTIMIZED mode graph replay
    // =====================================================================

    @Test
    fun `OPTIMIZED - unoptimized graph matches DIRECT at position 0`() {
        val directRuntime = createDirectRuntime()
        val directLogits = directRuntime.forward(1).data.copyToFloatArray()

        val unoptRuntime = createOptimizedRuntime()
        unoptRuntime.compileUnoptimized()
        val unoptLogits = unoptRuntime.forward(1).data.copyToFloatArray()
        val diff = maxAbsDiff(directLogits, unoptLogits)
        println("  Unoptimized vs DIRECT: maxDiff=${"%.6f".format(diff)}")
        println("    DIRECT first 5:      ${directLogits.take(5)}")
        println("    UNOPTIMIZED first 5: ${unoptLogits.take(5)}")

        assertTrue(diff < 1e-3f,
            "Unoptimized graph should match DIRECT (maxDiff=$diff)")
    }

    @Test
    @Ignore("Optimization passes introduce divergence — fused op handlers need debugging")
    fun `OPTIMIZED - full pipeline matches DIRECT at position 0`() {
        val directRuntime = createDirectRuntime()
        val directLogits = directRuntime.forward(1).data.copyToFloatArray()

        val optRuntime = createOptimizedRuntime()
        optRuntime.compile()
        val optLogits = optRuntime.forward(1).data.copyToFloatArray()
        val diff = maxAbsDiff(directLogits, optLogits)
        val mismatch = mismatchFraction(directLogits, optLogits, tol = 1e-4f)
        println("  Optimized vs DIRECT: maxDiff=${"%.6f".format(diff)}, mismatch=${"%.1f".format(mismatch * 100)}%")

        assertTrue(mismatch < 0.05f,
            "Optimized graph should match DIRECT at pos 0 (mismatch=${mismatch * 100}%)")
    }

    @Test
    @Ignore("Graph weight resolution produces near-zero outputs — needs graph executor fix")
    fun `OPTIMIZED - second token should still match DIRECT mode`() {
        // This is the critical test. After position 0:
        // - DIRECT mode: KVCache has cached K/V from step 0, RoPE uses position=1
        // - OPTIMIZED mode: graph replays position-0 trace (bug)
        val directRuntime = createDirectRuntime()
        val optimizedRuntime = createOptimizedRuntime()
        optimizedRuntime.compile()

        // Advance both to position 1
        directRuntime.forward(1)
        optimizedRuntime.forward(1)

        val directLogits = directRuntime.forward(5).data.copyToFloatArray()
        val optimizedLogits = optimizedRuntime.forward(5).data.copyToFloatArray()

        val diff = maxAbsDiff(directLogits, optimizedLogits)
        val mismatch = mismatchFraction(directLogits, optimizedLogits)
        println("OPTIMIZED vs DIRECT at position 1: maxDiff=$diff, mismatch=${mismatch * 100}%")
        println("  DIRECT first 5:    ${directLogits.take(5)}")
        println("  OPTIMIZED first 5: ${optimizedLogits.take(5)}")

        // EXPECTED TO FAIL until state management is fixed.
        // If this passes, the graph properly handles position advancement.
        assertTrue(mismatch < 0.05f,
            "At position 1, OPTIMIZED should match DIRECT (mismatch=${mismatch * 100}%). " +
            "This fails because the traced graph doesn't advance position/KV cache.")
    }

    @Test
    @Ignore("Graph weight resolution produces near-zero outputs — needs graph executor fix")
    fun `OPTIMIZED - logits should not be identical across positions`() {
        val runtime = createOptimizedRuntime()
        runtime.compile()

        val logits0 = runtime.forward(1).data.copyToFloatArray()
        val logits1 = runtime.forward(5).data.copyToFloatArray()

        val diff = maxAbsDiff(logits0, logits1)
        println("OPTIMIZED pos0 vs pos1: maxDiff=$diff")
        println("  pos0 first 5: ${logits0.take(5)}")
        println("  pos1 first 5: ${logits1.take(5)}")

        // If the graph properly handles state, these should differ.
        // If they're nearly identical, the graph is replaying position-0.
        assertTrue(diff > 0.01f,
            "OPTIMIZED mode logits should change across positions (maxDiff=$diff). " +
            "Nearly identical logits indicate the graph replays the position-0 trace.")
    }
}
