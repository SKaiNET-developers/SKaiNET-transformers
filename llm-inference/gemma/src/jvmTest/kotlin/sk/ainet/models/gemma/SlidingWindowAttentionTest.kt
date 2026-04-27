package sk.ainet.models.gemma

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertFails
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.transformer.AppendKVCache
import sk.ainet.lang.nn.transformer.MultiHeadAttention
import sk.ainet.lang.nn.transformer.RoPE
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

/**
 * Tests for sliding-window attention (Phase 5b pass 3). The window mask is
 * built in [MultiHeadAttention] and layered on top of SDPA as an additive
 * bias; the causal behaviour is subsumed, so `causal=true` + `slidingWindow`
 * combine cleanly.
 */
class SlidingWindowAttentionTest {

    private val ctx = DirectCpuExecutionContext()

    private val dim = 8
    private val nHeads = 2
    private val headDim = dim / nHeads
    private val seqLen = 6

    private fun ones(shape: Shape): Tensor<FP32, Float> {
        val values = FloatArray(shape.volume) { 1.0f }
        return ctx.fromFloatArray(shape, FP32::class, values)
    }

    private fun randn(shape: Shape, seed: Int): Tensor<FP32, Float> {
        val rng = kotlin.random.Random(seed)
        val values = FloatArray(shape.volume) { (rng.nextFloat() - 0.5f) * 0.2f }
        return ctx.fromFloatArray(shape, FP32::class, values)
    }

    @Suppress("UNCHECKED_CAST")
    private fun buildAttention(slidingWindow: Int?): MultiHeadAttention<FP32, Float> {
        val mha = MultiHeadAttention<FP32, Float>(
            dim = dim,
            nHeads = nHeads,
            nKVHeads = nHeads,
            causal = true,
            bias = false,
            name = "test_mha",
            rope = RoPE(headDim = headDim, maxSeqLen = seqLen),
            kvCache = AppendKVCache(seqLen, nHeads, headDim),
            slidingWindow = slidingWindow
        )
        // Params are indexed [qW, kW, vW, oW] when bias=false.
        val seeds = intArrayOf(1, 2, 3, 4)
        for (i in 0 until 4) {
            val p = mha.params[i] as ModuleParameter<FP32, Float>
            p.value = randn(Shape(dim, dim), seeds[i])
        }
        return mha
    }

    @Test
    fun `non-windowed and window-equal-to-seqLen produce identical output`() {
        val input = randn(Shape(seqLen, dim), seed = 42)

        val unwindowed = buildAttention(slidingWindow = null)
        val equivalent = buildAttention(slidingWindow = seqLen)

        val a = unwindowed.forward(input, ctx).data.copyToFloatArray()
        val b = equivalent.forward(input, ctx).data.copyToFloatArray()

        for (i in a.indices) {
            assertTrue(
                abs(a[i] - b[i]) < 1e-4f,
                "element $i: windowed(seqLen) should match non-windowed causal; got ${a[i]} vs ${b[i]}"
            )
        }
    }

    @Test
    fun `small window changes output vs full attention`() {
        val input = randn(Shape(seqLen, dim), seed = 7)

        val full = buildAttention(slidingWindow = null)
        val windowed = buildAttention(slidingWindow = 2)

        val a = full.forward(input, ctx).data.copyToFloatArray()
        val b = windowed.forward(input, ctx).data.copyToFloatArray()

        // For query positions beyond window+1, the two must disagree.
        var anyDiff = false
        for (i in a.indices) {
            if (abs(a[i] - b[i]) > 1e-4f) { anyDiff = true; break }
        }
        assertTrue(anyDiff, "window=2 must produce different output than full causal attention on seqLen=$seqLen")
    }

    @Test
    fun `slidingWindow must be positive`() {
        assertFails {
            MultiHeadAttention<FP32, Float>(
                dim = dim, nHeads = nHeads, nKVHeads = nHeads,
                causal = true, slidingWindow = 0
            )
        }
    }

    @Test
    fun `slidingWindow without causal is rejected`() {
        assertFails {
            MultiHeadAttention<FP32, Float>(
                dim = dim, nHeads = nHeads, nKVHeads = nHeads,
                causal = false, slidingWindow = 4
            )
        }
    }
}
