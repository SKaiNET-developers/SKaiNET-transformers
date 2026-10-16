package sk.ainet.lang.nn.transformer

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * A flattened padded batch through [MultiHeadAttention] equals the single-sequence forward on every
 * real row: padded keys are excluded by the attention op's key lengths, and the remaining modules
 * are row-wise. Covers the encoder shapes that batch (bidirectional with bias, sliding window with
 * right context and RoPE, GQA) plus a causal decoder prefill.
 */
class MultiHeadAttentionPaddedBatchTest {
    private val ctx: ExecutionContext = DirectCpuExecutionContext()
    private val dim = 64

    private fun weights(out: Int, inDim: Int, seed: Int): Tensor<FP32, Float> =
        ctx.fromFloatArray(Shape(out, inDim), FP32::class, FloatArray(out * inDim) { i -> kotlin.math.sin((seed * 1000 + i).toFloat()) * 0.3f })

    private fun vector(size: Int, seed: Int): Tensor<FP32, Float> =
        ctx.fromFloatArray(Shape(size), FP32::class, FloatArray(size) { i -> kotlin.math.cos((seed * 100 + i).toFloat()) * 0.1f })

    private fun mha(
        nHeads: Int,
        nKVHeads: Int = nHeads,
        causal: Boolean,
        bias: Boolean = false,
        slidingWindow: Int? = null,
        rightContext: Int = 0,
        rope: Boolean = false,
    ): MultiHeadAttention<FP32, Float> {
        val headDim = dim / nHeads
        val m = MultiHeadAttention<FP32, Float>(
            dim = dim, nHeads = nHeads, nKVHeads = nKVHeads, causal = causal, bias = bias,
            slidingWindow = slidingWindow, rightContext = rightContext, name = "attn", dtype = FP32::class,
        )
        var seed = 1
        for (p in m.params) {
            val shape = p.value.shape
            p.value = if (shape.rank == 2) weights(shape[0], shape[1], seed++) else vector(shape[0], seed++)
        }
        if (rope) m.rope = RoPE(headDim = headDim, maxSeqLen = 64, base = 10000f, mode = RoPEMode.SPLIT_HALF)
        return m
    }

    private fun rows(count: Int, seed: Int): FloatArray = FloatArray(count * dim) { i -> ((i * 7 + seed * 13) % 17 - 8) / 8f }

    private fun single(m: MultiHeadAttention<FP32, Float>, data: FloatArray, len: Int): FloatArray =
        m.forward(ctx.fromFloatArray(Shape(len, dim), FP32::class, data), ctx).data.copyToFloatArray()

    /** Max |batched − single| over the real rows of every sequence. */
    private fun maxBatchDiff(m: MultiHeadAttention<FP32, Float>, lengths: IntArray): Float {
        val seqLen = lengths.max()
        val inputs = lengths.mapIndexed { i, len -> rows(len, i + 1) }
        val flat = FloatArray(lengths.size * seqLen * dim)
        for ((i, data) in inputs.withIndex()) data.copyInto(flat, destinationOffset = i * seqLen * dim)
        val batch = PaddedBatch.of(ctx, seqLen, lengths)
        val batched = withPaddedBatch(m, batch) {
            m.forward(ctx.fromFloatArray(Shape(batch.rows, dim), FP32::class, flat), ctx)
        }.data.copyToFloatArray()
        assertNull(m.paddedBatch, "withPaddedBatch must clear the layout")
        var maxDiff = 0f
        for ((i, len) in lengths.withIndex()) {
            val expected = single(m, inputs[i], len)
            for (r in 0 until len) for (d in 0 until dim) {
                val diff = abs(batched[(i * seqLen + r) * dim + d] - expected[r * dim + d])
                if (diff > maxDiff) maxDiff = diff
            }
        }
        return maxDiff
    }

    private val lengths = intArrayOf(5, 3, 7, 1)

    @Test
    fun bidirectional_with_bias_matches_single_sequences() {
        val diff = maxBatchDiff(mha(nHeads = 8, causal = false, bias = true), lengths)
        assertTrue(diff < 1e-5f, "max |diff| = $diff")
    }

    @Test
    fun sliding_window_with_right_context_and_rope_matches_single_sequences() {
        // The Laya local layer: non-causal band of half-window 2 on both sides, RoPE from position 0.
        val diff = maxBatchDiff(mha(nHeads = 4, causal = false, slidingWindow = 3, rightContext = 2, rope = true), lengths)
        assertTrue(diff < 1e-5f, "max |diff| = $diff")
    }

    @Test
    fun causal_gqa_prefill_matches_single_sequences() {
        val diff = maxBatchDiff(mha(nHeads = 8, nKVHeads = 2, causal = true, rope = true), lengths)
        assertTrue(diff < 1e-5f, "max |diff| = $diff")
    }

    @Test
    fun a_batch_of_one_full_sequence_matches_the_single_pass() {
        val diff = maxBatchDiff(mha(nHeads = 8, causal = false), intArrayOf(6))
        assertTrue(diff < 1e-5f, "max |diff| = $diff")
    }

    @Test
    fun padded_rows_are_finite_and_padded_keys_do_not_leak() {
        val m = mha(nHeads = 4, causal = false)
        val seqLen = 6
        val real = rows(3, 1)
        // Two batches that differ only in the padding content must agree on the real rows.
        fun run(padValue: Float): FloatArray {
            val flat = FloatArray(seqLen * dim) { padValue }
            real.copyInto(flat)
            val batch = PaddedBatch.of(ctx, seqLen, intArrayOf(3))
            return withPaddedBatch(m, batch) { m.forward(ctx.fromFloatArray(Shape(seqLen, dim), FP32::class, flat), ctx) }.data.copyToFloatArray()
        }
        val a = run(0f)
        val b = run(50f)
        for (i in 0 until 3 * dim) assertEquals(a[i], b[i], 1e-6f, "padding content leaked into real row ${i / dim}")
        assertTrue(a.all { it.isFinite() } && b.all { it.isFinite() })
    }

    @Test
    fun layout_is_validated() {
        assertFailsWith<IllegalArgumentException> { PaddedBatch.of(ctx, seqLen = 4, lengths = intArrayOf(5)) }
        assertFailsWith<IllegalArgumentException> { PaddedBatch(batchSize = 2, seqLen = 4, lengths = intArrayOf(1), keyLengths = PaddedBatch.of(ctx, 4, intArrayOf(1)).keyLengths) }
        val m = mha(nHeads = 4, causal = false)
        val batch = PaddedBatch.of(ctx, 4, intArrayOf(2, 3))
        assertFailsWith<IllegalArgumentException> {
            withPaddedBatch(m, batch) { m.forward(ctx.fromFloatArray(Shape(5, dim), FP32::class, rows(5, 1)), ctx) }
        }
        val cached = mha(nHeads = 4, causal = true)
        cached.kvCache = AppendKVCache(16, 4, dim / 4)
        assertFailsWith<IllegalArgumentException> {
            withPaddedBatch(cached, batch) { cached.forward(ctx.fromFloatArray(Shape(8, dim), FP32::class, rows(8, 1)), ctx) }
        }
    }
}
