package sk.ainet.models.bert

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * [BertEncoderRuntime.encodeBatch] / [BertEncoderRuntime.forwardBatch]: a right-padded batch equals
 * the single-sequence passes on every real row, on the DIRECT and the OPTIMIZED (traced graph) path,
 * for mean and CLS pooling and with the dense projection. The lengths are a graph input, not a
 * constant: one compiled graph serves batches of the same shape with different lengths.
 */
class BertEncoderRuntimeBatchTest {
    private val ctx = DirectCpuExecutionContext()
    private val h = 8

    private fun runtime(
        mode: BertExecutionMode = BertExecutionMode.DIRECT,
        pooling: BertPooling = BertPooling.MEAN,
        projDim: Int? = null,
    ): BertEncoderRuntime<FP32> {
        val cfg = BertSyntheticFixtures.tinyConfig(h = h, layers = 2, projDim = projDim)
        return createBertEncoderRuntime(cfg, BertSyntheticFixtures.weightTensors(ctx, cfg, withProjection = projDim != null), ctx, mode = mode, pooling = pooling)
    }

    private val sequences = listOf(intArrayOf(1, 2, 3, 4, 5), intArrayOf(6, 7), intArrayOf(8, 9, 1, 2, 3, 4, 5, 6), intArrayOf(3))

    private fun padded(seqs: List<IntArray>, pad: Int = 0): Triple<IntArray, Int, IntArray> {
        val seqLen = seqs.maxOf { it.size }
        val ids = IntArray(seqs.size * seqLen) { pad }
        for ((i, s) in seqs.withIndex()) s.copyInto(ids, destinationOffset = i * seqLen)
        return Triple(ids, seqLen, IntArray(seqs.size) { seqs[it].size })
    }

    private fun maxDiff(a: FloatArray, b: FloatArray): Float {
        assertEquals(a.size, b.size)
        var m = 0f
        for (i in a.indices) m = maxOf(m, abs(a[i] - b[i]))
        return m
    }

    private fun checkEncodeParity(rt: BertEncoderRuntime<FP32>, tol: Float = 1e-5f) {
        val (ids, seqLen, lengths) = padded(sequences)
        val batched = rt.encodePadded(ids, seqLen, lengths)
        assertEquals(sequences.size, batched.size)
        for ((i, s) in sequences.withIndex()) {
            val single = rt.encode(s)
            val diff = maxDiff(single, batched[i])
            assertTrue(diff < tol, "sequence $i: batched vs single max |diff| = $diff")
        }
    }

    @Test
    fun direct_mean_pooling_matches_single_sequences() = checkEncodeParity(runtime())

    @Test
    fun direct_cls_pooling_matches_single_sequences() = checkEncodeParity(runtime(pooling = BertPooling.CLS))

    @Test
    fun direct_with_projection_matches_single_sequences() {
        val rt = runtime(projDim = 6)
        assertEquals(6, rt.dimensions)
        checkEncodeParity(rt)
    }

    @Test
    fun optimized_graph_matches_direct_and_takes_lengths_as_input() {
        val direct = runtime()
        val optimized = runtime(mode = BertExecutionMode.OPTIMIZED)
        val (ids, seqLen, lengths) = padded(sequences)
        val expected = direct.forwardBatch(ids, seqLen, lengths).data.copyToFloatArray()
        val actual = optimized.forwardBatch(ids, seqLen, lengths).data.copyToFloatArray()
        // Only real rows are compared; padded rows are unspecified on both paths.
        var diff = 0f
        for ((b, len) in lengths.withIndex()) for (r in 0 until len) for (d in 0 until h) {
            val i = (b * seqLen + r) * h + d
            diff = maxOf(diff, abs(expected[i] - actual[i]))
        }
        assertTrue(diff < 1e-5f, "OPTIMIZED vs DIRECT max |diff| = $diff")

        // Same batch shape, other lengths and ids: the compiled graph must read the new lengths.
        val other = listOf(intArrayOf(5, 4, 3, 2, 1, 9, 8, 7), intArrayOf(2), intArrayOf(1, 1, 1), intArrayOf(7, 7, 7, 7, 7, 7))
        val (ids2, seqLen2, lengths2) = padded(other)
        assertEquals(seqLen, seqLen2)
        val optimizedBatch = optimized.encodePadded(ids2, seqLen2, lengths2)
        for ((i, s) in other.withIndex()) {
            val d = maxDiff(direct.encode(s), optimizedBatch[i])
            assertTrue(d < 1e-5f, "second batch, sequence $i: max |diff| = $d")
        }
    }

    @Test
    fun padding_content_does_not_reach_real_rows() {
        val rt = runtime()
        val (ids, seqLen, lengths) = padded(sequences, pad = 0)
        val (ids9, _, _) = padded(sequences, pad = 9)
        val a = rt.encodePadded(ids, seqLen, lengths)
        val b = rt.encodePadded(ids9, seqLen, lengths)
        for (i in sequences.indices) assertTrue(maxDiff(a[i], b[i]) < 1e-6f, "pad id leaked into sequence $i")
    }

    @Test
    fun attention_mask_overload_derives_lengths_and_rejects_left_padding() {
        val rt = runtime()
        val (ids, seqLen, lengths) = padded(sequences)
        val mask = IntArray(ids.size) { i -> if (i % seqLen < lengths[i / seqLen]) 1 else 0 }
        val fromMask = rt.encodeBatch(ids, seqLen, attentionMask = mask)
        val fromLengths = rt.encodePadded(ids, seqLen, lengths)
        for (i in sequences.indices) assertEquals(0f, maxDiff(fromMask[i], fromLengths[i]))

        val leftPadded = mask.copyOf().also { it[0] = 0 }
        assertFailsWith<IllegalArgumentException> { rt.encodeBatch(ids, seqLen, attentionMask = leftPadded) }
        assertFailsWith<IllegalArgumentException> { rt.encodePadded(ids, seqLen, IntArray(sequences.size)) }
        assertFailsWith<IllegalArgumentException> { rt.forwardBatch(ids, seqLen + 1, lengths) }
    }

    @Test
    fun single_sequence_path_is_unaffected_after_a_batch() {
        val rt = runtime()
        val before = rt.encode(sequences[0])
        val (ids, seqLen, lengths) = padded(sequences)
        rt.encodePadded(ids, seqLen, lengths)
        val after = rt.encode(sequences[0])
        assertEquals(0f, maxDiff(before, after))
    }
}
