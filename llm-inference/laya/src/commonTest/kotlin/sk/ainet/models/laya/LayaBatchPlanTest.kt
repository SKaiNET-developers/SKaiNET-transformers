package sk.ainet.models.laya

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** [LayaAgent.planBatches]: similar lengths share a pass, a long sequence never pads short ones up. */
class LayaBatchPlanTest {

    private fun plan(lengths: IntArray, budget: Int = 4096) = LayaAgent.planBatches(lengths, budget).map { it.toList() }

    @Test
    fun every_index_appears_once_and_batches_are_sorted_by_length() {
        val lengths = intArrayOf(300, 1200, 310, 40, 295, 1180, 50)
        val batches = plan(lengths)
        assertEquals(lengths.indices.toList(), batches.flatten().sorted())
        for (b in batches) assertTrue(b.zipWithNext().all { (x, y) -> lengths[x] <= lengths[y] })
        // Three length clusters, three batches: {40, 50}, {295, 300, 310}, {1180, 1200}.
        assertEquals(listOf(listOf(3, 6), listOf(4, 0, 2), listOf(5, 1)), batches)
    }

    @Test
    fun a_long_sequence_runs_alone_instead_of_padding_the_short_ones() {
        val batches = plan(intArrayOf(200, 210, 1500, 205))
        assertEquals(listOf(listOf(0, 3, 1), listOf(2)), batches)
    }

    @Test
    fun the_token_budget_bounds_a_batch() {
        val batches = plan(IntArray(10) { 500 }, budget = 2000)
        assertEquals(listOf(4, 4, 2), batches.map { it.size })
    }

    @Test
    fun padding_fraction_is_respected() {
        // 100 and 130 together: padding 30 on 230 real tokens (13%) is allowed; 100 and 200 is not (50%).
        assertEquals(1, plan(intArrayOf(100, 130)).size)
        assertEquals(2, plan(intArrayOf(100, 200)).size)
        assertEquals(1, LayaAgent.planBatches(intArrayOf(100, 200), 4096, maxPaddingFraction = 1.0).size)
    }

    @Test
    fun one_sequence_is_one_batch() {
        assertEquals(listOf(listOf(0)), plan(intArrayOf(77)))
        assertEquals(emptyList(), plan(IntArray(0)))
    }
}
