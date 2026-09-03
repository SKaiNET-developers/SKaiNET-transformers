package sk.ainet.lang.nn.transformer

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.context.schedule.Schedule
import sk.ainet.lang.nn.transformer.schedule.AttentionSchedulePolicy
import sk.ainet.lang.nn.transformer.schedule.plan
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import java.util.concurrent.Executors
import java.util.concurrent.Future
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * SKEEP-005: the fused attention paths (in-place and copied K/V, decode and prefill) under a
 * shuffling multi-threaded schedule are bit-identical to the sequential run and to the general
 * `ops.scaledDotProductAttention` path, for every cache variant and GQA shape.
 */
class MultiHeadAttentionScheduleParityTest {

    /** Runs chunks on a pool, in shuffled order, and records that it was used. */
    private class ShufflingPoolSchedule(override val parallelism: Int) : Schedule {
        var regions = 0
        private val pool = Executors.newFixedThreadPool(parallelism)
        override val name: String get() = "shuffling($parallelism)"
        override fun forRange(n: Int, grain: Int, body: (Int, Int) -> Unit) {
            val tasks = Schedule.tasksFor(n, grain, parallelism)
            if (tasks == 0) return
            regions++
            val chunk = Schedule.chunkFor(n, tasks)
            val ranges = (0 until n step chunk).map { s -> s to minOf(s + chunk, n) }.shuffled()
            val futures: List<Future<*>> = ranges.map { (s, e) -> pool.submit { body(s, e) } }
            futures.forEach { it.get() }
        }
    }

    private val dim = 64

    private fun weights(ctx: ExecutionContext, out: Int, inDim: Int, seed: Int): Tensor<FP32, Float> =
        ctx.fromFloatArray(Shape(out, inDim), FP32::class, FloatArray(out * inDim) { i -> kotlin.math.sin((seed * 1000 + i).toFloat()) * 0.3f })

    private fun mha(ctx: ExecutionContext, nHeads: Int, nKVHeads: Int, cache: KVCache<FP32, Float>?, policy: AttentionSchedulePolicy, fused: Boolean = true): MultiHeadAttention<FP32, Float> {
        val headDim = dim / nHeads
        val m = MultiHeadAttention<FP32, Float>(dim = dim, nHeads = nHeads, nKVHeads = nKVHeads, causal = true, kvCache = cache, name = "attn")
        m.params[0].value = weights(ctx, nHeads * headDim, dim, 1)
        m.params[1].value = weights(ctx, nKVHeads * headDim, dim, 2)
        m.params[2].value = weights(ctx, nKVHeads * headDim, dim, 3)
        m.params[3].value = weights(ctx, dim, nHeads * headDim, 4)
        m.schedulePolicy = policy
        m.useFusedPaths = fused
        return m
    }

    private fun input(ctx: ExecutionContext, seq: Int, seed: Int): Tensor<FP32, Float> =
        ctx.fromFloatArray(Shape(seq, dim), FP32::class, FloatArray(seq * dim) { i -> ((i * 7 + seed * 13) % 17 - 8) / 8f })

    /** Prefill 17 tokens, then 6 single-token decode steps; returns every output concatenated. */
    private fun run(ctx: ExecutionContext, nHeads: Int, nKVHeads: Int, cacheKind: String, policy: AttentionSchedulePolicy, fused: Boolean = true): FloatArray {
        val headDim = dim / nHeads
        val cache: KVCache<FP32, Float>? = when (cacheKind) {
            "none" -> null
            "append" -> AppendKVCache(64, nKVHeads, headDim)
            "positional" -> PositionalKVCache(64, nKVHeads, headDim)
            else -> error(cacheKind)
        }
        val m = mha(ctx, nHeads, nKVHeads, cache, policy, fused)
        val outputs = mutableListOf<Float>()
        outputs += m.forward(input(ctx, 17, 1), ctx).data.copyToFloatArray().toList()
        if (cache != null) repeat(6) { step -> outputs += m.forward(input(ctx, 1, 10 + step), ctx).data.copyToFloatArray().toList() }
        return outputs.toFloatArray()
    }

    private val shapes = listOf(8 to 8, 8 to 2, 4 to 2)
    private val policies = listOf(AttentionSchedulePolicy.Sequential, AttentionSchedulePolicy.PerHead(minSeqKV = 1), AttentionSchedulePolicy.PerKVGroup(minSeqKV = 1), AttentionSchedulePolicy.Auto(minSeqKV = 1))

    @Test
    fun scheduledFusedAttentionMatchesSequentialBitForBit() {
        val sequential = DirectCpuExecutionContext(schedule = Schedule.Sequential)
        val shuffling = ShufflingPoolSchedule(parallelism = 4)
        val parallel = DirectCpuExecutionContext(schedule = shuffling)
        for ((nHeads, nKVHeads) in shapes) for (cacheKind in listOf("none", "append", "positional")) for (policy in policies) {
            val expected = run(sequential, nHeads, nKVHeads, cacheKind, AttentionSchedulePolicy.Sequential)
            val actual = run(parallel, nHeads, nKVHeads, cacheKind, policy)
            assertContentEquals(expected, actual, "heads=$nHeads kv=$nKVHeads cache=$cacheKind policy=$policy")
        }
        assertTrue(shuffling.regions > 0, "parallel policies must actually open regions")
    }

    @Test
    fun fusedPathsMatchTheGeneralSdpaPath() {
        val ctx = DirectCpuExecutionContext(schedule = Schedule.Sequential)
        val prefillFloats = 17 * dim
        for ((nHeads, nKVHeads) in shapes) for (cacheKind in listOf("none", "append", "positional")) {
            val general = run(ctx, nHeads, nKVHeads, cacheKind, AttentionSchedulePolicy.Sequential, fused = false)
            val fused = run(ctx, nHeads, nKVHeads, cacheKind, AttentionSchedulePolicy.Sequential, fused = true)
            // Prefill uses the engine SDPA's rounding order: bit-identical.
            assertContentEquals(general.copyOf(prefillFloats), fused.copyOf(prefillFloats), "prefill heads=$nHeads kv=$nKVHeads cache=$cacheKind")
            // Decode keeps the fused-decode order the golden gates were validated against
            // (Σ e·v then ·1/sum), which differs from SDPA (divide, then Σ) by rounding only.
            for (i in prefillFloats until general.size) {
                assertEquals(general[i], fused[i], 1e-6f, "decode heads=$nHeads kv=$nKVHeads cache=$cacheKind index=$i")
            }
        }
    }

    @Test
    fun inPlaceAndCopiedViewsAgree() {
        val ctx = DirectCpuExecutionContext(schedule = Schedule.Sequential)
        for ((nHeads, nKVHeads) in shapes) {
            val positional = run(ctx, nHeads, nKVHeads, "positional", AttentionSchedulePolicy.Sequential)
            val append = run(ctx, nHeads, nKVHeads, "append", AttentionSchedulePolicy.Sequential)
            assertContentEquals(append, positional, "heads=$nHeads kv=$nKVHeads")
        }
    }

    @Test
    fun slidingWindowLayersMatchTheGeneralPath() {
        val ctx = DirectCpuExecutionContext(schedule = ShufflingPoolSchedule(parallelism = 3))
        for ((window, right) in listOf(4 to 0, 6 to 2)) {
            fun runWindow(fused: Boolean): FloatArray {
                val m = MultiHeadAttention<FP32, Float>(dim = dim, nHeads = 4, nKVHeads = 2, causal = true, kvCache = SlidingWindowKVCache(64, 2, 16, window = window), slidingWindow = window, rightContext = right, name = "w")
                m.params[0].value = weights(ctx, 64, dim, 1); m.params[1].value = weights(ctx, 32, dim, 2)
                m.params[2].value = weights(ctx, 32, dim, 3); m.params[3].value = weights(ctx, dim, 64, 4)
                m.schedulePolicy = AttentionSchedulePolicy.PerHead(minSeqKV = 1)
                m.useFusedPaths = fused
                val out = mutableListOf<Float>()
                out += m.forward(input(ctx, 9, 1), ctx).data.copyToFloatArray().toList()
                repeat(3) { out += m.forward(input(ctx, 1, 20 + it), ctx).data.copyToFloatArray().toList() }
                return out.toFloatArray()
            }
            assertContentEquals(runWindow(fused = false), runWindow(fused = true), "window=$window right=$right")
        }
    }

    @Test
    fun planPicksGroupsWhenThereAreEnoughAndHeadsOtherwise() {
        val auto = AttentionSchedulePolicy.Auto(minSeqKV = 1)
        val llama3b = auto.plan(24, 8, 600, 12)!!
        assertEquals(24, llama3b.units); assertEquals(1, llama3b.headsPerUnit); assertEquals(2, llama3b.grain)
        val groups = auto.plan(32, 8, 600, 4)!!
        assertEquals(8, groups.units); assertEquals(4, groups.headsPerUnit); assertEquals(2, groups.grain)
        kotlin.test.assertNull(AttentionSchedulePolicy.Auto().plan(24, 8, 10, 12), "short context stays sequential")
        kotlin.test.assertNull(auto.plan(24, 8, 600, 1), "parallelism 1 stays sequential")
    }
}
