package sk.ainet.models.gemma

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.nn.transformer.AppendKVCache
import sk.ainet.lang.nn.transformer.PositionalKVCache
import sk.ainet.lang.nn.transformer.SharedKVCache
import sk.ainet.lang.nn.transformer.SharedPositionalKVCache
import sk.ainet.lang.nn.transformer.SlidingWindowKVCache
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

/**
 * Unit tests for the KVCache variants introduced for Phase 5b. Lives in the
 * gemma module because that's where the CPU backend + `DirectCpuExecutionContext`
 * are wired up as test dependencies; the types themselves live in llm-core.
 */
class KVCacheVariantsTest {

    private val ctx = DirectCpuExecutionContext()

    private val nKVHeads = 2
    private val headDim = 4
    private val maxSeqLen = 16

    /** Build a [nKVHeads, newLen, headDim] tensor whose entries are consecutive ints
     *  starting at [start], so test output is trivially legible. */
    private fun kv(newLen: Int, start: Int): Tensor<FP32, Float> {
        val total = nKVHeads * newLen * headDim
        val values = FloatArray(total) { (start + it).toFloat() }
        return ctx.fromFloatArray(Shape(nKVHeads, newLen, headDim), FP32::class, values)
    }

    @Test
    fun `AppendKVCache accumulates full history`() {
        val cache = AppendKVCache<FP32, Float>(maxSeqLen, nKVHeads, headDim)

        val (k1, _) = cache.update(kv(newLen = 2, start = 0), kv(newLen = 2, start = 0), ctx)
        assertEquals(2, k1.shape[1], "first update should have 2 seq positions")
        assertEquals(2, cache.position)

        val (k2, _) = cache.update(kv(newLen = 3, start = 100), kv(newLen = 3, start = 100), ctx)
        assertEquals(5, k2.shape[1], "second update should accumulate to 5 seq positions")
        assertEquals(5, cache.position)

        cache.reset()
        assertEquals(0, cache.position)
    }

    @Test
    fun `SlidingWindowKVCache trims to last N positions`() {
        val window = 3
        val cache = SlidingWindowKVCache<FP32, Float>(maxSeqLen, nKVHeads, headDim, window)

        cache.update(kv(newLen = 2, start = 0), kv(newLen = 2, start = 0), ctx)
        val (k2, _) = cache.update(kv(newLen = 2, start = 100), kv(newLen = 2, start = 100), ctx)

        // 4 appended so far, but trimmed to the last `window` = 3 positions.
        assertEquals(window, k2.shape[1], "sliding-window cache should keep only $window seq positions")
        // position counter still advances absolutely
        assertEquals(4, cache.position, "position counter tracks absolute steps, not buffer size")

        val (k3, _) = cache.update(kv(newLen = 5, start = 1000), kv(newLen = 5, start = 1000), ctx)
        assertEquals(window, k3.shape[1], "remains clamped at window after more updates")
        assertEquals(9, cache.position)
    }

    @Test
    fun `SharedKVCache writes through to its delegate`() {
        val owner = AppendKVCache<FP32, Float>(maxSeqLen, nKVHeads, headDim)
        val follower = SharedKVCache<FP32, Float>(owner)

        owner.update(kv(newLen = 2, start = 0), kv(newLen = 2, start = 0), ctx)
        assertEquals(2, follower.position, "follower.position mirrors delegate.position")

        val (fk, _) = follower.update(kv(newLen = 1, start = 100), kv(newLen = 1, start = 100), ctx)
        // Write through owner: both see 3 positions now.
        assertEquals(3, owner.position)
        assertEquals(3, follower.position)
        assertEquals(3, fk.shape[1], "returned keys reflect the full delegate state")
    }

    @Test
    fun `PositionalKVCache updates and returns full history tensor`() {
        val cache = PositionalKVCache<FP32, Float>(maxSeqLen, nKVHeads, headDim)

        val (k1, _) = cache.update(kv(newLen = 1, start = 0), kv(newLen = 1, start = 0), ctx)
        assertEquals(1, k1.shape[1])
        assertEquals(1, cache.position)

        val (k2, _) = cache.update(kv(newLen = 1, start = 50), kv(newLen = 1, start = 50), ctx)
        assertEquals(2, k2.shape[1])
        assertEquals(2, cache.position)

        cache.reset()
        assertEquals(0, cache.position)
    }

    @Test
    fun `SharedPositionalKVCache writes at follower position overwriting delegate slot`() {
        val owner = PositionalKVCache<FP32, Float>(maxSeqLen, nKVHeads, headDim)
        val follower = SharedPositionalKVCache<FP32, Float>(owner)

        // Step 0: owner writes at slot 0; then follower writes at slot 0 (overwrites).
        val (ownerK0, _) = owner.update(kv(newLen = 1, start = 0), kv(newLen = 1, start = 0), ctx)
        val (followerK0, _) = follower.update(kv(newLen = 1, start = 1000), kv(newLen = 1, start = 1000), ctx)
        assertEquals(1, owner.position)
        assertEquals(1, follower.position)
        assertEquals(1, ownerK0.shape[1])
        assertEquals(1, followerK0.shape[1])

        // Owner's returned tensor is a snapshot from before the follower wrote,
        // so it still reflects owner's data at slot 0.
        val ownerSlot0 = ownerK0.data.copyToFloatArray()[0]
        // Follower's returned tensor reflects its own data at slot 0.
        val followerSlot0 = followerK0.data.copyToFloatArray()[0]
        assertTrue(ownerSlot0 != followerSlot0, "owner and follower snapshots should differ: $ownerSlot0 vs $followerSlot0")

        // Step 1: owner writes at slot 1 — it should see the follower's data at slot 0
        // because the follower overwrote it in step 0 after the owner's own read.
        val (ownerK1, _) = owner.update(kv(newLen = 1, start = 500), kv(newLen = 1, start = 500), ctx)
        assertEquals(2, ownerK1.shape[1], "after step 1 owner cache should have 2 positions")
        val ownerHistoryAtStep1 = ownerK1.data.copyToFloatArray()
        // First slot in owner's step-1 snapshot must match follower's step-0 write.
        // (Compare the first float of each head's slot-0 position.)
        assertEquals(followerSlot0, ownerHistoryAtStep1[0], 1e-6f,
            "owner reads follower's slot-0 write in subsequent steps")
    }

    @Test
    fun `SharedKVCache reset is a no-op on the follower`() {
        val owner = AppendKVCache<FP32, Float>(maxSeqLen, nKVHeads, headDim)
        val follower = SharedKVCache<FP32, Float>(owner)

        owner.update(kv(newLen = 2, start = 0), kv(newLen = 2, start = 0), ctx)
        follower.reset()
        assertEquals(2, owner.position, "follower.reset() must not clear delegate storage")
        assertEquals(2, follower.position)

        owner.reset()
        assertEquals(0, owner.position)
        assertEquals(0, follower.position)
    }

}
