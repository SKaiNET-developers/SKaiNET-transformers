package sk.ainet.lang.nn.transformer

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.MemorySegmentTensorDataFactory
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

/** SKEEP-005: `updateInPlace` views read exactly what the copying `update` returns, for every cache variant. */
class KVCacheInPlaceViewTest {
    private val ctx = DirectCpuExecutionContext()
    private val nKV = 2; private val headDim = 4

    private fun kv(ctx: ExecutionContext, len: Int, seed: Int): Tensor<FP32, Float> =
        ctx.fromFloatArray(Shape(nKV, len, headDim), FP32::class, FloatArray(nKV * len * headDim) { (it * 3 + seed) % 11 / 10f })

    private fun readView(v: KVBufferView, rowHeadDim: Int = v.headDim): Pair<FloatArray, FloatArray> {
        val k = FloatArray(nKV * v.length * rowHeadDim); val vals = FloatArray(k.size)
        for (g in 0 until nKV) for (s in 0 until v.length) for (d in 0 until rowHeadDim) {
            k[(g * v.length + s) * rowHeadDim + d] = v.keys[g * v.headStride + s * v.rowStride + d]
            vals[(g * v.length + s) * rowHeadDim + d] = v.values[g * v.headStride + s * v.rowStride + d]
        }
        return k to vals
    }

    private fun assertViewMatchesCopy(view: KVBufferView, copied: Pair<Tensor<FP32, Float>, Tensor<FP32, Float>>) {
        val (k, v) = readView(view)
        assertEquals(copied.first.data.copyToFloatArray().toList(), k.toList())
        assertEquals(copied.second.data.copyToFloatArray().toList(), v.toList())
    }

    @Test
    fun positionalViewMatchesCopiedPrefixAfterEachStep() {
        val a = PositionalKVCache<FP32, Float>(16, nKV, headDim)
        val b = PositionalKVCache<FP32, Float>(16, nKV, headDim)
        for ((len, seed) in listOf(5 to 1, 1 to 2, 3 to 3)) {
            val view = assertNotNull(a.updateInPlace(kv(ctx, len, seed), kv(ctx, len, seed + 50), ctx))
            val copied = b.update(kv(ctx, len, seed), kv(ctx, len, seed + 50), ctx)
            assertEquals(b.position, view.length); assertEquals(a.position, b.position)
            assertViewMatchesCopy(view, copied)
        }
    }

    @Test
    fun sharedPaddedAndOwnerReadOnlyWrappersAgreeWithTheirCopies() {
        val ownerA = PositionalKVCache<FP32, Float>(16, nKV, headDim); val ownerB = PositionalKVCache<FP32, Float>(16, nKV, headDim)
        ownerA.updateInPlace(kv(ctx, 4, 1), kv(ctx, 4, 2), ctx); ownerB.update(kv(ctx, 4, 1), kv(ctx, 4, 2), ctx)
        val sharedA = SharedPositionalKVCache(ownerA); val sharedB = SharedPositionalKVCache(ownerB)
        assertViewMatchesCopy(assertNotNull(sharedA.updateInPlace(kv(ctx, 4, 3), kv(ctx, 4, 4), ctx)), sharedB.update(kv(ctx, 4, 3), kv(ctx, 4, 4), ctx))
        val roA = OwnerReadOnlyKVCache(ownerA); val roB = OwnerReadOnlyKVCache(ownerB)
        assertViewMatchesCopy(assertNotNull(roA.updateInPlace(kv(ctx, 1, 9), kv(ctx, 1, 9), ctx)), roB.update(kv(ctx, 1, 9), kv(ctx, 1, 9), ctx))
        val wideA = PositionalKVCache<FP32, Float>(16, nKV, 8); val wideB = PositionalKVCache<FP32, Float>(16, nKV, 8)
        val padA = PaddedSharedPositionalKVCache(wideA, layerHeadDim = headDim); val padB = PaddedSharedPositionalKVCache(wideB, layerHeadDim = headDim)
        val view = assertNotNull(padA.updateInPlace(kv(ctx, 3, 5), kv(ctx, 3, 6), ctx))
        assertEquals(8, view.rowStride); assertEquals(headDim, view.headDim)
        assertViewMatchesCopy(view, padB.update(kv(ctx, 3, 5), kv(ctx, 3, 6), ctx))
        assertViewMatchesCopy(assertNotNull(SharedKVCache(ownerA).updateInPlace(kv(ctx, 1, 7), kv(ctx, 1, 8), ctx)), SharedKVCache(ownerB).update(kv(ctx, 1, 7), kv(ctx, 1, 8), ctx))
    }

    @Test
    fun appendCacheOffersAViewOnlyWhenItsHistoryIsHeapBacked() {
        val heap = AppendKVCache<FP32, Float>(16, nKV, headDim)
        assertNotNull(heap.updateInPlace(kv(ctx, 3, 1), kv(ctx, 3, 2), ctx), "dense heap factory: the history is a FloatArray")
        val segCtx = DirectCpuExecutionContext(tensorDataFactory = MemorySegmentTensorDataFactory())
        val seg = AppendKVCache<FP32, Float>(16, nKV, headDim)
        assertNull(seg.updateInPlace(kv(segCtx, 3, 1), kv(segCtx, 3, 2), segCtx), "segment-backed history: fall back to the copy")
        assertEquals(3, seg.position, "the fallback still advanced the cache")
    }

    @Test
    fun recordingContextsNeverGetAView() {
        val recording = object : ExecutionContext by ctx { override val isRecording: Boolean get() = true }
        assertNull(PositionalKVCache<FP32, Float>(16, nKV, headDim).updateInPlace(kv(ctx, 2, 1), kv(ctx, 2, 2), recording))
    }
}
