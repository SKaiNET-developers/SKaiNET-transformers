package sk.ainet.lang.nn.transformer

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Slice
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.slice
import sk.ainet.lang.types.DType

/**
 * Base class for KV cache variants used by [MultiHeadAttention].
 *
 * All variants share the same update/reset contract so the attention module
 * does not need to know which storage strategy is in play. Pick a subclass:
 *
 * - [AppendKVCache] — unbounded append-and-concat (default LLaMA/Gemma-global).
 * - [SlidingWindowKVCache] — drops entries older than `window` steps (Gemma
 *   local/sliding attention layers).
 * - [SharedKVCache] — delegates to another cache instance (Gemma 4 "last N
 *   layers share one KV slot").
 *
 * `is KVCache<*, *>` checks in runtime code continue to match every variant.
 *
 * @param maxSeqLen maximum sequence length the cache can hold (informational)
 * @param nKVHeads number of key-value heads
 * @param headDim dimension of each head
 * @param name module name
 */
public abstract class KVCache<T : DType, V>(
    public val maxSeqLen: Int,
    public val nKVHeads: Int,
    public val headDim: Int,
    override val name: String = "KVCache"
) : Module<T, V>() {

    override val modules: List<Module<T, V>> = emptyList()

    /**
     * Append new key/value entries and return the cache state the attention
     * step should use. Concrete variants decide whether the returned tensors
     * include all history, only a trailing window, or a shared view from
     * another layer.
     *
     * @param newKey new key tensor for this step [nKVHeads, newLen, headDim]
     * @param newValue new value tensor for this step [nKVHeads, newLen, headDim]
     * @param ctx execution context
     * @return pair of (keys, values) the attention op should read
     */
    public abstract fun update(
        newKey: Tensor<T, V>,
        newValue: Tensor<T, V>,
        ctx: ExecutionContext
    ): Pair<Tensor<T, V>, Tensor<T, V>>

    /** Reset the cache (e.g. at start of new sequence). */
    public abstract fun reset()

    /** Current number of cached positions. */
    public abstract val position: Int

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        // KVCache is not used via standard forward() — use update() instead.
        // This passthrough exists for Module tree traversal / tracing compatibility.
        return input
    }
}

/**
 * Default KV cache: appends every new K/V step and returns the full history.
 * Memory grows unbounded with sequence length (up to [maxSeqLen]).
 */
public class AppendKVCache<T : DType, V>(
    maxSeqLen: Int,
    nKVHeads: Int,
    headDim: Int,
    name: String = "AppendKVCache"
) : KVCache<T, V>(maxSeqLen, nKVHeads, headDim, name) {

    private var cachedKeys: Tensor<T, V>? = null
    private var cachedValues: Tensor<T, V>? = null
    private var cachePosition: Int = 0

    override fun update(
        newKey: Tensor<T, V>,
        newValue: Tensor<T, V>,
        ctx: ExecutionContext
    ): Pair<Tensor<T, V>, Tensor<T, V>> {
        val ops = ctx.ops
        val prevK = cachedKeys
        val prevV = cachedValues

        val fullK = if (prevK != null) ops.concat(listOf(prevK, newKey), dim = prevK.rank - 2) else newKey
        val fullV = if (prevV != null) ops.concat(listOf(prevV, newValue), dim = prevV.rank - 2) else newValue

        cachedKeys = fullK
        cachedValues = fullV
        cachePosition += newKey.shape[newKey.rank - 2]

        return fullK to fullV
    }

    override fun reset() {
        cachedKeys = null
        cachedValues = null
        cachePosition = 0
    }

    override val position: Int get() = cachePosition
}

/**
 * Sliding-window KV cache: after appending, truncates to the last [window]
 * positions along the sequence dimension. Used by decoder layers that attend
 * to a bounded window (e.g. Gemma 4 local-attention layers with window 512).
 *
 * The [position] counter still advances with every step so RoPE / causal
 * masking see absolute token positions, but the returned keys/values only
 * cover the last [window] of them.
 */
public class SlidingWindowKVCache<T : DType, V>(
    maxSeqLen: Int,
    nKVHeads: Int,
    headDim: Int,
    public val window: Int,
    name: String = "SlidingWindowKVCache"
) : KVCache<T, V>(maxSeqLen, nKVHeads, headDim, name) {

    init {
        require(window > 0) { "SlidingWindowKVCache: window must be > 0, got $window" }
    }

    private var cachedKeys: Tensor<T, V>? = null
    private var cachedValues: Tensor<T, V>? = null
    private var cachePosition: Int = 0

    override fun update(
        newKey: Tensor<T, V>,
        newValue: Tensor<T, V>,
        ctx: ExecutionContext
    ): Pair<Tensor<T, V>, Tensor<T, V>> {
        val ops = ctx.ops
        val prevK = cachedKeys
        val prevV = cachedValues

        val seqDim = newKey.rank - 2
        val joinedK = if (prevK != null) ops.concat(listOf(prevK, newKey), dim = seqDim) else newKey
        val joinedV = if (prevV != null) ops.concat(listOf(prevV, newValue), dim = seqDim) else newValue

        val currentLen = joinedK.shape[seqDim]
        val (trimmedK, trimmedV) = if (currentLen > window) {
            val start = currentLen - window
            // Use slice along seqDim: keep indices [start, currentLen).
            trimSeq(ops, joinedK, seqDim, start, currentLen) to trimSeq(ops, joinedV, seqDim, start, currentLen)
        } else {
            joinedK to joinedV
        }

        cachedKeys = trimmedK
        cachedValues = trimmedV
        cachePosition += newKey.shape[seqDim]

        return trimmedK to trimmedV
    }

    private fun trimSeq(
        ops: sk.ainet.lang.tensor.ops.TensorOps,
        t: Tensor<T, V>,
        seqDim: Int,
        start: Int,
        end: Int
    ): Tensor<T, V> {
        // Build a per-axis Slice.Range for the seq dim, Slice.All for the rest.
        val slices = (0 until t.rank).map { axis ->
            if (axis == seqDim) Slice.Range<T, V>(start, end)
            else Slice.All<T, V>()
        }
        return t.slice(slices)
    }

    override fun reset() {
        cachedKeys = null
        cachedValues = null
        cachePosition = 0
    }

    override val position: Int get() = cachePosition
}

/**
 * KV cache backed by a pre-allocated `[nKVHeads, maxSeqLen, headDim]` buffer.
 *
 * Each [update] writes the incoming K/V at the cache's own `position` counter,
 * then advances the counter. Reads return a fresh tensor of shape
 * `[nKVHeads, position, headDim]` copied from the buffer's used prefix.
 *
 * Unlike [AppendKVCache] (which concats into a growing tensor), this variant's
 * storage is addressable by absolute position. That lets several cache
 * instances share one buffer via [SharedPositionalKVCache], matching the
 * "last writer wins at this slot" semantics of Gemma 4's shared-KV layers.
 *
 * For layers that are *not* part of a shared-KV group, [PositionalKVCache]
 * produces the same observable output as [AppendKVCache] step for step — the
 * trade-off is up-front allocation of `maxSeqLen × nKVHeads × headDim`
 * floats vs the concat cost of the growing variant.
 */
public class PositionalKVCache<T : DType, V>(
    maxSeqLen: Int,
    nKVHeads: Int,
    headDim: Int,
    name: String = "PositionalKVCache"
) : KVCache<T, V>(maxSeqLen, nKVHeads, headDim, name) {

    internal val keyBuf: FloatArray = FloatArray(maxSeqLen * nKVHeads * headDim)
    internal val valueBuf: FloatArray = FloatArray(maxSeqLen * nKVHeads * headDim)
    private var pos: Int = 0

    override fun update(
        newKey: Tensor<T, V>,
        newValue: Tensor<T, V>,
        ctx: ExecutionContext
    ): Pair<Tensor<T, V>, Tensor<T, V>> {
        val newLen = newKey.shape[newKey.rank - 2]
        writeAt(pos, newKey, newValue)
        pos += newLen
        return currentView(ctx, newKey.dtype)
    }

    /**
     * Write [newKey] / [newValue] into the buffer starting at absolute position
     * [startPos], without advancing the cache's own position counter. Used by
     * [SharedPositionalKVCache] followers to overwrite the owner's slot at
     * their own step.
     */
    internal fun writeAt(startPos: Int, newKey: Tensor<T, V>, newValue: Tensor<T, V>) {
        val newLen = newKey.shape[newKey.rank - 2]
        require(startPos + newLen <= maxSeqLen) {
            "PositionalKVCache overflow: startPos=$startPos newLen=$newLen maxSeqLen=$maxSeqLen"
        }
        val kHeadDim = newKey.shape[newKey.rank - 1]
        val vHeadDim = newValue.shape[newValue.rank - 1]
        require(kHeadDim == headDim) {
            "$name: K tensor headDim=$kHeadDim does not match cache headDim=$headDim. " +
                "Shared caches on mixed-head_dim groups need a padded variant, not raw writeAt."
        }
        require(vHeadDim == headDim) {
            "$name: V tensor headDim=$vHeadDim does not match cache headDim=$headDim."
        }
        val kHeads = newKey.shape[newKey.rank - 3]
        require(kHeads == nKVHeads) {
            "$name: K tensor nKVHeads=$kHeads does not match cache nKVHeads=$nKVHeads."
        }
        val kData = newKey.data.copyToFloatArray()
        val vData = newValue.data.copyToFloatArray()
        // newKey layout: [nKVHeads, newLen, headDim]
        //  buffer layout: [nKVHeads, maxSeqLen, headDim]
        for (h in 0 until nKVHeads) {
            for (s in 0 until newLen) {
                val srcOff = (h * newLen + s) * headDim
                val dstOff = (h * maxSeqLen + startPos + s) * headDim
                kData.copyInto(keyBuf, dstOff, srcOff, srcOff + headDim)
                vData.copyInto(valueBuf, dstOff, srcOff, srcOff + headDim)
            }
        }
    }

    /** Build a `[nKVHeads, position, headDim]` tensor view of the current prefix. */
    internal fun currentView(
        ctx: ExecutionContext,
        dtype: kotlin.reflect.KClass<T>
    ): Pair<Tensor<T, V>, Tensor<T, V>> = sliceView(ctx, dtype, upToPos = pos, sliceHeadDim = headDim)

    /**
     * Build a `[nKVHeads, upToPos, sliceHeadDim]` view of the buffer. Lets a
     * [PaddedSharedPositionalKVCache] wrapper read back only its layer's
     * `headDim` slice from a buffer whose full `headDim` is the shared-group
     * max, and control how many positions to return without mutating the
     * cache's own position counter.
     */
    internal fun sliceView(
        ctx: ExecutionContext,
        dtype: kotlin.reflect.KClass<T>,
        upToPos: Int,
        sliceHeadDim: Int
    ): Pair<Tensor<T, V>, Tensor<T, V>> {
        require(upToPos in 0..maxSeqLen) {
            "$name.sliceView: upToPos=$upToPos out of [0, $maxSeqLen]"
        }
        require(sliceHeadDim in 1..headDim) {
            "$name.sliceView: sliceHeadDim=$sliceHeadDim out of [1, $headDim]"
        }
        val outSize = nKVHeads * upToPos * sliceHeadDim
        val kOut = FloatArray(outSize)
        val vOut = FloatArray(outSize)
        for (h in 0 until nKVHeads) {
            for (s in 0 until upToPos) {
                val srcOff = (h * maxSeqLen + s) * headDim
                val dstOff = (h * upToPos + s) * sliceHeadDim
                keyBuf.copyInto(kOut, dstOff, srcOff, srcOff + sliceHeadDim)
                valueBuf.copyInto(vOut, dstOff, srcOff, srcOff + sliceHeadDim)
            }
        }
        val shape = sk.ainet.lang.tensor.Shape(nKVHeads, upToPos, sliceHeadDim)
        val k: Tensor<T, V> = ctx.fromFloatArray<T, V>(shape, dtype, kOut)
        val v: Tensor<T, V> = ctx.fromFloatArray<T, V>(shape, dtype, vOut)
        return k to v
    }

    override fun reset() {
        pos = 0
    }

    override val position: Int get() = pos
}

/**
 * Gemma 4-style shared KV cache: writes through to a [PositionalKVCache]
 * delegate at the follower's own step, overwriting whatever was there before.
 * Multiple followers pointing at the same delegate all share its storage; the
 * LAST layer to write at a given step wins for that position, and reads
 * return the delegate's current state (same view every peer sees).
 *
 * Each follower tracks its own position counter so RoPE in
 * [MultiHeadAttention] sees absolute step count (identical for every layer
 * within a single decode step) rather than aliasing through the delegate.
 *
 * [reset] is a no-op on the owner's buffer — only the follower's step
 * counter resets, matching the "owner layer owns the storage lifecycle"
 * convention used by the hand-coded Gemma 4 reference.
 */
public class SharedPositionalKVCache<T : DType, V>(
    public val delegate: PositionalKVCache<T, V>,
    name: String = "SharedPositionalKVCache"
) : KVCache<T, V>(delegate.maxSeqLen, delegate.nKVHeads, delegate.headDim, name) {

    private var pos: Int = 0

    override fun update(
        newKey: Tensor<T, V>,
        newValue: Tensor<T, V>,
        ctx: ExecutionContext
    ): Pair<Tensor<T, V>, Tensor<T, V>> {
        val newLen = newKey.shape[newKey.rank - 2]
        delegate.writeAt(pos, newKey, newValue)
        pos += newLen
        return delegate.currentView(ctx, newKey.dtype)
    }

    override fun reset() {
        pos = 0
    }

    override val position: Int get() = pos
}

/**
 * KV cache that forwards reads and writes to another cache instance.
 *
 * Used by Gemma 4 where the last `kvSharedLayers` decoder layers share the
 * KV state of one owner layer — saving memory and compute for long-context
 * models. Every shared follower points at the same [delegate]; calling
 * [update] on any of them writes through the one underlying storage, so all
 * followers see the same cached history.
 *
 * [reset] is a no-op on a follower — the owner layer is responsible for
 * resetting the shared storage at the start of a sequence. This avoids
 * accidentally clearing a cache that other layers still read from.
 */
public class SharedKVCache<T : DType, V>(
    public val delegate: KVCache<T, V>,
    name: String = "SharedKVCache"
) : KVCache<T, V>(delegate.maxSeqLen, delegate.nKVHeads, delegate.headDim, name) {

    override fun update(
        newKey: Tensor<T, V>,
        newValue: Tensor<T, V>,
        ctx: ExecutionContext
    ): Pair<Tensor<T, V>, Tensor<T, V>> = delegate.update(newKey, newValue, ctx)

    override fun reset() {
        // Intentional no-op: the owner layer resets the delegate; followers
        // clearing it would race with other followers still attending to it.
    }

    override val position: Int get() = delegate.position
}

/**
 * Mixed-head_dim shared positional KV cache. Wraps a [PositionalKVCache]
 * whose `headDim` is the *maximum* head_dim across every layer in the
 * shared group (e.g. Gemma 4 E2B's last-20-layers group contains SLIDING
 * layers at head_dim=256 and GLOBAL layers at head_dim=512, so the delegate
 * is sized at 512).
 *
 * On write, the incoming K/V — produced at this layer's own [layerHeadDim]
 * — is zero-padded up to `delegate.headDim` before writing to the shared
 * slot. On read, the delegate's full `[nKVHeads, upToPos, delegate.headDim]`
 * view is sliced back down to `[nKVHeads, pos, layerHeadDim]` so the
 * attention kernel sees the shapes its projections are dimensioned for.
 *
 * Direct DSL analogue of `HeapGemma4KvCache.store` (pads to `maxKvDim`) and
 * `HeapGemma4KvCache.getKey` (reads `headDim` values from the padded slot)
 * in the hand-coded `Gemma4Runtime`. Position tracking follows
 * [SharedPositionalKVCache]: each wrapper carries its own counter and
 * advances in lockstep with its peers as layers process each token.
 *
 * [reset] resets this wrapper's position counter only — the delegate
 * buffer lifecycle is owned by the outer runtime (matches
 * [SharedPositionalKVCache] / [SharedKVCache]).
 */
public class PaddedSharedPositionalKVCache<T : DType, V>(
    public val delegate: PositionalKVCache<T, V>,
    public val layerHeadDim: Int,
    name: String = "PaddedSharedPositionalKVCache"
) : KVCache<T, V>(delegate.maxSeqLen, delegate.nKVHeads, layerHeadDim, name) {

    private var pos: Int = 0

    init {
        require(layerHeadDim in 1..delegate.headDim) {
            "$name: layerHeadDim ($layerHeadDim) must be in [1, ${delegate.headDim}]"
        }
    }

    override fun update(
        newKey: Tensor<T, V>,
        newValue: Tensor<T, V>,
        ctx: ExecutionContext
    ): Pair<Tensor<T, V>, Tensor<T, V>> {
        val newLen = newKey.shape[newKey.rank - 2]
        val paddedK = padHeadDim(newKey, delegate.headDim, ctx)
        val paddedV = padHeadDim(newValue, delegate.headDim, ctx)
        delegate.writeAt(pos, paddedK, paddedV)
        pos += newLen
        return delegate.sliceView(ctx, newKey.dtype, upToPos = pos, sliceHeadDim = layerHeadDim)
    }

    override fun reset() {
        pos = 0
    }

    override val position: Int get() = pos

    private fun padHeadDim(
        t: Tensor<T, V>,
        targetHeadDim: Int,
        ctx: ExecutionContext
    ): Tensor<T, V> {
        val srcHeadDim = t.shape[t.rank - 1]
        if (srcHeadDim == targetHeadDim) return t
        val nKV = t.shape[0]
        val seq = t.shape[1]
        val srcBuf = t.data.copyToFloatArray()
        val out = FloatArray(nKV * seq * targetHeadDim)
        for (h in 0 until nKV) {
            for (s in 0 until seq) {
                val srcOff = (h * seq + s) * srcHeadDim
                val dstOff = (h * seq + s) * targetHeadDim
                srcBuf.copyInto(out, dstOff, srcOff, srcOff + srcHeadDim)
                // remaining [srcHeadDim, targetHeadDim) stays zero
            }
        }
        return ctx.fromFloatArray<T, V>(
            sk.ainet.lang.tensor.Shape(nKV, seq, targetHeadDim),
            t.dtype,
            out
        )
    }
}
