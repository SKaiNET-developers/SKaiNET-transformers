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
