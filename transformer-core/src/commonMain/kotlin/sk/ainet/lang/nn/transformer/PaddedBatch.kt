package sk.ainet.lang.nn.transformer

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.Int32

/**
 * A right-padded batch flattened into rows: row `r` of a `[batchSize * seqLen, dim]` activation holds
 * position `r % seqLen` of sequence `r / seqLen`, and sequence `b` has [lengths]`[b]` real tokens
 * followed by padding.
 *
 * Flattening keeps every row-wise module (projections, norms, feed-forward, residuals, embedding
 * lookups) byte-identical to the single-sequence path. Only attention and position lookups depend
 * on the layout; they implement [PaddedBatchAware] and read it for the duration of
 * [withPaddedBatch]. Attention passes [keyLengths] to the engine's `scaledDotProductAttention`, so
 * padded keys are excluded by the op's key-padding rule (recorded as `maskKind` metadata) instead
 * of a dense additive mask, and padded query rows produce finite but meaningless output that the
 * caller ignores (masked pooling, marker gathering).
 */
public class PaddedBatch(
    public val batchSize: Int,
    public val seqLen: Int,
    /** Real tokens per sequence, each in `0..seqLen`. */
    public val lengths: IntArray,
    /** [lengths] as the `[batchSize]` Int32 tensor the attention op takes. */
    public val keyLengths: Tensor<DType, *>,
) {
    init {
        require(batchSize > 0 && seqLen > 0) { "PaddedBatch: batchSize=$batchSize and seqLen=$seqLen must be positive" }
        require(lengths.size == batchSize) { "PaddedBatch: ${lengths.size} lengths for batchSize=$batchSize" }
        require(lengths.all { it in 0..seqLen }) { "PaddedBatch: every length must be in 0..$seqLen, got ${lengths.toList()}" }
    }

    /** Rows of the flattened activation, `batchSize * seqLen`. */
    public val rows: Int get() = batchSize * seqLen

    public companion object {
        /** Builds the layout and its Int32 lengths tensor on [ctx]. */
        public fun of(ctx: ExecutionContext, seqLen: Int, lengths: IntArray): PaddedBatch {
            @Suppress("UNCHECKED_CAST")
            val tensor = ctx.fromIntArray<Int32, Int>(Shape(lengths.size), Int32::class, lengths) as Tensor<DType, *>
            return PaddedBatch(lengths.size, seqLen, lengths, tensor)
        }
    }
}

/**
 * A module whose forward depends on the layout of a flattened padded batch. Module state, like a
 * KV cache: set it before the forward and clear it afterwards ([withPaddedBatch] does both), and
 * do not run the same module tree concurrently with different layouts.
 */
public interface PaddedBatchAware {
    public var paddedBatch: PaddedBatch?
}

/**
 * Sets [batch] on every [PaddedBatchAware] module under [root] for the duration of [block] and
 * clears it afterwards, so the next single-sequence forward sees no batch.
 */
public inline fun <R> withPaddedBatch(root: Module<*, *>, batch: PaddedBatch, block: () -> R): R {
    val aware = ArrayList<PaddedBatchAware>()
    collectPaddedBatchAware(root, aware)
    aware.forEach { it.paddedBatch = batch }
    try {
        return block()
    } finally {
        aware.forEach { it.paddedBatch = null }
    }
}

@PublishedApi
internal fun collectPaddedBatchAware(module: Module<*, *>, out: MutableList<PaddedBatchAware>) {
    if (module is PaddedBatchAware) out += module
    for (child in module.modules) collectPaddedBatchAware(child, out)
}
