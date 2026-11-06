package sk.ainet.lang.nn.dsl.decoder

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.transformer.MultiHeadAttention
import sk.ainet.lang.nn.transformer.RoPE
import sk.ainet.lang.nn.transformer.RoPEMode
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/*
 * Inputs and outputs of the KV-cache with-past graphs, shared by every decoder that has them
 * ([DecoderKvModel], `GemmaModel`). Generic over the model's dtype; a model with one RoPE base and
 * full attention everywhere (Llama, Qwen) sets only the global entries, Gemma 3 also the sliding
 * ones. The model routes per layer by `MultiHeadAttention.slidingWindow`.
 */

/** One RoPE base's sign-baked cos/sin tables `[rows, headDim]`, in that base's [RoPEMode] layout. */
public class RopeTables<T : DType, V>(
    public val cos: Tensor<T, V>,
    public val sin: Tensor<T, V>,
)

/**
 * RoPE tables per RoPE base, fed to a with-past graph as runtime inputs so the graph carries no
 * compile-time position. [sliding] is [global] for a model with one base.
 */
public class RopeCosSin<T : DType, V>(
    public val global: RopeTables<T, V>,
    public val sliding: RopeTables<T, V> = global,
) {
    public constructor(cos: Tensor<T, V>, sin: Tensor<T, V>) : this(RopeTables(cos, sin))

    public constructor(
        cosGlobal: Tensor<T, V>,
        sinGlobal: Tensor<T, V>,
        cosSliding: Tensor<T, V>,
        sinSliding: Tensor<T, V>,
    ) : this(RopeTables(cosGlobal, sinGlobal), RopeTables(cosSliding, sinSliding))

    /** Whether the sliding layers use their own base (a separate pair of graph inputs). */
    public val hasSlidingBase: Boolean get() = sliding !== global

    /** The tables [mha]'s layer rotates with. */
    public fun forLayer(mha: MultiHeadAttention<T, V>): RopeTables<T, V> = if (mha.slidingWindow == null) global else sliding
}

/**
 * Per-chunk inputs of a chunk prefill against a cache: RoPE tables `[C, headDim]` for the absolute
 * positions `past .. past+C-1`, and the additive attention masks `[1, M, C, past+C]` (0 = attend,
 * -1e30 = masked) per layer type. The masks carry everything position-dependent (the causal band,
 * the zero padding, a sliding window), so the graph itself stays position-agnostic.
 */
public class ChunkContext<T : DType, V>(
    public val rope: RopeCosSin<T, V>,
    public val maskGlobal: Tensor<T, V>,
    public val maskSliding: Tensor<T, V> = maskGlobal,
) {
    /** The mask [mha]'s layer attends with. */
    public fun maskFor(mha: MultiHeadAttention<T, V>): Tensor<T, V> = if (mha.slidingWindow == null) maskGlobal else maskSliding
}

/**
 * Logits plus the per-layer self K/V (`[1, nKVHeads, S, headDim]`) of a KV-cache graph: the
 * initial cache after a prefill, or the cache extended by this call's rows after a with-past step.
 */
public class KvStepOutput<T : DType, V>(
    public val logits: Tensor<T, V>,
    public val selfK: List<Tensor<T, V>>,
    public val selfV: List<Tensor<T, V>>,
)

/**
 * Build the RoPE tables for [rows] positions from [position] for the layers in [attention]: one
 * pair from a global layer's RoPE and, when the sliding layers have a RoPE of their own, one from a
 * sliding layer's. Each table follows its RoPE's [RoPEMode].
 */
public fun <T : DType, V> buildRopeCosSin(
    attention: List<MultiHeadAttention<T, V>>,
    position: Int,
    dtype: KClass<T>,
    ctx: ExecutionContext,
    rows: Int = 1,
): RopeCosSin<T, V> {
    val globalRope = attention.firstOrNull { it.slidingWindow == null }?.rope
    val slidingRope = attention.firstOrNull { it.slidingWindow != null }?.rope
    val first = globalRope ?: slidingRope ?: error("buildRopeCosSin: no attention layer has a RoPE")
    fun tables(rope: RoPE<T, V>): RopeTables<T, V> {
        val (c, s) = when (rope.mode) {
            RoPEMode.SPLIT_HALF -> rope.buildSplitHalfCosSin(position, rows)
            RoPEMode.INTERLEAVED -> rope.buildInterleavedCosSin(position, rows)
        }
        val shape = Shape(rows, rope.headDim)
        return RopeTables(ctx.fromFloatArray(shape, dtype, c), ctx.fromFloatArray(shape, dtype, s))
    }
    val global = tables(first)
    return if (slidingRope == null || slidingRope === first) RopeCosSin(global) else RopeCosSin(global, tables(slidingRope))
}
