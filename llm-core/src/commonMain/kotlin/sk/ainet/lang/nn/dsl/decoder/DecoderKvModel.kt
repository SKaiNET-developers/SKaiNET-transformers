package sk.ainet.lang.nn.dsl.decoder

import sk.ainet.apps.llm.HybridTransformerBlock
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.layers.EmbeddingAdapter
import sk.ainet.lang.nn.normalization.RMSNormalization
import sk.ainet.lang.nn.transformer.MultiHeadAttention
import sk.ainet.lang.nn.transformer.RoPEMode
import sk.ainet.lang.nn.transformer.SwiGLUFFN
import sk.ainet.lang.nn.transformer.VoidDense
import sk.ainet.lang.nn.transformer.linearProject
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.types.DType
import kotlin.math.sqrt
import kotlin.reflect.KClass

/**
 * Architecture-neutral KV-cache decode path over any module built by [decoderTransformerNetwork]
 * (Llama / Qwen / Mistral / ... — the plain `Embedding -> N x [RMSNorm, MHA, Residual, RMSNorm,
 * SwiGLU, Residual] -> RMSNorm -> Dense` shape: no sandwich norms, no PLE, one RoPE base, and
 * every layer unconditionally causal full attention — `decoderTransformerNetwork` never sets
 * `MultiHeadAttention.slidingWindow`).
 *
 * Mirrors the three-graph KV-cache contract [sk.ainet.models.gemma.GemmaModel]'s with-past
 * forwards pioneered (SKaiNET-transformers#411, the `qwen-kv-v1` contract):
 * [forwardPrefillAt] for the one-time catalog-prefix prefill, [forwardPrefillWithPast] for a
 * fixed-size chunk against an existing cache, [forwardWithPast] for one autoregressive step.
 * Differences from GemmaModel's version, all a consequence of a plain decoder-only architecture
 * instead of Gemma's:
 *  - one RoPE base / one attention mask per call — no sliding-vs-global split;
 *  - K/V stay at their native `nKVHeads` (no `expandKV` up to `nHeads`): SDPA is GQA-native
 *    (SKEEP-005 phase 2 — see `MultiHeadAttention.attentionImpl`'s own "Grouped-query attention
 *    is native to SDPA" comment, the exact same call shape this class uses), and the StableHLO
 *    exporter's `AttentionOperationsConverter` reshapes Q into the GQA group form
 *    `[b, nKV, nRep, Sq, hd]` — expanding K/V first would only add dead `narrow` + `concat` nodes
 *    to the trace. With a dynamic key length the mask must be head-shared `[b, 1, Sq, ?]` (see
 *    [ChunkContext]);
 *  - adds the Q/K/V/O projection bias when [MultiHeadAttention.bias] is set (Qwen2.5's `qwen2`
 *    architecture GGUFs carry real biases; Qwen3/Llama do not) — GemmaModel never needed this,
 *    Gemma has no attention bias;
 *  - no sandwich post-norms, no per-layer-embedding, no final-logit softcapping, no embedding
 *    scale (Gemma-only: `embed_scale = hidden_size**0.5`; Qwen/Llama embeddings are unscaled).
 */
public class DecoderKvModel<T : DType, V>(
    public val module: Module<T, V>,
    public val dtype: KClass<T>,
) {
    private val tokenEmbedding: EmbeddingAdapter<T, V>
    private val blocks: List<HybridTransformerBlock<T, V>>
    private val outputNorm: RMSNormalization<T, V>
    private val lmHead: VoidDense<T, V>

    init {
        val top = module.modules
        @Suppress("UNCHECKED_CAST")
        tokenEmbedding = top.filterIsInstance<EmbeddingAdapter<T, V>>().singleOrNull()
            ?: error("DecoderKvModel: expected exactly one top-level EmbeddingAdapter (token_embd)")
        @Suppress("UNCHECKED_CAST")
        blocks = top.filterIsInstance<HybridTransformerBlock<T, V>>()
        require(blocks.isNotEmpty()) { "DecoderKvModel: no HybridTransformerBlock found at the top level" }
        @Suppress("UNCHECKED_CAST")
        outputNorm = top.filterIsInstance<RMSNormalization<T, V>>().singleOrNull()
            ?: error("DecoderKvModel: expected exactly one top-level RMSNormalization (output_norm)")
        @Suppress("UNCHECKED_CAST")
        lmHead = top.filterIsInstance<VoidDense<T, V>>().singleOrNull()
            ?: error("DecoderKvModel: expected exactly one top-level VoidDense (lm head / output)")
    }

    /** logits `[seq, vocab]` + per-layer initial self K/V (`[1, nKVHeads, seq, headDim]`). */
    public class PrefillOutput<T : DType, V>(
        public val logits: Tensor<T, V>,
        public val selfK: List<Tensor<T, V>>,
        public val selfV: List<Tensor<T, V>>,
    )

    /** logits `[1, vocab]` + per-layer extended self K/V (one call's worth of new rows appended). */
    public class WithPastOutput<T : DType, V>(
        public val logits: Tensor<T, V>,
        public val selfK: List<Tensor<T, V>>,
        public val selfV: List<Tensor<T, V>>,
    )

    /** Per-step RoPE cos/sin (`[1, headDim]`) — the model's one RoPE base (see [buildRopeCosSin]). */
    public class RopeCosSin<T : DType, V>(
        public val cos: Tensor<T, V>,
        public val sin: Tensor<T, V>,
    )

    /**
     * [forwardPrefillWithPast]'s per-chunk inputs: RoPE cos/sin tables `[C, headDim]` for absolute
     * positions `past .. past+C-1` (split-half or interleaved layout per the model's [RoPEMode]),
     * and the additive causal+padding mask `[1, M, C, past+C]` (0 = attend, -1e30 = masked), where
     * `M` is 1 (head-shared) or nHeads; [forwardWithPast] never needs a mask (a single query attends
     * the whole, already-causal cache). Under GQA with a dynamic `past` use `M = 1`: SKaiNET 0.57.0
     * broadcasts a head-shared mask with the `dynamic_broadcast_in_dim` form IREE 3.11 lowers, while
     * a per-head one needs `dynamic_reshape`, which IREE 3.11 does not lower (SKaiNET#1302).
     */
    public class ChunkContext<T : DType, V>(
        public val cos: Tensor<T, V>,
        public val sin: Tensor<T, V>,
        public val mask: Tensor<T, V>,
    )

    private class BlockRefs<T : DType, V>(
        val attnNorm: RMSNormalization<T, V>,
        val mha: MultiHeadAttention<T, V>,
        val ffnNorm: RMSNormalization<T, V>,
        val ffn: SwiGLUFFN<T, V>,
    )

    @Suppress("UNCHECKED_CAST")
    private fun refsFor(block: HybridTransformerBlock<T, V>): BlockRefs<T, V> {
        val mods = block.modules
        val norms = mods.filterIsInstance<RMSNormalization<T, V>>()
        val mha = mods.filterIsInstance<MultiHeadAttention<T, V>>().firstOrNull()
            ?: error("DecoderKvModel: block ${block.name} has no MultiHeadAttention")
        val ffn = mods.filterIsInstance<SwiGLUFFN<T, V>>().firstOrNull()
            ?: error("DecoderKvModel: block ${block.name} has no SwiGLUFFN")
        require(norms.size == 2) {
            "DecoderKvModel: block ${block.name} expected exactly 2 RMSNorm (attn_norm, ffn_norm), got ${norms.size} " +
                "— a BitNet (attn_sub_norm) or sandwich-norm architecture needs its own KV-cache path, not this one"
        }
        return BlockRefs(attnNorm = norms[0], mha = mha, ffnNorm = norms[1], ffn = ffn)
    }

    /** Param indices [q, k, v, o] into MHA.params — bias shifts them, mirrors MultiHeadAttention's own qWIdx/kWIdx/vWIdx/oWIdx. */
    private fun paramIdx(mha: MultiHeadAttention<T, V>): IntArray =
        if (mha.bias) intArrayOf(0, 2, 4, 6) else intArrayOf(0, 1, 2, 3)

    /** `x @ W^T`, plus the paired bias at `idx+1` when [MultiHeadAttention.bias] is set. */
    private fun proj(ops: TensorOps, mha: MultiHeadAttention<T, V>, x: Tensor<T, V>, idx: Int): Tensor<T, V> {
        var y = linearProject(ops, x, mha.params[idx].value)
        if (mha.bias) y = ops.add(y, mha.params[idx + 1].value)
        return y
    }

    /** Project a single-token input to heads-first `[heads, 1, headDim]`. */
    private fun projHeadsOne(ops: TensorOps, mha: MultiHeadAttention<T, V>, x: Tensor<T, V>, idx: Int, heads: Int): Tensor<T, V> =
        ops.reshape(proj(ops, mha, x, idx), Shape(heads, 1, mha.headDim))

    // ==================== with-past: one new token ====================

    /**
     * KV-cache DECODE step: one new [tokenId] (`[1]` or `[1,1]`) at the position encoded by [rope]
     * (see [buildRopeCosSin]); the incoming per-layer self cache ([selfKIn]/[selfVIn],
     * `[1, nKVHeads, past, headDim]`) → logits `[1, vocab]` + the per-layer extended self cache.
     */
    public fun forwardWithPast(
        tokenId: Tensor<T, V>,
        rope: RopeCosSin<T, V>,
        selfKIn: List<Tensor<T, V>>,
        selfVIn: List<Tensor<T, V>>,
        ctx: ExecutionContext,
    ): WithPastOutput<T, V> {
        val ops = ctx.ops
        var h = tokenEmbedding.forward(tokenId, ctx)
        val nsk = ArrayList<Tensor<T, V>>(blocks.size)
        val nsv = ArrayList<Tensor<T, V>>(blocks.size)
        for ((i, block) in blocks.withIndex()) {
            val r = refsFor(block)
            val sn = r.attnNorm.forward(h, ctx)
            val (attnOut, fullK, fullV) = attnWithPast(r.mha, sn, rope.cos, rope.sin, selfKIn[i], selfVIn[i], ctx)
            val h1 = ops.add(h, attnOut)
            val ffnOut = r.ffn.forward(r.ffnNorm.forward(h1, ctx), ctx)
            h = ops.add(h1, ffnOut)
            nsk += ops.reshape(fullK, fullK.shape)   // identity → distinct graph output node
            nsv += ops.reshape(fullV, fullV.shape)
        }
        val logits = lmHead.forward(outputNorm.forward(h, ctx), ctx)
        return WithPastOutput(logits, nsk, nsv)
    }

    private fun attnWithPast(
        mha: MultiHeadAttention<T, V>,
        sn: Tensor<T, V>,
        cos: Tensor<T, V>, sin: Tensor<T, V>,
        pastK: Tensor<T, V>, pastV: Tensor<T, V>,
        ctx: ExecutionContext,
    ): Triple<Tensor<T, V>, Tensor<T, V>, Tensor<T, V>> {
        val ops = ctx.ops
        val rope = mha.rope ?: error("DecoderKvModel.attnWithPast: MHA has no RoPE")
        val idx = paramIdx(mha)
        var q = projHeadsOne(ops, mha, sn, idx[0], mha.nHeads)     // [nHeads, 1, headDim]
        var k = projHeadsOne(ops, mha, sn, idx[1], mha.nKVHeads)   // [nKVHeads, 1, headDim]
        val v = projHeadsOne(ops, mha, sn, idx[2], mha.nKVHeads)
        val qn = mha.qNorm; val kn = mha.kNorm
        if (qn != null && kn != null) { q = qn.forward(q, ctx); k = kn.forward(k, ctx) }  // qkNorm BEFORE RoPE
        q = rope.forwardWithCosSin(q, cos, sin, ctx)
        k = rope.forwardWithCosSin(k, cos, sin, ctx)
        val fullK = ops.concat(listOf(pastK, ops.unsqueeze(k, 0)), dim = 2)   // [1, nKVHeads, past+1, headDim]
        val fullV = ops.concat(listOf(pastV, ops.unsqueeze(v, 0)), dim = 2)
        val scale = mha.attentionScale ?: (1f / sqrt(mha.headDim.toFloat()))
        // GQA-native: fullK/fullV keep nKVHeads, no expandKV (see class doc).
        val o = ops.scaledDotProductAttention(
            query = ops.unsqueeze(q, 0), key = fullK, value = fullV,
            mask = null, scale = scale, causal = false,   // single query attends the whole (already causal) cache
        )   // [1, nHeads, 1, headDim]
        val merged = ops.reshape(ops.squeeze(o, 0), Shape(1, mha.nHeads * mha.headDim))
        val attnOut = proj(ops, mha, merged, idx[3])   // o_proj
        return Triple(attnOut, fullK, fullV)
    }

    // ==================== prefill-with-past: a C-token chunk ====================

    /**
     * KV-cache CHUNK PREFILL: run a fixed-size chunk [tokens] (`[C]` token ids, zero-padded)
     * against the incoming per-layer self cache → logits `[1, vocab]` for the ONE position picked
     * by the one-hot [selectAt] (`[1, C]`), plus the per-layer cache extended by all C positions
     * (the caller slices the padding off). This is what makes an utterance cost one call instead
     * of C single-token [forwardWithPast] steps.
     */
    public fun forwardPrefillWithPast(
        tokens: Tensor<T, V>,
        chunk: ChunkContext<T, V>,
        selectAt: Tensor<T, V>,
        selfKIn: List<Tensor<T, V>>,
        selfVIn: List<Tensor<T, V>>,
        ctx: ExecutionContext,
    ): WithPastOutput<T, V> {
        val ops = ctx.ops
        var h = tokenEmbedding.forward(tokens, ctx)
        val nsk = ArrayList<Tensor<T, V>>(blocks.size)
        val nsv = ArrayList<Tensor<T, V>>(blocks.size)
        for ((i, block) in blocks.withIndex()) {
            val r = refsFor(block)
            val sn = r.attnNorm.forward(h, ctx)
            val (attnOut, fullK, fullV) = attnWithPastChunk(r.mha, sn, chunk.cos, chunk.sin, chunk.mask, selfKIn[i], selfVIn[i], ctx)
            val h1 = ops.add(h, attnOut)
            val ffnOut = r.ffn.forward(r.ffnNorm.forward(h1, ctx), ctx)
            h = ops.add(h1, ffnOut)
            nsk += ops.reshape(fullK, fullK.shape)
            nsv += ops.reshape(fullV, fullV.shape)
        }
        val normed = outputNorm.forward(h, ctx)
        val hAt = ops.matmul(selectAt, normed)   // [1, C] x [C, hidden] -> [1, hidden]
        val logits = lmHead.forward(hAt, ctx)
        return WithPastOutput(logits, nsk, nsv)
    }

    /** [attnWithPast] for a C-row chunk: heads-first `[heads, C, headDim]` projections, RoPE from
     *  `[C, headDim]` tables, and the caller's additive mask `[1, M, C, past+C]` (see [ChunkContext]). */
    private fun attnWithPastChunk(
        mha: MultiHeadAttention<T, V>,
        sn: Tensor<T, V>,
        cos: Tensor<T, V>, sin: Tensor<T, V>,
        mask: Tensor<T, V>,
        pastK: Tensor<T, V>, pastV: Tensor<T, V>,
        ctx: ExecutionContext,
    ): Triple<Tensor<T, V>, Tensor<T, V>, Tensor<T, V>> {
        val ops = ctx.ops
        val rope = mha.rope ?: error("DecoderKvModel.attnWithPastChunk: MHA has no RoPE")
        val idx = paramIdx(mha)
        val c = sn.shape[0]
        fun headsFirst(x: Tensor<T, V>, heads: Int): Tensor<T, V> =
            ops.permute(ops.reshape(x, Shape(c, heads, mha.headDim)), intArrayOf(1, 0, 2))   // [heads, C, headDim]
        var q = headsFirst(proj(ops, mha, sn, idx[0]), mha.nHeads)
        var k = headsFirst(proj(ops, mha, sn, idx[1]), mha.nKVHeads)
        val v = headsFirst(proj(ops, mha, sn, idx[2]), mha.nKVHeads)
        val qn = mha.qNorm; val kn = mha.kNorm
        if (qn != null && kn != null) { q = qn.forward(q, ctx); k = kn.forward(k, ctx) }  // qkNorm BEFORE RoPE
        val cos3 = ops.unsqueeze(cos, 0); val sin3 = ops.unsqueeze(sin, 0)                // [1, C, headDim] over heads
        q = rope.forwardWithCosSin(q, cos3, sin3, ctx)
        k = rope.forwardWithCosSin(k, cos3, sin3, ctx)
        val fullK = ops.concat(listOf(pastK, ops.unsqueeze(k, 0)), dim = 2)   // [1, nKVHeads, past+C, headDim]
        val fullV = ops.concat(listOf(pastV, ops.unsqueeze(v, 0)), dim = 2)
        val scale = mha.attentionScale ?: (1f / sqrt(mha.headDim.toFloat()))
        val o = ops.scaledDotProductAttention(
            query = ops.unsqueeze(q, 0), key = fullK, value = fullV,
            mask = mask, scale = scale, causal = false,   // the additive mask carries causal + padding
        )   // [1, nHeads, C, headDim]
        val merged = ops.reshape(ops.permute(ops.squeeze(o, 0), intArrayOf(1, 0, 2)), Shape(c, mha.nHeads * mha.headDim))
        val attnOut = proj(ops, mha, merged, idx[3])   // o_proj
        return Triple(attnOut, fullK, fullV)
    }

    // ==================== prefill-at: the catalog prefix, once ====================

    /** Null every block's self-attention KV cache so [MultiHeadAttention.forwardWithKV] returns the
     *  freshly computed (uncached) K/V. Idempotent. Mirrors GemmaModel.stripCaches. */
    private fun stripCaches() { for (block in blocks) refsFor(block).mha.kvCache = null }

    /**
     * KV-cache PREFILL: run prompt [input] (`[seq]` or `[1, seq]` token ids) → logits `[1, vocab]`
     * for the ONE position picked by the one-hot [selectAt] (`[1, seq]`), plus the per-layer initial
     * self K/V that seeds [forwardWithPast] / [forwardPrefillWithPast]. Stateless one-pass prefill.
     */
    public fun forwardPrefillAt(input: Tensor<T, V>, selectAt: Tensor<T, V>, ctx: ExecutionContext): PrefillOutput<T, V> {
        stripCaches()
        val ops = ctx.ops
        var h = tokenEmbedding.forward(input, ctx)
        val selfK = ArrayList<Tensor<T, V>>(blocks.size)
        val selfV = ArrayList<Tensor<T, V>>(blocks.size)
        for (block in blocks) {
            val r = refsFor(block)
            val kv = r.mha.forwardWithKV(r.attnNorm.forward(h, ctx), null, ctx)
            val h1 = ops.add(h, kv.output)
            val ffnOut = r.ffn.forward(r.ffnNorm.forward(h1, ctx), ctx)
            h = ops.add(h1, ffnOut)
            selfK += ops.unsqueeze(kv.k, 0)
            selfV += ops.unsqueeze(kv.v, 0)
        }
        val normed = outputNorm.forward(h, ctx)
        val hAt = ops.matmul(selectAt, if (normed.rank == 3) ops.squeeze(normed, dim = 0) else normed)   // [1, hidden]
        val logits = lmHead.forward(hAt, ctx)                                                            // [1, vocab]
        return PrefillOutput(logits, selfK, selfV)
    }

    /**
     * Build the per-step RoPE cos/sin table for [position] (seqLen 1), from the first block's RoPE
     * (every block shares the same base — `decoderTransformerNetwork` builds one `ropeBase` for
     * the whole model). Fed to [forwardWithPast] as a runtime input so the decode graph carries no
     * compile-time position.
     */
    public fun buildRopeCosSin(position: Int, ctx: ExecutionContext): RopeCosSin<T, V> {
        val rope = refsFor(blocks.first()).mha.rope ?: error("DecoderKvModel: no RoPE on the first block")
        val (c, s) = when (rope.mode) {
            RoPEMode.SPLIT_HALF -> rope.buildSplitHalfCosSin(position, 1)
            RoPEMode.INTERLEAVED -> rope.buildInterleavedCosSin(position, 1)
        }
        val shape = Shape(1, rope.headDim)
        return RopeCosSin(cos = ctx.fromFloatArray(shape, dtype, c), sin = ctx.fromFloatArray(shape, dtype, s))
    }
}
