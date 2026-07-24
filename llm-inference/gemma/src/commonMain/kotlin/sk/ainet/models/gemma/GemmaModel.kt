package sk.ainet.models.gemma

import sk.ainet.apps.llm.HybridTransformerBlock
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.layers.Embedding
import sk.ainet.lang.nn.layers.EmbeddingAdapter
import sk.ainet.lang.nn.normalization.RMSNormalization
import sk.ainet.lang.nn.transformer.GeGLUFFN
import sk.ainet.lang.nn.transformer.LayerScalarMul
import sk.ainet.lang.nn.transformer.MultiHeadAttention
import sk.ainet.lang.nn.transformer.VoidDense
import sk.ainet.lang.nn.transformer.linearProject
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.math.sqrt
import kotlin.reflect.KClass

/**
 * Top-level Gemma 4 model that orchestrates Per-Layer Embedding (PLE)
 * threading through the decoder layer stack.
 *
 * The generic DSL Sequential wrapper that `dslImpl.create()` produces can't
 * express PLE: each decoder layer needs both the rolling `hidden_states`
 * tensor AND a pre-computed, per-layer auxiliary tensor
 * `per_layer_inputs[:, :, layerIdx, :]`. This wrapper:
 *
 * 1. Runs [embedding] on the raw token IDs to get `inputs_embeds`.
 * 2. If [ple] is non-null, computes `per_layer_inputs` via
 *    [PerLayerEmbedding.compute] using both the raw IDs and `inputs_embeds`.
 * 3. For each block, slices the per-layer tensor, finds that block's
 *    [PerLayerInputBlockHook] in its module list, sets
 *    [PerLayerInputBlockHook.perLayerInput], and calls the block forward.
 *    The hook then consumes the field inside the block's sequential run.
 * 4. Applies final [outputNorm] and [lmHead] projections.
 *
 * When [ple] is null, this degenerates to a plain
 * `embedding → blocks → norm → lm_head` sequence — identical observable
 * behaviour to the old `dslImpl.create()` shape.
 *
 * Not a regular Sequential because the block loop needs to interleave
 * hook-field writes with module forwards. The cost of leaving the generic
 * DSL for one model is small; Gemma 4 is architecturally unique enough
 * that a dedicated top-level is justified.
 */
public class GemmaModel<T : DType, V>(
    public val tokenEmbedding: EmbeddingAdapter<T, V>,
    public val ple: PerLayerEmbedding<T, V>?,
    public val blocks: List<HybridTransformerBlock<T, V>>,
    public val outputNorm: RMSNormalization<T, V>,
    public val lmHead: VoidDense<T, V>,
    public val dtype: KClass<T>,
    /**
     * Multiplier applied to token-embedding output before the trunk. Gemma 4
     * uses `sqrt(hidden_size)` (HF: `embed_scale=config.hidden_size**0.5`).
     * Without this scale, trunk activations are ~1/sqrt(dim) of their trained
     * magnitude and decode collapses to a near-uniform distribution. Set to
     * `1f` for models that do not require it.
     */
    public val embedScale: Float = 1f,
    /**
     * `final_logit_softcapping` from the Gemma 4 config. When `> 0`, the
     * lm_head output is passed through `softcap * tanh(logits / softcap)`
     * — matches `Gemma4ForCausalLM.forward` in transformers 5.6.0. Gemma
     * 4 E2B sets this to 30.0 in the GGUF (`gemma4.final_logit_softcapping`).
     * Set to `0f` to disable.
     */
    public val finalLogitSoftcapping: Float = 0f,
    override val name: String = "GemmaModel"
) : Module<T, V>() {

    override val modules: List<Module<T, V>> = buildList {
        add(tokenEmbedding)
        if (ple != null) add(ple)
        addAll(blocks)
        add(outputNorm)
        add(lmHead)
    }

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        // Diagnostic: dump per-block hidden state stats when GEMMA4_DUMP_HIDDEN=1.
        // Compares against expected llama.cpp magnitudes to localize the
        // BOS-loop forward-pass bug. See gemma4-research/findings/dsl_vs_llamacpp_logit_divergence.md.
        val dumpHidden = sk.ainet.apps.llm.diag.envFlag("GEMMA4_DUMP_HIDDEN")

        // Step 1: run main embedding. OptimizedLLMRuntime passes a token-id
        // tensor (Int32, shape [seq] — often [1] for single-token decode);
        // Embedding expands to [seq, hiddenSize]. Apply the Gemma 4
        // sqrt(hidden_size) scale immediately on the result.
        val rawEmbeds = tokenEmbedding.forward(input, ctx)
        val inputsEmbeds = if (embedScale != 1f) ctx.ops.mulScalar(rawEmbeds, embedScale) else rawEmbeds
        if (dumpHidden) dumpHiddenStats("embed-raw    ", rawEmbeds)
        if (dumpHidden) dumpHiddenStats("embed-scaled ", inputsEmbeds)

        // Step 2: compute per_layer_inputs iff PLE is active.
        val perLayerInputs: Tensor<T, V>? = ple?.let { pleModule ->
            val tokenIds2d = ensureRank2Ids(input, ctx)
            val embeds3d = ensureRank3Embeds(inputsEmbeds, ctx)
            pleModule.compute(tokenIds2d, embeds3d, ctx, dtype)
        }
        if (dumpHidden && perLayerInputs != null) dumpHiddenStats("ple-inputs   ", perLayerInputs)

        // Step 3: iterate blocks. For each, pre-populate the hook's
        // perLayerInput with the corresponding slice if PLE is on.
        var hidden = inputsEmbeds
        for ((layerIdx, block) in blocks.withIndex()) {
            if (perLayerInputs != null) {
                val hook = findHook(block)
                    ?: error("GemmaModel: PLE is active but blk.$layerIdx has no PerLayerInputBlockHook")
                val slice3d = perLayerInputSliceFor(perLayerInputs, layerIdx, ctx)
                // Hook expects [batch, seq, perLayerDim] matching the
                // block's [seq, hiddenSize] working shape. Squeeze batch
                // dim if the block is operating in rank-2 mode.
                hook.perLayerInput = if (hidden.rank == 2) {
                    ctx.ops.squeeze(slice3d, dim = 0)
                } else {
                    slice3d
                }
            }
            hidden = block.forward(hidden, ctx)
            if (dumpHidden) dumpHiddenStats("blk.${layerIdx.toString().padStart(2, '0')}     ", hidden)
        }

        // Step 4: final norm + lm_head.
        hidden = outputNorm.forward(hidden, ctx)
        if (dumpHidden) dumpHiddenStats("post-norm    ", hidden)
        var logits = lmHead.forward(hidden, ctx)
        if (dumpHidden) dumpHiddenStats("logits-pre-sc", logits)

        // Step 5: Gemma 4 final logit softcapping (matches HF
        // `Gemma4ForCausalLM.forward`). Without this the model latches
        // onto degenerate attractor tokens during decode.
        if (finalLogitSoftcapping > 0f) {
            val ops = ctx.ops
            // Heap-backed scalar wrap — fromFloatArray copies even
            // single-float tables into a fresh Arena.ofAuto MemorySegment;
            // running per forward step accumulates direct memory the GC
            // can't see. Same root cause as commit 319c394.
            val scaleShape = sk.ainet.lang.tensor.Shape(1)
            val scale: Tensor<T, V> = ctx.fromData(
                sk.ainet.lang.tensor.data.DenseFloatArrayTensorData<T>(scaleShape, floatArrayOf(1f / finalLogitSoftcapping)) as sk.ainet.lang.tensor.data.TensorData<T, V>,
                dtype
            )
            val inv: Tensor<T, V> = ctx.fromData(
                sk.ainet.lang.tensor.data.DenseFloatArrayTensorData<T>(scaleShape, floatArrayOf(finalLogitSoftcapping)) as sk.ainet.lang.tensor.data.TensorData<T, V>,
                dtype
            )
            logits = ops.multiply(ops.tanh(ops.multiply(logits, scale)), inv)
        }

        return logits
    }

    /**
     * Diagnostic: dump per-block hidden state stats. Delegates to the
     * platform diagnostic helper so the multiplatform metadata compile
     * doesn't see JVM-only formatter / MemorySegment references. JVM
     * does the full formatted dump; non-JVM targets are no-ops (no
     * profiling story there worth maintaining).
     */
    private fun dumpHiddenStats(stage: String, t: Tensor<T, V>) {
        sk.ainet.apps.llm.diag.dumpStats("[hidden] $stage", t)
    }

    @Suppress("UNCHECKED_CAST")
    private fun findHook(block: HybridTransformerBlock<T, V>): PerLayerInputBlockHook<T, V>? =
        block.modules.firstOrNull { it is PerLayerInputBlockHook<*, *> }
            as? PerLayerInputBlockHook<T, V>

    /** Extract `per_layer_inputs[..., layerIdx, :]` as a `[batch, seq, perLayerDim]` tensor. */
    private fun perLayerInputSliceFor(
        perLayerInputs: Tensor<T, V>,
        layerIdx: Int,
        ctx: ExecutionContext
    ): Tensor<T, V> {
        val ops = ctx.ops
        // perLayerInputs shape: [batch, seq, numLayers, perLayerDim]
        // narrow on dim=2 from `layerIdx` for 1 step, then squeeze dim=2.
        val narrowed = ops.narrow(perLayerInputs, dim = 2, start = layerIdx, length = 1)
        return ops.squeeze(narrowed, dim = 2)
    }

    /** OptimizedLLMRuntime may pass a rank-1 token-id tensor `[seq]`
     *  (typically `[1]` for single-token decode). PLE requires
     *  `[batch, seq]` — unsqueeze a leading batch dim. */
    private fun ensureRank2Ids(ids: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> = when (ids.rank) {
        2 -> ids
        1 -> ctx.ops.unsqueeze(ids, dim = 0)
        else -> error("GemmaModel: expected token-id tensor of rank 1 or 2, got ${ids.rank}")
    }

    /** inputsEmbeds may come in as `[seq, hidden]` (rank 2) when the
     *  runtime decodes single-token, or `[batch, seq, hidden]` (rank 3)
     *  when batched. PLE needs rank 3 (`[batch, seq, hidden]`) to combine
     *  with the `[batch, seq, numLayers, perLayerDim]` PLE tensor. */
    private fun ensureRank3Embeds(embeds: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> = when (embeds.rank) {
        3 -> embeds
        2 -> ctx.ops.unsqueeze(embeds, dim = 0)
        else -> error("GemmaModel: expected inputs_embeds of rank 2 or 3, got ${embeds.rank}")
    }

    // ================================================================
    // KV-cache with_past decode path (perf program, Phase 1).
    //
    // forwardPrefill + forwardWithPast author the two-graph KV-cached decode in
    // the DSL, mirroring MoonshineDecoderModel.forwardPrefill/forwardWithPast but
    // for the decoder-only Gemma block: GQA (nKVHeads < nHeads), qkNorm-before-
    // RoPE, SPLIT_HALF RoPE with two bases, sandwich post-norms, layer_output_
    // scale, and final logit softcap. Per-block sub-modules are resolved
    // POSITIONALLY from HybridTransformerBlock.modules — not via that class's own
    // typed fields, which assume SwiGLU and mis-select Gemma's four RMSNorms.
    // ================================================================

    /** logits `[seq, vocab]` + per-layer initial self K/V (`[1, nKVHeads, seq, headDim]`). */
    public class GemmaPrefillOutput<T : DType, V>(
        public val logits: Tensor<T, V>,
        public val selfK: List<Tensor<T, V>>,
        public val selfV: List<Tensor<T, V>>,
    )

    /** logits `[1, vocab]` + per-layer extended self K/V (`[1, nKVHeads, past+1, headDim]`). */
    public class GemmaWithPastOutput<T : DType, V>(
        public val logits: Tensor<T, V>,
        public val selfK: List<Tensor<T, V>>,
        public val selfV: List<Tensor<T, V>>,
    )

    /** Per-step SPLIT_HALF RoPE cos/sin (`[1, headDim]`), one pair per RoPE base
     *  (global / sliding); [forwardWithPast] routes per layer by `MHA.slidingWindow`. */
    public class RopeCosSin<T : DType, V>(
        public val cosGlobal: Tensor<T, V>,
        public val sinGlobal: Tensor<T, V>,
        public val cosSliding: Tensor<T, V>,
        public val sinSliding: Tensor<T, V>,
    )

    private class GemmaBlockRefs<T : DType, V>(
        val attnNorm: RMSNormalization<T, V>,
        val mha: MultiHeadAttention<T, V>,
        val postAttnNorm: RMSNormalization<T, V>?,
        val ffnNorm: RMSNormalization<T, V>,
        val ffn: GeGLUFFN<T, V>,
        val postFfwNorm: RMSNormalization<T, V>?,
        val outScale: LayerScalarMul<T, V>?,
    )

    @Suppress("UNCHECKED_CAST")
    private fun refsFor(block: HybridTransformerBlock<T, V>): GemmaBlockRefs<T, V> {
        val mods = block.modules
        val mha = mods.filterIsInstance<MultiHeadAttention<T, V>>().firstOrNull()
            ?: error("GemmaModel: block ${block.name} has no MultiHeadAttention")
        val ffn = mods.filterIsInstance<GeGLUFFN<T, V>>().firstOrNull()
            ?: error("GemmaModel: block ${block.name} has no GeGLUFFN")
        val mhaIdx = mods.indexOf(mha)
        val ffnIdx = mods.indexOf(ffn)
        fun rmsAt(i: Int): RMSNormalization<T, V>? = mods.getOrNull(i) as? RMSNormalization<T, V>
        val attnNorm = rmsAt(mhaIdx - 1) ?: error("GemmaModel: no attn_norm before attn in ${block.name}")
        val ffnNorm = rmsAt(ffnIdx - 1) ?: error("GemmaModel: no ffn_norm before ffn in ${block.name}")
        return GemmaBlockRefs(
            attnNorm = attnNorm,
            mha = mha,
            postAttnNorm = rmsAt(mhaIdx + 1),   // present only with sandwich norms
            ffnNorm = ffnNorm,
            ffn = ffn,
            postFfwNorm = rmsAt(ffnIdx + 1),     // present only with sandwich norms
            outScale = mods.filterIsInstance<LayerScalarMul<T, V>>().firstOrNull(),
        )
    }

    /** Null every block's self-attention KV cache so [forwardPrefill]'s
     *  `MHA.forwardWithKV` returns the freshly computed (uncached) K/V and does not
     *  run PositionalKVCache's stateful eager copy path. Idempotent. */
    private fun stripCaches() {
        for (block in blocks) refsFor(block).mha.kvCache = null
    }

    /** Param indices [q, k, v, o] into MHA.params (bias shifts them). */
    private fun paramIdx(mha: MultiHeadAttention<T, V>): IntArray =
        if (mha.bias) intArrayOf(0, 2, 4, 6) else intArrayOf(0, 1, 2, 3)

    /** Project q/k/v (`paramIdx` into MHA.params) of a single-token input to heads-first
     *  `[heads, 1, headDim]`. For seq==1 this equals MHA's swap-then-reshape. */
    private fun projHeads(mha: MultiHeadAttention<T, V>, x: Tensor<T, V>, paramIdx: Int, heads: Int, ctx: ExecutionContext): Tensor<T, V> {
        val p = linearProject(ctx.ops, x, mha.params[paramIdx].value)
        return ctx.ops.reshape(p, Shape(heads, 1, mha.headDim))
    }

    /** Repeat each KV head `nHeads/nKVHeads` times along dim=1 of `[1, nKVHeads, S, headDim]`
     *  → `[1, nHeads, S, headDim]`, matching the GQA head mapping (head h → KV head h/nRep). */
    private fun expandKV(t: Tensor<T, V>, nHeads: Int, nKVHeads: Int, ops: sk.ainet.lang.tensor.ops.TensorOps): Tensor<T, V> {
        if (nHeads == nKVHeads) return t
        val nRep = nHeads / nKVHeads
        val parts = ArrayList<Tensor<T, V>>(nHeads)
        for (h in 0 until nKVHeads) {
            val slice = ops.narrow(t, 1, h, 1)   // [1, 1, S, headDim]
            repeat(nRep) { parts.add(slice) }
        }
        return ops.concat(parts, dim = 1)
    }

    /** Hand-wired single-new-token self-attention over the past cache for one block. RoPE applies
     *  at the runtime position via the fed-in [cos]/[sin]. Returns (attnOut `[1, qDim]`,
     *  extendedK `[1, nKVHeads, past+1, headDim]`, extendedV). */
    private fun attnWithPast(
        mha: MultiHeadAttention<T, V>,
        sn: Tensor<T, V>,
        cos: Tensor<T, V>, sin: Tensor<T, V>,
        pastK: Tensor<T, V>, pastV: Tensor<T, V>,
        ctx: ExecutionContext,
    ): Triple<Tensor<T, V>, Tensor<T, V>, Tensor<T, V>> {
        val ops = ctx.ops
        val rope = mha.rope ?: error("GemmaModel.attnWithPast: MHA has no RoPE")
        val idx = paramIdx(mha)
        var q = projHeads(mha, sn, idx[0], mha.nHeads, ctx)     // [nHeads, 1, headDim]
        var k = projHeads(mha, sn, idx[1], mha.nKVHeads, ctx)   // [nKVHeads, 1, headDim]
        val v = projHeads(mha, sn, idx[2], mha.nKVHeads, ctx)
        val qn = mha.qNorm; val kn = mha.kNorm
        if (qn != null && kn != null) { q = qn.forward(q, ctx); k = kn.forward(k, ctx) }  // qkNorm BEFORE RoPE
        q = rope.forwardWithCosSin(q, cos, sin, ctx)
        k = rope.forwardWithCosSin(k, cos, sin, ctx)
        val fullK = ops.concat(listOf(pastK, ops.unsqueeze(k, 0)), dim = 2)   // [1, nKVHeads, past+1, headDim]
        val fullV = ops.concat(listOf(pastV, ops.unsqueeze(v, 0)), dim = 2)
        val eK = expandKV(fullK, mha.nHeads, mha.nKVHeads, ops)
        val eV = expandKV(fullV, mha.nHeads, mha.nKVHeads, ops)
        val scale = mha.attentionScale ?: (1f / sqrt(mha.headDim.toFloat()))
        val o = ops.scaledDotProductAttention(
            query = ops.unsqueeze(q, 0), key = eK, value = eV,
            mask = null, scale = scale, causal = false,   // single query attends the whole cache → no mask
        )   // [1, nHeads, 1, headDim]
        val merged = ops.reshape(ops.squeeze(o, 0), Shape(1, mha.nHeads * mha.headDim))
        val attnOut = linearProject(ops, merged, mha.params[idx[3]].value)   // o_proj
        return Triple(attnOut, fullK, fullV)
    }

    /** Gemma-4 final logit softcapping `softcap * tanh(logits / softcap)` (no-op when disabled). */
    private fun applySoftcap(logits: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        if (finalLogitSoftcapping <= 0f) return logits
        val ops = ctx.ops
        val scaleShape = Shape(1)
        val scale: Tensor<T, V> = ctx.fromData(
            sk.ainet.lang.tensor.data.DenseFloatArrayTensorData<T>(scaleShape, floatArrayOf(1f / finalLogitSoftcapping)) as sk.ainet.lang.tensor.data.TensorData<T, V>,
            dtype,
        )
        val inv: Tensor<T, V> = ctx.fromData(
            sk.ainet.lang.tensor.data.DenseFloatArrayTensorData<T>(scaleShape, floatArrayOf(finalLogitSoftcapping)) as sk.ainet.lang.tensor.data.TensorData<T, V>,
            dtype,
        )
        return ops.multiply(ops.tanh(ops.multiply(logits, scale)), inv)
    }

    /**
     * KV-cache PREFILL: run prompt [input] (`[seq]` or `[1, seq]` token ids) → logits `[seq, vocab]`
     * plus the per-layer initial self K/V (`[1, nKVHeads, seq, headDim]`) that seed [forwardWithPast].
     * Nulls the block KV caches first (stateless one-pass prefill). PLE is unsupported here.
     */
    public fun forwardPrefill(input: Tensor<T, V>, ctx: ExecutionContext): GemmaPrefillOutput<T, V> {
        require(ple == null) { "GemmaModel.forwardPrefill: PLE models not supported by the KV-cache path yet" }
        stripCaches()
        val ops = ctx.ops
        val rawEmbeds = tokenEmbedding.forward(input, ctx)
        var h = if (embedScale != 1f) ops.mulScalar(rawEmbeds, embedScale) else rawEmbeds
        val selfK = ArrayList<Tensor<T, V>>(blocks.size)
        val selfV = ArrayList<Tensor<T, V>>(blocks.size)
        for (block in blocks) {
            val r = refsFor(block)
            val kv = r.mha.forwardWithKV(r.attnNorm.forward(h, ctx), null, ctx)
            val postAttn = r.postAttnNorm?.forward(kv.output, ctx) ?: kv.output
            val h1 = ops.add(h, postAttn)
            val ffnOut = r.ffn.forward(r.ffnNorm.forward(h1, ctx), ctx)
            val postFfw = r.postFfwNorm?.forward(ffnOut, ctx) ?: ffnOut
            var h2 = ops.add(h1, postFfw)
            if (r.outScale != null) h2 = r.outScale.forward(h2, ctx)
            h = h2
            selfK += ops.unsqueeze(kv.k, 0)
            selfV += ops.unsqueeze(kv.v, 0)
        }
        val logits = applySoftcap(lmHead.forward(outputNorm.forward(h, ctx), ctx), ctx)
        return GemmaPrefillOutput(logits, selfK, selfV)
    }

    /**
     * KV-cache DECODE step: one new [tokenId] (`[1]` or `[1,1]`) at the position encoded by [rope]
     * ([buildRopeCosSin]); the incoming per-layer self cache ([selfKIn]/[selfVIn],
     * `[1, nKVHeads, past, headDim]`) → logits `[1, vocab]` + the per-layer extended self cache
     * (`[1, nKVHeads, past+1, headDim]`). Decoder-only — no cross-attention.
     */
    public fun forwardWithPast(
        tokenId: Tensor<T, V>,
        rope: RopeCosSin<T, V>,
        selfKIn: List<Tensor<T, V>>,
        selfVIn: List<Tensor<T, V>>,
        ctx: ExecutionContext,
    ): GemmaWithPastOutput<T, V> {
        require(ple == null) { "GemmaModel.forwardWithPast: PLE models not supported by the KV-cache path yet" }
        val ops = ctx.ops
        val rawEmbeds = tokenEmbedding.forward(tokenId, ctx)
        var h = if (embedScale != 1f) ops.mulScalar(rawEmbeds, embedScale) else rawEmbeds
        val nsk = ArrayList<Tensor<T, V>>(blocks.size)
        val nsv = ArrayList<Tensor<T, V>>(blocks.size)
        for ((i, block) in blocks.withIndex()) {
            val r = refsFor(block)
            val isGlobal = r.mha.slidingWindow == null
            val cos = if (isGlobal) rope.cosGlobal else rope.cosSliding
            val sin = if (isGlobal) rope.sinGlobal else rope.sinSliding
            val sn = r.attnNorm.forward(h, ctx)
            val (attnOut, fullK, fullV) = attnWithPast(r.mha, sn, cos, sin, selfKIn[i], selfVIn[i], ctx)
            val postAttn = r.postAttnNorm?.forward(attnOut, ctx) ?: attnOut
            val h1 = ops.add(h, postAttn)
            val ffnOut = r.ffn.forward(r.ffnNorm.forward(h1, ctx), ctx)
            val postFfw = r.postFfwNorm?.forward(ffnOut, ctx) ?: ffnOut
            var h2 = ops.add(h1, postFfw)
            if (r.outScale != null) h2 = r.outScale.forward(h2, ctx)
            h = h2
            nsk += ops.reshape(fullK, fullK.shape)   // identity → distinct graph output node
            nsv += ops.reshape(fullV, fullV.shape)
        }
        val logits = applySoftcap(lmHead.forward(outputNorm.forward(h, ctx), ctx), ctx)
        return GemmaWithPastOutput(logits, nsk, nsv)
    }

    /**
     * Build the per-step SPLIT_HALF RoPE cos/sin tables for [position] (seqLen 1): one pair from a
     * global-base layer, one from a sliding-base layer (gemma3 uses two RoPE bases). Fed to
     * [forwardWithPast] as runtime inputs so the decode graph carries no compile-time position.
     */
    public fun buildRopeCosSin(position: Int, ctx: ExecutionContext): RopeCosSin<T, V> {
        val mhas = blocks.map { refsFor(it).mha }
        val globalMha = mhas.firstOrNull { it.slidingWindow == null }
        val slidingMha = mhas.firstOrNull { it.slidingWindow != null }
        val gRope = (globalMha ?: slidingMha ?: error("GemmaModel: no attention layers")).rope!!
        val sRope = (slidingMha ?: globalMha)!!.rope!!
        val (gc, gs) = gRope.buildSplitHalfCosSin(position, 1)
        val (sc, ss) = sRope.buildSplitHalfCosSin(position, 1)
        val gShape = Shape(1, gRope.headDim)
        val sShape = Shape(1, sRope.headDim)
        return RopeCosSin(
            cosGlobal = ctx.fromFloatArray(gShape, dtype, gc),
            sinGlobal = ctx.fromFloatArray(gShape, dtype, gs),
            cosSliding = ctx.fromFloatArray(sShape, dtype, sc),
            sinSliding = ctx.fromFloatArray(sShape, dtype, ss),
        )
    }

    public companion object {
        /**
         * Walk [stageModules] and return the first [PerLayerInputBlockHook]
         * found, or `null`. Package-internal helper for tests that want to
         * poke at a block's hook without reflection.
         */
        @Suppress("UNCHECKED_CAST")
        public fun <T : DType, V> findHookIn(
            stageModules: List<Module<T, V>>
        ): PerLayerInputBlockHook<T, V>? =
            stageModules.firstOrNull { it is PerLayerInputBlockHook<*, *> }
                as? PerLayerInputBlockHook<T, V>
    }
}
