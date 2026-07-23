package sk.ainet.models.moonshine

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.hooks.withForwardHooks
import sk.ainet.lang.nn.normalization.LayerNormalization
import sk.ainet.lang.nn.transformer.MultiHeadAttention
import sk.ainet.lang.nn.transformer.RoPE
import sk.ainet.lang.nn.transformer.RoPEMode
import sk.ainet.lang.nn.transformer.VoidDense
import sk.ainet.lang.nn.transformer.linearProject
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.operators.bind
import sk.ainet.lang.types.DType
import kotlin.math.sqrt
import kotlin.reflect.KClass

/**
 * Moonshine-tiny DECODER, authored in the SKaiNET NN DSL (Phase A stub).
 *
 * The decoder is seq2seq. Each pre-norm layer is three sublayers:
 *   x + SelfAttn(LN(x))                       — causal, RoPE, no bias, KV-cached at runtime
 *   x + CrossAttn(LN(x), encoderMemory)       — attends to the encoder memory, no RoPE, non-causal
 *   x + MLP(LN(x))                            — GATED SiLU MLP (see below)
 * then a final LayerNorm and the `proj_out` vocab projection (tied to `embed_tokens`).
 *
 * The MLP is the gated form Moonshine's decoder uses (verified against HF
 * `MoonshineDecoderMLP`, `decoder_hidden_act="silu"`): a single fused `fc1`
 * `[2·ffnDim, dim]` produces `[value | gate]`, then `silu(gate) * value`, then `fc2`
 * `[dim, ffnDim]`; both fc1 and fc2 are biased. (The encoder MLP, by contrast, is a
 * plain biased GELU MLP.)
 *
 * ## Why a bespoke module instead of the `HybridTransformerBlock` used by the encoder
 * Cross-attention needs a **second** input — the encoder memory — but the DSL's
 * `Stage`/`HybridTransformerBlock` thread a single tensor through a module list, so
 * there is no way to route the memory through them. [MultiHeadAttention] already
 * exposes the cross-attention entry `forward(input, encoderMemory, ctx)` (Q from the
 * decoder state, K/V from the memory, RoPE + KV-cache bypassed, causal forced off);
 * this layer just drives it imperatively, mirroring that two-argument contract. This
 * keeps all decoder wiring inside the moonshine module — no `transformer-core` change.
 *
 * ## Scope of this stub
 * Phase A proves the wiring: the model builds and traces to StableHLO with two graph
 * inputs (`inputs_embeds` + encoder memory). It is **not** yet numerically validated
 * (weights, Phase B) and does **not** yet export the two KV-cached decode graphs
 * (Phase D) — self-attention here runs a plain causal prefill with no exported cache,
 * and cross-K/V are re-projected rather than cached. The token embedding lookup is
 * intentionally left OUT of the graph: the board decoder consumes `inputs_embeds`
 * `[1, seq, dim]` (the embedding table is applied host-side), matching the vendor.
 *
 * ⚠️ Known Phase D blocker surfaced by this stub: at seq == 1 (the real one-token
 * decode shape) the self-attention takes [MultiHeadAttention]'s fused-decode fast
 * path, whose buffer-direct ops do not record on the trace tape — the seq=1 graph
 * leaks per-layer self K/V as dangling outputs. seq ≥ 2 traces cleanly (general SDPA
 * path). Exporting the KV-cached decode graphs must disable the fast path for tracing.
 *
 * **dtype-portable**, like the encoder: pass `FP32` for portable host/GPU builds, `BF16` for the Torq
 * NPU (where weights must stay bf16 at the matmul). bf16 is a *target choice*, not a model property —
 * see this module's `README.md`. Param ids are layer-qualified for by-name weight loading.
 */

/** One Moonshine decoder layer: causal self-attn → cross-attn(memory) → GELU MLP, all pre-norm. */
public class MoonshineDecoderLayer<T : DType, V>(
    cfg: MoonshineConfig,
    layer: Int,
    dtype: KClass<T>,
) : Module<T, V>() {

    override val name: String = "dec.$layer"

    private val dim = cfg.dim

    private val selfNorm = LayerNormalization<T, V>(
        normalizedShape = intArrayOf(dim),
        eps = cfg.layerNormEps.toDouble(),
        name = "dec.$layer.self_attn_norm",
        dtype = dtype,
    )

    private val selfAttn = MultiHeadAttention<T, V>(
        dim = dim,
        nHeads = cfg.nHeads,
        nKVHeads = cfg.nHeads,
        causal = true, // decoder self-attention is causal
        bias = false,
        name = "dec.$layer.self_attn",
        rope = RoPE(
            headDim = cfg.headDim,
            maxSeqLen = cfg.maxDecodeTokens,
            base = cfg.ropeBase,
            mode = RoPEMode.INTERLEAVED, // same partial-rotary form as the encoder (verified bit-exact vs ONNX)
            partialRotaryFactor = cfg.partialRotaryFactor,
            freqDenomRotaryDim = true, // inv_freq denom = rotaryDim (32), not headDim (36)
            name = "dec.$layer.self_attn.rope",
        ),
        kvCache = null, // Phase D exports the self KV cache as graph I/O
        dtype = dtype,
    )

    private val crossNorm = LayerNormalization<T, V>(
        normalizedShape = intArrayOf(dim),
        eps = cfg.layerNormEps.toDouble(),
        name = "dec.$layer.cross_attn_norm",
        dtype = dtype,
    )

    // Cross-attention: Q from the decoder state, K/V from the encoder memory. No RoPE
    // (memory frames are already positioned by the encoder), non-causal, no cache here
    // (cross-K/V are cached at the runtime/graph layer — the module rejects kvCache in
    // cross mode). causal=false is also enforced internally for the cross path.
    private val crossAttn = MultiHeadAttention<T, V>(
        dim = dim,
        nHeads = cfg.nHeads,
        nKVHeads = cfg.nHeads,
        causal = false,
        bias = false,
        name = "dec.$layer.cross_attn",
        rope = null,
        kvCache = null,
        dtype = dtype,
    )

    private val mlpNorm = LayerNormalization<T, V>(
        normalizedShape = intArrayOf(dim),
        eps = cfg.layerNormEps.toDouble(),
        name = "dec.$layer.mlp_norm",
        dtype = dtype,
    )

    // Gated SiLU MLP. Fused fc1 [2·ffnDim, dim] emits [value | gate]; fc2 [dim, ffnDim].
    private val ffnDim = cfg.ffnDim
    private val mlpFc1 = VoidDense<T, V>("dec.$layer.mlp_fc1", cfg.ffnDim * 2, dim, dtype, addBias = true)
    private val mlpFc2 = VoidDense<T, V>("dec.$layer.mlp_fc2", dim, cfg.ffnDim, dtype, addBias = true)

    override val modules: List<Module<T, V>> =
        listOf(selfNorm, selfAttn, crossNorm, crossAttn, mlpNorm, mlpFc1, mlpFc2)

    /** The self-attention RoPE (all layers share config) — used to build runtime cos/sin tables. */
    public val selfRope: RoPE<T, V> get() = selfAttn.rope!!

    /**
     * Cross-attention-aware forward. [input] is the decoder hidden state
     * `[·, seq, dim]`; [encoderMemory] is the encoder output `[·, frames, dim]`.
     */
    public fun forward(
        input: Tensor<T, V>,
        encoderMemory: Tensor<T, V>,
        ctx: ExecutionContext,
    ): Tensor<T, V> {
        val boundInput = input.bind(ctx)
        val boundMemory = encoderMemory.bind(ctx)
        return withForwardHooks(ctx, this, boundInput) {
            val ops = ctx.ops
            // x + SelfAttn(LN(x))
            val afterSelf = ops.add(boundInput, selfAttn.forward(selfNorm.forward(boundInput, ctx), ctx))
            // x + CrossAttn(LN(x), memory)
            val afterCross = ops.add(afterSelf, crossAttn.forward(crossNorm.forward(afterSelf, ctx), boundMemory, ctx))
            // x + MLP(LN(x)) — gated SiLU: fc1 → [value | gate] → silu(gate)*value → fc2
            val h = mlpFc1.forward(mlpNorm.forward(afterCross, ctx), ctx)
            val lastDim = h.rank - 1
            val value = ops.narrow(h, lastDim, 0, ffnDim)
            val gate = ops.narrow(h, lastDim, ffnDim, ffnDim)
            val mlpOut = mlpFc2.forward(ops.multiply(ops.silu(gate), value), ctx)
            ops.add(afterCross, mlpOut)
        }
    }

    /**
     * Prefill variant of [forward] that also returns the K/V this layer computed:
     * `selfK/selfV` `[nHeads, seq, headDim]` (post-RoPE) and `crossK/crossV`
     * `[nHeads, frames, headDim]` (memory projections). Used to export the KV-cached
     * decode graphs — the `decoder` prefill graph surfaces these so `decoder_with_past`
     * can consume them. Numerically identical to [forward] (same attention code).
     */
    public fun forwardWithKV(
        input: Tensor<T, V>,
        encoderMemory: Tensor<T, V>,
        ctx: ExecutionContext,
    ): MoonshineLayerKV<T, V> {
        val ops = ctx.ops
        val selfKV = selfAttn.forwardWithKV(selfNorm.forward(input, ctx), null, ctx)
        val afterSelf = ops.add(input, selfKV.output)
        val crossKV = crossAttn.forwardWithKV(crossNorm.forward(afterSelf, ctx), encoderMemory, ctx)
        val afterCross = ops.add(afterSelf, crossKV.output)
        val h = mlpFc1.forward(mlpNorm.forward(afterCross, ctx), ctx)
        val lastDim = h.rank - 1
        val value = ops.narrow(h, lastDim, 0, ffnDim)
        val gate = ops.narrow(h, lastDim, ffnDim, ffnDim)
        val mlpOut = mlpFc2.forward(ops.multiply(ops.silu(gate), value), ctx)
        return MoonshineLayerKV(ops.add(afterCross, mlpOut), selfKV.k, selfKV.v, crossKV.k, crossKV.v)
    }

    /**
     * `decoder_with_past` step for ONE new token at [position]. Self-attention appends the
     * token's K/V to the incoming self cache ([selfKIn]/[selfVIn], `[1, nHeads, pos, headDim]`)
     * and attends over the full cache; cross-attention consumes the CACHED cross K/V
     * ([crossKIn]/[crossVIn], `[1, nHeads, frames, headDim]`) directly — no memory, no cross
     * projection. Returns the layer output + the extended self cache (`[1, nHeads, pos+1, headDim]`).
     *
     * Hand-wired (not via `MultiHeadAttention.forward`) because the cache is graph I/O and RoPE
     * must apply at the runtime [position]; it reuses the module's projection weights, RoPE module
     * and the shared SDPA op, and for a single token the head reshape is a plain reshape — so it is
     * numerically identical to the validated attention (checked by the two-graph transcript test).
     */
    public fun forwardWithPast(
        input: Tensor<T, V>,
        ropeCos: Tensor<T, V>,
        ropeSin: Tensor<T, V>,
        selfKIn: Tensor<T, V>,
        selfVIn: Tensor<T, V>,
        crossKIn: Tensor<T, V>,
        crossVIn: Tensor<T, V>,
        ctx: ExecutionContext,
    ): MoonshinePastKV<T, V> {
        val ops = ctx.ops
        // --- self-attention: project, RoPE@runtime-position (cos/sin fed in), append + attend ---
        val sn = selfNorm.forward(input, ctx)
        val q = selfAttn.rope!!.forwardWithCosSin(projHeads(selfAttn, sn, 0, ctx), ropeCos, ropeSin, ctx)
        val nk = selfAttn.rope!!.forwardWithCosSin(projHeads(selfAttn, sn, 1, ctx), ropeCos, ropeSin, ctx)
        val nv = projHeads(selfAttn, sn, 2, ctx)
        val fullK = ops.concat(listOf(selfKIn, ops.unsqueeze(nk, 0)), dim = 2) // [1, nHeads, pos+1, headDim]
        val fullV = ops.concat(listOf(selfVIn, ops.unsqueeze(nv, 0)), dim = 2)
        val afterSelf = ops.add(input, sdpaMerge(selfAttn, q, fullK, fullV, ctx))
        // --- cross-attention: project Q only, attend over the CACHED cross K/V ---
        val cq = projHeads(crossAttn, crossNorm.forward(afterSelf, ctx), 0, ctx)
        val afterCross = ops.add(afterSelf, sdpaMerge(crossAttn, cq, crossKIn, crossVIn, ctx))
        // --- gated SiLU MLP ---
        val h = mlpFc1.forward(mlpNorm.forward(afterCross, ctx), ctx)
        val ld = h.rank - 1
        val mlpOut = mlpFc2.forward(ops.multiply(ops.silu(ops.narrow(h, ld, ffnDim, ffnDim)), ops.narrow(h, ld, 0, ffnDim)), ctx)
        return MoonshinePastKV(ops.add(afterCross, mlpOut), fullK, fullV)
    }

    // Project one of q/k/v (paramIdx 0/1/2) for a single-token input to heads-first
    // [nHeads, 1, headDim]. For seq==1 the reshape equals MHA's swap-then-reshape.
    private fun projHeads(mha: MultiHeadAttention<T, V>, x: Tensor<T, V>, paramIdx: Int, ctx: ExecutionContext): Tensor<T, V> {
        val p = linearProject(ctx.ops, x, mha.params[paramIdx].value) // [1, 1, qDim]
        return ctx.ops.reshape(p, Shape(mha.nHeads, 1, mha.headDim))
    }

    // SDPA over cached K/V then output projection. q [nHeads,1,headDim]; k/v [1,nHeads,S,headDim].
    private fun sdpaMerge(mha: MultiHeadAttention<T, V>, q: Tensor<T, V>, k: Tensor<T, V>, v: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        val ops = ctx.ops
        val o = ops.scaledDotProductAttention(
            query = ops.unsqueeze(q, 0), key = k, value = v,
            mask = null, scale = 1f / sqrt(mha.headDim.toFloat()), causal = false,
        ) // [1, nHeads, 1, headDim]; single query attends to all cached positions → no mask needed
        val merged = ops.reshape(ops.squeeze(o, 0), Shape(1, mha.nHeads * mha.headDim)) // [1, qDim]
        return linearProject(ops, merged, mha.params[3].value) // o_proj (bias=false)
    }

    // This layer must be driven through the two-argument [forward] so the encoder
    // memory is supplied; the single-input entry has no memory to cross-attend to.
    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> =
        error("MoonshineDecoderLayer requires encoder memory: call forward(input, encoderMemory, ctx)")
}

/** `decoder_with_past` layer output + extended self cache (`[1, nHeads, pos+1, headDim]`). */
public class MoonshinePastKV<T : DType, V>(
    public val out: Tensor<T, V>,
    public val newSelfK: Tensor<T, V>,
    public val newSelfV: Tensor<T, V>,
)

/** One layer's output plus its self/cross K/V (`[nHeads, seq, headDim]`), for prefill KV export. */
public class MoonshineLayerKV<T : DType, V>(
    public val out: Tensor<T, V>,
    public val selfK: Tensor<T, V>,
    public val selfV: Tensor<T, V>,
    public val crossK: Tensor<T, V>,
    public val crossV: Tensor<T, V>,
)

/**
 * Full Moonshine decoder: N [MoonshineDecoderLayer]s + final LayerNorm + `lm_head`.
 * Input is `inputs_embeds` `[·, seq, dim]` (embedding lookup is host-side); output is
 * logits `[·, seq, vocab]`.
 */
public class MoonshineDecoderModel<T : DType, V>(
    private val cfg: MoonshineConfig,
    private val dtype: KClass<T>,
) : Module<T, V>() {

    override val name: String = "moonshine_decoder"

    private val layers: List<MoonshineDecoderLayer<T, V>> =
        (0 until cfg.decoderLayers).map { MoonshineDecoderLayer<T, V>(cfg, it, dtype) }

    private val outNorm = LayerNormalization<T, V>(
        normalizedShape = intArrayOf(cfg.dim),
        eps = cfg.layerNormEps.toDouble(),
        name = "dec_out_norm",
        dtype = dtype,
    )

    // lm_head vocab projection [vocab, dim], no bias. Moonshine's `proj_out` is TIED to
    // `decoder.embed_tokens` (the checkpoint stores only embed_tokens; there is no separate
    // proj_out tensor) — the weight baker feeds embed_tokens here. logits = hidden @ Wᵀ.
    private val lmHead = VoidDense<T, V>("lm_head", cfg.vocabSize, cfg.dim, dtype, addBias = false)

    override val modules: List<Module<T, V>> = layers + listOf(outNorm, lmHead)

    /**
     * Prefill/decode forward over [inputsEmbeds] `[·, seq, dim]` attending to
     * [encoderMemory] `[·, frames, dim]`; returns logits `[·, seq, vocab]`.
     */
    public fun forward(
        inputsEmbeds: Tensor<T, V>,
        encoderMemory: Tensor<T, V>,
        ctx: ExecutionContext,
    ): Tensor<T, V> {
        val memory = encoderMemory.bind(ctx)
        var h = inputsEmbeds.bind(ctx)
        for (layer in layers) h = layer.forward(h, memory, ctx)
        return lmHead.forward(outNorm.forward(h, ctx), ctx)
    }

    /**
     * Prefill export forward: returns logits plus the per-layer self/cross K/V as batched
     * tensors (`selfK/V [1, nHeads, seq, headDim]`, `crossK/V [1, nHeads, frames, headDim]`) —
     * matching the board's `decoder` graph outputs. Trace this to emit the KV-cached prefill
     * graph; the returned tensors, being unconsumed, become graph outputs.
     */
    public fun forwardPrefill(
        inputsEmbeds: Tensor<T, V>,
        encoderMemory: Tensor<T, V>,
        ctx: ExecutionContext,
    ): MoonshinePrefillOutput<T, V> {
        val ops = ctx.ops
        val memory = encoderMemory.bind(ctx)
        var h = inputsEmbeds.bind(ctx)
        val selfK = ArrayList<Tensor<T, V>>(layers.size)
        val selfV = ArrayList<Tensor<T, V>>(layers.size)
        val crossK = ArrayList<Tensor<T, V>>(layers.size)
        val crossV = ArrayList<Tensor<T, V>>(layers.size)
        for (layer in layers) {
            val kv = layer.forwardWithKV(h, memory, ctx)
            h = kv.out
            // add the batch dim so the exported shapes match the board's [1, nHeads, ·, headDim].
            selfK += ops.unsqueeze(kv.selfK, 0)
            selfV += ops.unsqueeze(kv.selfV, 0)
            crossK += ops.unsqueeze(kv.crossK, 0)
            crossV += ops.unsqueeze(kv.crossV, 0)
        }
        val logits = lmHead.forward(outNorm.forward(h, ctx), ctx)
        return MoonshinePrefillOutput(logits, selfK, selfV, crossK, crossV)
    }

    /**
     * `decoder_with_past` graph: one new [tokenEmbed] `[1,1,dim]` at [position], the incoming
     * per-layer self cache ([selfKIn]/[selfVIn]) and the fixed cross cache ([crossKIn]/[crossVIn])
     * → logits `[1,1,vocab]` + the per-layer extended self cache. Trace this to emit the second
     * board graph; drive it in a loop after [forwardPrefill] for autoregressive decode.
     */
    public fun forwardWithPast(
        tokenEmbed: Tensor<T, V>,
        ropeCos: Tensor<T, V>,
        ropeSin: Tensor<T, V>,
        selfKIn: List<Tensor<T, V>>,
        selfVIn: List<Tensor<T, V>>,
        crossKIn: List<Tensor<T, V>>,
        crossVIn: List<Tensor<T, V>>,
        ctx: ExecutionContext,
    ): MoonshineWithPastOutput<T, V> {
        val ops = ctx.ops
        var h = tokenEmbed.bind(ctx)
        val nsk = ArrayList<Tensor<T, V>>(layers.size)
        val nsv = ArrayList<Tensor<T, V>>(layers.size)
        for ((i, layer) in layers.withIndex()) {
            val r = layer.forwardWithPast(h, ropeCos, ropeSin, selfKIn[i], selfVIn[i], crossKIn[i], crossVIn[i], ctx)
            h = r.out
            // The extended cache also feeds this layer's SDPA, so it is not a graph sink. Route the
            // exported copy through a shape-preserving reshape so it becomes a distinct output node
            // (mirrors how the prefill graph's unsqueeze creates its KV output sinks). Identity at
            // runtime; the compiler folds it.
            nsk += ops.reshape(r.newSelfK, r.newSelfK.shape)
            nsv += ops.reshape(r.newSelfV, r.newSelfV.shape)
        }
        val logits = lmHead.forward(outNorm.forward(h, ctx), ctx)
        return MoonshineWithPastOutput(logits, nsk, nsv)
    }

    /**
     * Build the RoPE cos/sin table tensors `[seqLen, headDim]` for [position], to feed
     * [forwardWithPast]. The runtime supplies these because the decode position is dynamic
     * (no in-graph gather); all layers share RoPE config so layer 0's tables apply to every layer.
     */
    public fun buildRopeCosSin(position: Int, seqLen: Int, ctx: ExecutionContext): Pair<Tensor<T, V>, Tensor<T, V>> {
        val (c, s) = layers[0].selfRope.buildInterleavedCosSin(position, seqLen)
        val shape = Shape(seqLen, cfg.headDim)
        return ctx.fromFloatArray<T, V>(shape, dtype, c) to ctx.fromFloatArray<T, V>(shape, dtype, s)
    }

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> =
        error("MoonshineDecoderModel requires encoder memory: call forward(inputsEmbeds, encoderMemory, ctx)")
}

/** `decoder_with_past` outputs: logits + per-layer extended self K/V. */
public class MoonshineWithPastOutput<T : DType, V>(
    public val logits: Tensor<T, V>,
    public val selfK: List<Tensor<T, V>>,
    public val selfV: List<Tensor<T, V>>,
)

/** Prefill graph outputs: logits + per-layer self/cross K/V (batched, board layout). */
public class MoonshinePrefillOutput<T : DType, V>(
    public val logits: Tensor<T, V>,
    public val selfK: List<Tensor<T, V>>,
    public val selfV: List<Tensor<T, V>>,
    public val crossK: List<Tensor<T, V>>,
    public val crossV: List<Tensor<T, V>>,
)

/**
 * Build the Moonshine-tiny decoder in the NN DSL. Mirrors [moonshineEncoder]; returns
 * the concrete [MoonshineDecoderModel] so callers can use its two-argument
 * `forward(inputsEmbeds, encoderMemory, ctx)`.
 */
public fun <T : DType, V> moonshineDecoder(
    cfg: MoonshineConfig,
    dtype: KClass<T>,
): MoonshineDecoderModel<T, V> = MoonshineDecoderModel(cfg, dtype)
