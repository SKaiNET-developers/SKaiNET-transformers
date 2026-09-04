package sk.ainet.lang.nn.transformer

import sk.ainet.context.ExecutionContext
import sk.ainet.context.schedule.Schedule
import sk.ainet.lang.nn.transformer.schedule.AttentionSchedulePolicy
import sk.ainet.lang.nn.transformer.schedule.plan
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.normalization.RMSNormalization
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.operators.bind
import sk.ainet.lang.types.DType
import kotlin.math.sqrt
import kotlin.reflect.KClass

/**
 * Multi-Head Attention module.
 *
 * Supports:
 * - Grouped-Query Attention (nKVHeads < nHeads)
 * - Optional QK-Norm (per-head RMSNorm on Q and K before attention)
 * - Optional RoPE (rotary position embeddings)
 * - Optional KV Cache (autoregressive decoding)
 * - Causal or bidirectional masking
 * - Encoder-decoder cross-attention via the [forward] entry that takes a
 *   second `encoderMemory` tensor. In cross-attention mode Q is projected
 *   from the decoder hidden state, K and V from the encoder memory; RoPE
 *   and the KV cache are bypassed; the causal mask is forced off. Cross-K/V
 *   caching is a runtime concern — pass the raw memory tensor each step
 *   (one extra matmul vs. cached projections; acceptable for the typical
 *   ASR / NMT shapes).
 *
 * Weight parameters are exposed as ModuleParameters and loaded via WeightMapper.
 * This module does NOT use Linear submodules because LLM projections often omit bias.
 *
 * @param dim model dimension
 * @param nHeads number of query heads
 * @param nKVHeads number of key-value heads (for GQA; defaults to nHeads)
 * @param causal whether to apply causal masking
 * @param qkNorm whether to apply per-head RMSNorm on Q and K
 * @param bias whether Q/K/V/O projections have bias (true for BERT, false for Llama)
 * @param name module name
 */
/**
 * Attention output plus the reshaped K/V it computed (`[nKVHeads, seq, headDim]`, heads-first,
 * post-RoPE for self-attention). Returned by [MultiHeadAttention.forwardWithKV] for KV-cache export.
 */
public class AttentionKV<T : DType, V>(
    public val output: Tensor<T, V>,
    public val k: Tensor<T, V>,
    public val v: Tensor<T, V>,
)

public class MultiHeadAttention<T : DType, V>(
    public val dim: Int,
    public val nHeads: Int,
    public val nKVHeads: Int = nHeads,
    public val causal: Boolean = true,
    public val qkNorm: Boolean = false,
    /**
     * When `true`, the q_norm/k_norm RMSNorm layers use the Gemma "unit-offset"
     * formula `output = normalized * (1 + weight)`. Required for Gemma
     * checkpoints whose RMSNorm gain tensors are stored centered at zero.
     */
    public val qkNormUnitOffset: Boolean = false,
    /**
     * Epsilon used by the per-head Q / K RMSNorm when [qkNorm] is `true`.
     * Defaults to `1e-5` (the LLaMA convention). Qwen3 metadata says `1e-6`;
     * passing the model's `metadata.rmsNormEps` here matches the legacy
     * `LlamaRuntime.applyPerHeadRMSNorm` and prevents a small but real
     * numerical divergence between paths. See #114.
     */
    public val qkNormEps: Double = 1e-5,
    /**
     * BitNet-style attention sub-layer norm (`attn_sub_norm`): an RMSNorm applied to the merged
     * attention output BEFORE the o_projection (b1.58-2B4T; verified against NeoGPU's reference
     * driver). Off for every other architecture.
     */
    public val attnSubNorm: Boolean = false,
    public val attnSubNormEps: Double = 1e-5,
    /**
     * Explicit scale applied to the attention scores `Q @ K^T`. When `null`
     * (default), uses the standard `1 / sqrt(head_dim)`. Gemma 4 sets this
     * to `1.0f` because q_norm/k_norm have already normalized Q and K to
     * unit-RMS — adding `1 / sqrt(head_dim)` on top makes the softmax
     * over-flat (~uniform), averaging V across positions and producing
     * residual-stream outputs dominated by the most-magnitude embedding
     * (typically BOS). HF Gemma4TextAttention.__init__ sets `self.scaling =
     * 1.0` and passes that to `eager_attention_forward`, bypassing the
     * default `head_dim ** -0.5`.
     */
    public val attentionScale: Float? = null,
    /**
     * When `true`, applies an unscaled per-head RMS norm to V before the
     * attention dot product (`v / sqrt(mean(v²) + eps)` over `head_dim`).
     * HF `Gemma4TextAttention` constructs `v_norm = Gemma4RMSNorm(head_dim,
     * eps=rms_norm_eps, with_scale=False)` — there's no learnable scale, the
     * norm just divides by RMS so V values land at unit RMS regardless of
     * what magnitudes v_proj produced. Without this, V values drift positive
     * and attention output (a weighted sum of V) inherits a strong positive
     * mean bias — the symptom that produces BOS-dominated logits.
     */
    public val vNormNoScale: Boolean = false,
    public val bias: Boolean = false,
    override val name: String = "MultiHeadAttention",
    public var rope: RoPE<T, V>? = null,
    public var kvCache: KVCache<T, V>? = null,
    explicitHeadDim: Int? = null,
    /**
     * Sliding-window size (in tokens). When non-null, each query position only
     * attends to keys within `slidingWindow` positions back (inclusive). Used
     * by Gemma 4 local-attention layers. Null = no windowing.
     */
    public val slidingWindow: Int? = null,
    /**
     * Bounded lookahead (right context) for a windowed attention band, in tokens.
     * With [slidingWindow] = w and rightContext = r, query at absolute position q attends
     * to keys in `[q - w + 1, q + r]` — a **local bidirectional** band. `r = 0` is the
     * classic causal left-only window (Gemma). `r > 0` makes the window non-causal
     * (bounded lookahead), as used by streaming encoders (e.g. Moonshine v2's (16,4)/(16,0)
     * layers). Only meaningful when [slidingWindow] is set; default 0 preserves prior behavior.
     */
    public val rightContext: Int = 0,
    // Logical element type prescribed by the DSL. When provided, placeholder
    // (void) projection/bias weights carry it instead of erasing to Object — so
    // the module can be traced to a graph (and StableHLO) before weights load.
    private val dtype: KClass<T>? = null,
) : Module<T, V>(), ModuleParameters<T, V> {

    init {
        require(rightContext >= 0) { "MultiHeadAttention: rightContext must be >= 0, got $rightContext" }
        if (slidingWindow != null) {
            require(slidingWindow > 0) {
                "MultiHeadAttention: slidingWindow must be > 0 when set, got $slidingWindow"
            }
            // A left-only window (rightContext == 0) is causal by construction. A bounded-lookahead
            // window (rightContext > 0) is a local *bidirectional* band, so causal=false is expected there.
            require(causal || rightContext > 0) {
                "MultiHeadAttention: a left-only sliding window requires causal=true; for a bounded-lookahead " +
                    "(non-causal) window set rightContext > 0"
            }
        } else {
            require(rightContext == 0) {
                "MultiHeadAttention: rightContext is only meaningful with slidingWindow set, got $rightContext"
            }
        }
    }

    public val headDim: Int = explicitHeadDim ?: (dim / nHeads)
    public val qDim: Int = headDim * nHeads
    public val kvDim: Int = headDim * nKVHeads

    // Weight parameters — placeholder tensors, replaced by WeightMapper
    @Suppress("UNCHECKED_CAST")
    private fun voidWeight(paramName: String, rows: Int, cols: Int): ModuleParameter<T, V> {
        val tensor = VoidOpsTensor(
            object : sk.ainet.lang.tensor.data.TensorData<T, V> {
                override val shape = Shape(rows, cols)
                override fun get(vararg indices: Int): V = 0.0f as V
                override fun set(vararg indices: Int, value: V) {}
            },
            (dtype ?: Any::class) as KClass<T>
        )
        return ModuleParameter.WeightParameter(paramName, tensor)
    }

    @Suppress("UNCHECKED_CAST")
    private fun voidBias(paramName: String, size: Int): ModuleParameter<T, V> {
        val tensor = VoidOpsTensor(
            object : sk.ainet.lang.tensor.data.TensorData<T, V> {
                override val shape = Shape(size)
                override fun get(vararg indices: Int): V = 0.0f as V
                override fun set(vararg indices: Int, value: V) {}
            },
            (dtype ?: Any::class) as KClass<T>
        )
        return ModuleParameter.BiasParameter(paramName, tensor)
    }

    // Indices into params list depend on whether bias is enabled.
    // With bias=false: [qW, kW, vW, oW]
    // With bias=true:  [qW, qB, kW, kB, vW, vB, oW, oB]
    private val qWIdx = 0
    private val kWIdx = if (bias) 2 else 1
    private val vWIdx = if (bias) 4 else 2
    private val oWIdx = if (bias) 6 else 3

    override val params: List<ModuleParameter<T, V>> = buildList {
        add(voidWeight("$name.q_proj.weight", qDim, dim))
        if (bias) add(voidBias("$name.q_proj.bias", qDim))
        add(voidWeight("$name.k_proj.weight", kvDim, dim))
        if (bias) add(voidBias("$name.k_proj.bias", kvDim))
        add(voidWeight("$name.v_proj.weight", kvDim, dim))
        if (bias) add(voidBias("$name.v_proj.bias", kvDim))
        add(voidWeight("$name.o_proj.weight", dim, qDim))
        if (bias) add(voidBias("$name.o_proj.bias", dim))
    }

    // Optional QK-Norm layers
    public val qNorm: RMSNormalization<T, V>? = if (qkNorm) {
        RMSNormalization(intArrayOf(headDim), eps = qkNormEps, name = "$name.q_norm", unitOffset = qkNormUnitOffset, dtype = dtype)
    } else null

    public val kNorm: RMSNormalization<T, V>? = if (qkNorm) {
        RMSNormalization(intArrayOf(headDim), eps = qkNormEps, name = "$name.k_norm", unitOffset = qkNormUnitOffset, dtype = dtype)
    } else null

    /** BitNet `attn_sub_norm` over the merged attention output (dim = nHeads·headDim). */
    public val subNorm: RMSNormalization<T, V>? = if (attnSubNorm) {
        RMSNormalization(intArrayOf(qDim), eps = attnSubNormEps, name = "$name.sub_norm", dtype = dtype)
    } else null

    /**
     * How this layer's heads are split across the context schedule's tasks (SKEEP-005). A
     * deployment knob, not a model property: every policy is bit-identical to [AttentionSchedulePolicy.Sequential].
     */
    public var schedulePolicy: AttentionSchedulePolicy = AttentionSchedulePolicy.Auto()

    /** Explicit schedule for this layer; `null` (default) reads `ctx.schedule`. */
    public var schedule: Schedule? = null

    /** Test seam: `false` forces the general `ops.scaledDotProductAttention` path for parity checks. */
    internal var useFusedPaths: Boolean = true

    /** Coordinator-owned per-task `scores` scratch, one slot per schedule unit (grown on demand, never allocated through the context). */
    private var scoresScratch: Array<FloatArray> = emptyArray()

    private fun resolveSchedule(ctx: ExecutionContext): Schedule = schedule ?: ctx.schedule

    private fun scratch(slots: Int, seqKV: Int): Array<FloatArray> {
        val cap = maxOf(seqKV, kvCache?.maxSeqLen ?: 0)
        if (scoresScratch.size < slots || scoresScratch.isEmpty() || scoresScratch[0].size < cap) {
            scoresScratch = Array(maxOf(slots, scoresScratch.size)) { FloatArray(cap) }
        }
        return scoresScratch
    }

    @Suppress("UNCHECKED_CAST")
    override val modules: List<Module<T, V>>
        get() = buildList {
            if (qNorm != null) add(qNorm)
            if (kNorm != null) add(kNorm)
            if (subNorm != null) add(subNorm)
            if (rope != null) add(rope as Module<T, V>)
            if (kvCache != null) add(kvCache as Module<T, V>)
        }

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> =
        attentionImpl(qInput = input, kvInput = input, isCrossAttention = false, ctx = ctx).output

    /**
     * Like [forward], but also returns the reshaped K/V this attention computed
     * (`[nKVHeads, seq, headDim]`, heads-first, post-RoPE for self-attention). Used by
     * encoder-decoder KV-cache export (e.g. the Moonshine prefill graph) to surface the
     * self/cross K/V as graph outputs without duplicating the projection/RoPE/reshape.
     * The K/V are pre-GQA-expansion (one entry per KV head). Cross-attention (`encoderMemory
     * != null`) returns the memory-projected K/V with no RoPE.
     */
    public fun forwardWithKV(
        input: Tensor<T, V>,
        encoderMemory: Tensor<T, V>?,
        ctx: ExecutionContext,
        crossMask: Tensor<T, V>? = null,
    ): AttentionKV<T, V> {
        val boundInput = input.bind(ctx)
        return if (encoderMemory == null) {
            attentionImpl(qInput = boundInput, kvInput = boundInput, isCrossAttention = false, ctx = ctx, wantKV = true)
        } else {
            require(kvCache == null && slidingWindow == null) {
                "MultiHeadAttention.forwardWithKV: cross-attention supports neither kvCache nor slidingWindow."
            }
            val boundMemory = encoderMemory.bind(ctx)
            attentionImpl(qInput = boundInput, kvInput = boundMemory, isCrossAttention = true, ctx = ctx, crossMask = crossMask)
        }
    }

    /**
     * Forward entry that supports cross-attention. When [encoderMemory] is `null`
     * this is equivalent to `forward(input, ctx)` (self-attention). When non-null,
     * Q is projected from [input] and K, V are projected from [encoderMemory];
     * RoPE and the KV cache are skipped and the causal mask is forced off.
     *
     * Cross-attention requires `kvCache == null` and `slidingWindow == null` —
     * both are rejected with a clear error message.
     */
    public fun forward(
        input: Tensor<T, V>,
        encoderMemory: Tensor<T, V>?,
        ctx: ExecutionContext,
    ): Tensor<T, V> {
        if (encoderMemory == null) {
            return forward(input, ctx)
        }
        require(kvCache == null) {
            "MultiHeadAttention: cross-attention (non-null encoderMemory) does not support kvCache. " +
                "Cross-attention K/V should be cached at the runtime layer."
        }
        require(slidingWindow == null) {
            "MultiHeadAttention: cross-attention is incompatible with slidingWindow."
        }
        val boundInput = input.bind(ctx)
        val boundMemory = encoderMemory.bind(ctx)
        return sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, boundInput) {
            attentionImpl(qInput = boundInput, kvInput = boundMemory, isCrossAttention = true, ctx = ctx).output
        }
    }

    private fun attentionImpl(
        qInput: Tensor<T, V>,
        kvInput: Tensor<T, V>,
        isCrossAttention: Boolean,
        ctx: ExecutionContext,
        crossMask: Tensor<T, V>? = null,
        /** The caller reads the returned K/V (export); the in-place cache path cannot serve that. */
        wantKV: Boolean = false,
    ): AttentionKV<T, V> {
        val ops = ctx.ops
        val scale = attentionScale ?: (1.0f / sqrt(headDim.toFloat()))

        // Per-substep diagnostic: dump rms/min/max of intermediate tensors
        // (post-Q-proj, post-K-proj, post-V-proj, post-Q-norm, post-K-norm,
        // post-V-norm, post-RoPE-Q, post-RoPE-K, cached-K, cached-V,
        // post-SDPA, final output) ONLY for blk.0's attention. Gated on a
        // per-call flag set by `HybridTransformerBlock.directForward` when
        // `block.name == "blk.0"` and `GEMMA4_DUMP_MHA=1`. The MHA module
        // itself is named just "attn" (every block's MHA shares that name)
        // so we can't gate on `this.name` alone.
        val mhaDump = MultiHeadAttentionDiag.shouldDumpThisCall

        val wQ = params[qWIdx].value
        val wK = params[kWIdx].value
        val wV = params[vWIdx].value
        val wO = params[oWIdx].value

        // Project Q, K, V: input @ W^T (+ bias if enabled).
        // Q always comes from qInput; in self-attn kvInput === qInput,
        // in cross-attn kvInput is the encoder memory (different seqLen).
        var q = PhaseProfile.time("attn.qkv_proj") { linearProject(ops, qInput, wQ) }
        var k = PhaseProfile.time("attn.qkv_proj") { linearProject(ops, kvInput, wK) }
        var v = PhaseProfile.time("attn.qkv_proj") { linearProject(ops, kvInput, wV) }
        if (mhaDump) {
            mhaDumpStat("[blk.0.mha post-Q-proj      ]", q)
            mhaDumpStat("[blk.0.mha post-K-proj      ]", k)
            mhaDumpStat("[blk.0.mha post-V-proj      ]", v)
        }

        if (bias) {
            q = ops.add(q, params[qWIdx + 1].value)
            k = ops.add(k, params[kWIdx + 1].value)
            v = ops.add(v, params[vWIdx + 1].value)
        }

        // Reshape to multi-head and put heads first.
        //
        // Q/K/V projections produce [seqLen, qDim] where qDim = nHeads*headDim.
        // Row-major flat layout is [s, h, d] → s*qDim + h*headDim + d. SDPA
        // expects [batch, nHeads, seqLen, headDim] — i.e. heads-first layout
        // [h, s, d] → h*seqLen*headDim + s*headDim + d.
        //
        // For seqLen == 1 the two layouts coincide flat-byte-for-flat-byte,
        // so a naked `reshape(t, Shape(nHeads, seqLen, headDim))` was visibly
        // correct in the autoregressive (one-token-per-forward) path. For
        // seqLen > 1 it silently reorders the data: `t.get(h, s, d)` reads
        // `data[h*N*headDim + s*headDim + d]` from a buffer laid out as
        // `s*nHeads*headDim + h*headDim + d`, mixing the rows of head h with
        // values from other heads. That is the root cause of the batched-
        // prefill divergence (commit `bd3eb9c`).
        //
        // The correct transformation needs an explicit dim-0/dim-1 swap.
        // SKaiNET's `ops.transpose` only swaps the LAST two dims, so we
        // can't reuse it here; we materialise the permute via a copy.
        val qSeqLen = if (qInput.rank >= 2) qInput.shape[qInput.rank - 2] else 1
        val kvSeqLen = if (kvInput.rank >= 2) kvInput.shape[kvInput.rank - 2] else 1
        q = PhaseProfile.time("attn.reshape") { swapSeqHeadDims(ops.reshape(q, Shape(qSeqLen, nHeads, headDim)), ctx) }
        k = PhaseProfile.time("attn.reshape") { swapSeqHeadDims(ops.reshape(k, Shape(kvSeqLen, nKVHeads, headDim)), ctx) }
        var vReshaped = PhaseProfile.time("attn.reshape") { swapSeqHeadDims(ops.reshape(v, Shape(kvSeqLen, nKVHeads, headDim)), ctx) }

        // Optional QK-Norm
        if (qNorm != null && kNorm != null) {
            q = qNorm.forward(q, ctx)
            k = kNorm.forward(k, ctx)
            if (mhaDump) {
                mhaDumpStat("[blk.0.mha post-Q-norm      ]", q)
                mhaDumpStat("[blk.0.mha post-K-norm      ]", k)
            }
        }

        // Optional V-Norm (no scale): divide V by per-head RMS so attention
        // output (sum_t softmax × V_t) doesn't inherit a positive mean bias
        // from raw v_proj output. HF Gemma4TextAttention's `v_norm` is
        // `Gemma4RMSNorm(..., with_scale=False)`.
        if (vNormNoScale) {
            val vSq = ops.multiply(vReshaped, vReshaped)
            val vMean = ops.mean(vSq, dim = vReshaped.rank - 1)
            val vRms = ops.unsqueeze(ops.sqrt(ops.addScalar(vMean, 1e-6f)), vReshaped.rank - 1)
            vReshaped = ops.divide(vReshaped, vRms)
            if (mhaDump) mhaDumpStat("[blk.0.mha post-V-norm      ]", vReshaped)
        }

        // Optional RoPE — skip in cross-attention. Cross-attn keys come from
        // encoder memory that's already positioned (its RoPE was applied
        // during encoder self-attention); rotating them again with the
        // decoder's position would corrupt the alignment.
        val ropeModule = rope
        if (ropeModule != null && !isCrossAttention) {
            val position = kvCache?.position ?: 0
            if (mhaDump) println("[blk.0.mha pos=$position]")
            q = PhaseProfile.time("attn.rope") { ropeModule.forward(q, position, ctx) }
            k = PhaseProfile.time("attn.rope") { ropeModule.forward(k, position, ctx) }
            if (mhaDump) {
                mhaDumpStat("[blk.0.mha post-RoPE-Q      ]", q)
                mhaDumpStat("[blk.0.mha post-RoPE-K      ]", k)
            }
        }

        // Optional KV Cache update — only self-attention. Cross-attention
        // K/V cannot share a cache with self-attention (different shapes,
        // different ownership). Caching is the runtime's responsibility
        // for cross-attention and is rejected at the entry above.
        // SKEEP-005 fused eager paths: decode (one query row) and prefill (many rows) run the
        // scalar per-head kernels over a heap view of K/V — no GQA concat, no permute chain, no
        // intermediate tensors — with heads split across the context schedule's tasks. Kept off
        // while recording (the kernels read concrete floats and would detach the tape), for
        // cross-attention and for an explicit cross mask, which stay on the symbolic path below.
        val cache = kvCache
        val eagerFusable = useFusedPaths && !isCrossAttention && !ctx.isRecording && crossMask == null
        if (eagerFusable && !wantKV && cache != null) {
            // In place: the cache writes this step's K/V and hands back a view over its own buffers.
            val view = PhaseProfile.time("attn.kvcache") { cache.updateInPlace(k, vReshaped, ctx) }
            if (view != null) {
                val merged = fusedAttention(q, qSeqLen, view, scale, ctx)
                return finishFused(merged, wO, ctx, mhaDump, k, vReshaped)
            }
        }

        val (fullK, fullV) = if (cache != null && !isCrossAttention) {
            PhaseProfile.time("attn.kvcache") { cache.update(k, vReshaped, ctx) }
        } else {
            k to vReshaped
        }
        if (mhaDump) {
            mhaDumpStat("[blk.0.mha cached-K (full)  ]", fullK)
            mhaDumpStat("[blk.0.mha cached-V (full)  ]", fullV)
        }
        if (eagerFusable) {
            val view = PhaseProfile.time("attn.fused_copy") { copiedView(fullK, fullV) }
            val merged = fusedAttention(q, qSeqLen, view, scale, ctx)
            return finishFused(merged, wO, ctx, mhaDump, fullK, fullV)
        }

        // Grouped-query attention is native to SDPA (SKEEP-005 phase 2): K/V stay
        // [nKVHeads, seq, headDim]; head h reads KV head h / (nHeads / nKVHeads) inside the op.
        // Nothing is narrowed or concatenated, on the tape or in the export.
        // Unsqueeze batch dim for SDPA: [1, nHeads, seqLen, headDim] / [1, nKVHeads, seqKV, headDim]
        val qBatched = ops.unsqueeze(q, 0)
        val kBatched = ops.unsqueeze(fullK, 0)
        val vBatched = ops.unsqueeze(fullV, 0)

        // When sliding-window attention is active, we build a combined
        // causal+window mask ourselves and disable SDPA's built-in causal
        // path (which would otherwise re-mask the same positions).
        // Sliding-window + cross-attention is rejected at the entry; in the
        // cross-attn branch slidingWindow is guaranteed null, so this lambda
        // is never invoked.
        val seqKV = kBatched.shape[2]
        val slidingMask = slidingWindow?.let {
            require(!isCrossAttention) { "slidingWindow + cross-attention is not supported" }
            buildSlidingWindowMask(qSeqLen, seqKV, it, rightContext, ctx, qBatched.dtype)
        }
        // Cross-attention never applies a causal mask — there's no temporal
        // ordering between decoder query positions and encoder memory frames.
        val useCausalPath = !isCrossAttention && causal && slidingMask == null

        // Scaled dot-product attention. `crossMask` (additive, e.g. [1,1,1,seqKV]) masks padded encoder-memory
        // frames when the cross cache is fixed-max-padded (streaming) — used only on the cross path, where
        // slidingMask is guaranteed null.
        val attnOut = ops.scaledDotProductAttention(
            query = qBatched,
            key = kBatched,
            value = vBatched,
            mask = slidingMask ?: crossMask,
            scale = scale,
            causal = useCausalPath
        )
        if (mhaDump) mhaDumpStat("[blk.0.mha post-SDPA        ]", attnOut)

        // Remove batch dim and merge heads.
        //
        // SDPA returns [1, nHeads, seqLen, headDim]. We need [seqLen, qDim].
        // Symmetric inverse of the heads-first permute on the input side:
        // first squeeze the batch dim → [nHeads, seqLen, headDim], then
        // swap dims 0/1 → [seqLen, nHeads, headDim], finally reshape to
        // [seqLen, qDim] (contiguous: row s = concatenation of head 0..N-1
        // for that token). For seqLen == 1 the swap is identity, so this
        // matches the prior naked reshape for the autoregressive case.
        val squeezed = ops.squeeze(attnOut, 0)
        val swappedBack = swapSeqHeadDims(squeezed, ctx)
        // Output sequence length follows Q, not K/V — relevant for cross-attn
        // where kvSeqLen and qSeqLen differ.
        var merged = ops.reshape(swappedBack, Shape(qSeqLen, qDim))

        // Output projection: merged @ wO^T (+ bias if enabled)
        subNorm?.let { merged = it.forward(merged, ctx) }
        var output = linearProject(ops, merged, wO)
        if (bias) {
            output = ops.add(output, params[oWIdx + 1].value)
        }
        return AttentionKV(output, fullK, fullV)
    }

    /**
     * Scheduled fused attention over a heap view of K/V (SKEEP-005). [q] is heads-first
     * `[nHeads, seqQ, headDim]` (post-RoPE); the result is the merged `[seqQ, qDim]` context —
     * what the general SDPA + squeeze + permute + reshape chain produces, with zero intermediate
     * tensors. GQA query head `h` reads KV head `h / (nHeads / nKVHeads)`.
     *
     * Decode (`seqQ == 1`, no window) uses the fused-decode rounding order; everything else the
     * engine SDPA's. Heads are the schedule's units: each task gets its own `scores` scratch and
     * writes a disjoint slice of `out`; nothing inside the region touches the context.
     */
    private fun fusedAttention(q: Tensor<T, V>, seqQ: Int, kv: KVBufferView, scale: Float, ctx: ExecutionContext): Tensor<T, V> {
        val qBuf = PhaseProfile.time("attn.fused_copy") {
            (q.data as? FloatArrayTensorData<*>)?.buffer ?: q.data.copyToFloatArray()
        }
        val nRep = nHeads / nKVHeads
        val out = FloatArray(seqQ * qDim)
        val decode = seqQ == 1 && slidingWindow == null
        val absOffset = kv.length - seqQ
        val window = slidingWindow
        val sched = resolveSchedule(ctx)
        val plan = schedulePolicy.plan(nHeads, nKVHeads, kv.length, sched.parallelism)
        val scratch = scratch(slots = plan?.units ?: 1, seqKV = kv.length)

        fun head(h: Int, scores: FloatArray) {
            val g = h / nRep
            if (decode) {
                ScalarHeadAttentionKernel.decodeHead(qBuf, h * headDim, kv, g, scale, scores, out, h * headDim)
            } else {
                ScalarHeadAttentionKernel.prefillRows(
                    qBuf, qBase = h * seqQ * headDim, qRowStride = headDim, qi0 = 0, qi1 = seqQ,
                    kv = kv, g = g, scale = scale, causal = causal, absOffset = absOffset,
                    window = window, rightContext = rightContext,
                    scores = scores, out = out, outOff = h * headDim, outRowStride = qDim,
                )
            }
        }

        PhaseProfile.time("attn.fused_compute") {
            if (plan == null) {
                val scores = scratch[0]
                for (h in 0 until nHeads) head(h, scores)
            } else {
                sched.forRange(plan.units, plan.grain) { start, end ->
                    // Slot = first unit of this range: distinct per task under any schedule.
                    val scores = scratch[start]
                    for (u in start until end) {
                        val h0 = u * plan.headsPerUnit
                        for (h in h0 until h0 + plan.headsPerUnit) head(h, scores)
                    }
                }
            }
        }
        @Suppress("UNCHECKED_CAST")
        return ctx.fromData(DenseFloatArrayTensorData<T>(Shape(seqQ, qDim), out) as TensorData<T, V>, q.dtype)
    }

    /** Today's copies, for caches that cannot expose their buffers: `[nKVHeads, seqKV, headDim]` heads-first. */
    private fun copiedView(fullK: Tensor<T, V>, fullV: Tensor<T, V>): KVBufferView {
        val seqKV = fullK.shape[fullK.rank - 2]
        val kBuf = (fullK.data as? FloatArrayTensorData<*>)?.buffer ?: fullK.data.copyToFloatArray()
        val vBuf = (fullV.data as? FloatArrayTensorData<*>)?.buffer ?: fullV.data.copyToFloatArray()
        return KVBufferView.contiguous(kBuf, vBuf, seqKV, headDim)
    }

    private fun finishFused(
        merged0: Tensor<T, V>,
        wO: Tensor<T, V>,
        ctx: ExecutionContext,
        mhaDump: Boolean,
        k: Tensor<T, V>,
        v: Tensor<T, V>,
    ): AttentionKV<T, V> {
        val ops = ctx.ops
        var merged = merged0
        subNorm?.let { merged = it.forward(merged, ctx) }
        var output = PhaseProfile.time("attn.o_proj") { linearProject(ops, merged, wO) }
        if (bias) output = ops.add(output, params[oWIdx + 1].value)
        if (mhaDump) mhaDumpStat("[blk.0.mha post-fused        ]", output)
        return AttentionKV(output, k, v)
    }

    /**
     * Build an additive mask tensor of shape `[1, 1, seqQ, seqKV]` where allowed
     * (query, key) cells are 0 and masked cells are a large negative value so
     * the post-softmax attention weight is effectively zero.
     *
     * Query qi at (absolute) position `abs_q = seqKV - seqQ + qi` attends to
     * any key ki at abs position `ki` such that:
     *   abs_q - window < ki <= abs_q
     *
     * The causal bound is subsumed by the sliding bound so this mask also
     * covers the causal case — SDPA's own causal path is disabled when this
     * mask is in use.
     *
     * Assumes keys are laid out in ascending absolute position order (true for
     * [AppendKVCache] and [SlidingWindowKVCache] with the trailing tail).
     */
    private fun buildSlidingWindowMask(
        seqQ: Int,
        seqKV: Int,
        window: Int,
        rightContext: Int,
        ctx: ExecutionContext,
        dtype: KClass<T>
    ): Tensor<T, V> {
        val neg = -1.0e30f
        val data = FloatArray(seqQ * seqKV)
        val qAbsOffset = seqKV - seqQ
        for (qi in 0 until seqQ) {
            val absQ = qAbsOffset + qi
            // Local band: [absQ - window + 1, absQ + rightContext]. rightContext == 0 → causal left window.
            val lower = absQ - window + 1
            val upper = absQ + rightContext
            for (ki in 0 until seqKV) {
                val allowed = ki in lower..upper
                data[qi * seqKV + ki] = if (allowed) 0f else neg
            }
        }
        // Heap-backed wrap — fromFloatArray would copy into a fresh
        // Arena.ofAuto MemorySegment every forward (× layers using the
        // sliding-mask path), and direct memory doesn't pressure the GC.
        // Same root cause as the sliceView leak (commit 319c394).
        val maskShape = Shape(1, 1, seqQ, seqKV)
        return ctx.fromData(
            sk.ainet.lang.tensor.data.DenseFloatArrayTensorData<T>(maskShape, data) as sk.ainet.lang.tensor.data.TensorData<T, V>,
            dtype
        )
    }

    /**
     * Swap dims 0 and 1 of a rank-3 tensor: `[D0, D1, D2]` → `[D1, D0, D2]`.
     *
     * Routes through upstream [TensorOps.permute] (sk.ainet.core ≥ 0.21.0)
     * for the general case. When `D0 == 1` or `D1 == 1` the two layouts
     * coincide flat-byte-for-flat-byte, so we short-circuit to a shape-only
     * reshape — the autoregressive decode path (where seqLen is always 1)
     * pays zero data-movement cost.
     */
    private fun swapSeqHeadDims(t: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        require(t.rank == 3) { "swapSeqHeadDims: expected rank-3 tensor, got rank ${t.rank}" }
        val d0 = t.shape[0]
        val d1 = t.shape[1]
        val d2 = t.shape[2]
        if (d0 == 1 || d1 == 1) {
            return ctx.ops.reshape(t, Shape(d1, d0, d2))
        }
        return ctx.ops.permute(t, intArrayOf(1, 0, 2))
    }


    /** Diagnostic stat dump for MHA substeps. Gated by [MultiHeadAttentionDiag.shouldDumpThisCall];
     *  delegates to the platform diagnostic helper so the multiplatform
     *  metadata compile doesn't see JVM-only formatter / MemorySegment
     *  references. Stats are over the *whole* tensor, not just last
     *  position — different MHA substeps have different shapes. */
    private fun mhaDumpStat(label: String, t: Tensor<T, V>) {
        mhaStatSink?.invoke(label, t)
    }
}

/**
 * Optional diagnostic sink for MHA substep stats — decouples `transformer-core` from llm-core's
 * platform `dumpStats`. Defaults to no-op (diagnostics off); llm-core wires its `dumpStats` into it.
 */
public var mhaStatSink: ((String, Tensor<*, *>) -> Unit)? = null

/**
 * Per-call MHA-substep dump gate. The MHA module itself is named just `"attn"`
 * (every block's MHA shares that name), so we can't gate the dump from inside
 * MHA based on its own `name`. Instead, the calling `HybridTransformerBlock`
 * (which knows it's `blk.0`) sets [shouldDumpThisCall] to `true` immediately
 * before calling MHA's forward, and resets it after. Effective mutex:
 * single-threaded forward path, so a static var is OK.
 */
public object MultiHeadAttentionDiag {
    public var shouldDumpThisCall: Boolean = false
}
