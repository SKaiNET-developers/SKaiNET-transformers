package sk.ainet.lang.nn.transformer

import sk.ainet.context.ExecutionContext
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
    // Logical element type prescribed by the DSL. When provided, placeholder
    // (void) projection/bias weights carry it instead of erasing to Object — so
    // the module can be traced to a graph (and StableHLO) before weights load.
    private val dtype: KClass<T>? = null,
) : Module<T, V>(), ModuleParameters<T, V> {

    init {
        if (slidingWindow != null) {
            require(slidingWindow > 0) {
                "MultiHeadAttention: slidingWindow must be > 0 when set, got $slidingWindow"
            }
            require(causal) {
                "MultiHeadAttention: slidingWindow currently requires causal=true (non-causal windowed attention not supported)"
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

    @Suppress("UNCHECKED_CAST")
    override val modules: List<Module<T, V>>
        get() = buildList {
            if (qNorm != null) add(qNorm)
            if (kNorm != null) add(kNorm)
            if (rope != null) add(rope as Module<T, V>)
            if (kvCache != null) add(kvCache as Module<T, V>)
        }

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> =
        attentionImpl(qInput = input, kvInput = input, isCrossAttention = false, ctx = ctx)

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
            attentionImpl(qInput = boundInput, kvInput = boundMemory, isCrossAttention = true, ctx = ctx)
        }
    }

    private fun attentionImpl(
        qInput: Tensor<T, V>,
        kvInput: Tensor<T, V>,
        isCrossAttention: Boolean,
        ctx: ExecutionContext,
    ): Tensor<T, V> {
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
        var q = linearProject(ops, qInput, wQ)
        var k = linearProject(ops, kvInput, wK)
        var v = linearProject(ops, kvInput, wV)
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
        q = swapSeqHeadDims(ops.reshape(q, Shape(qSeqLen, nHeads, headDim)), ctx)
        k = swapSeqHeadDims(ops.reshape(k, Shape(kvSeqLen, nKVHeads, headDim)), ctx)
        var vReshaped = swapSeqHeadDims(ops.reshape(v, Shape(kvSeqLen, nKVHeads, headDim)), ctx)

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
            q = ropeModule.forward(q, position, ctx)
            k = ropeModule.forward(k, position, ctx)
            if (mhaDump) {
                mhaDumpStat("[blk.0.mha post-RoPE-Q      ]", q)
                mhaDumpStat("[blk.0.mha post-RoPE-K      ]", k)
            }
        }

        // Optional KV Cache update — only self-attention. Cross-attention
        // K/V cannot share a cache with self-attention (different shapes,
        // different ownership). Caching is the runtime's responsibility
        // for cross-attention and is rejected at the entry above.
        val (fullK, fullV) = if (kvCache != null && !isCrossAttention) {
            kvCache!!.update(k, vReshaped, ctx)
        } else {
            k to vReshaped
        }
        if (mhaDump) {
            mhaDumpStat("[blk.0.mha cached-K (full)  ]", fullK)
            mhaDumpStat("[blk.0.mha cached-V (full)  ]", fullV)
        }

        // Expand KV heads for GQA if needed
        val expandedK = if (nKVHeads < nHeads) repeatKVHeads(fullK, nHeads / nKVHeads, ops) else fullK
        val expandedV = if (nKVHeads < nHeads) repeatKVHeads(fullV, nHeads / nKVHeads, ops) else fullV

        // Unsqueeze batch dim for SDPA: [1, nHeads, seqLen, headDim]
        val qBatched = ops.unsqueeze(q, 0)
        val kBatched = ops.unsqueeze(expandedK, 0)
        val vBatched = ops.unsqueeze(expandedV, 0)

        // When sliding-window attention is active, we build a combined
        // causal+window mask ourselves and disable SDPA's built-in causal
        // path (which would otherwise re-mask the same positions).
        // Sliding-window + cross-attention is rejected at the entry; in the
        // cross-attn branch slidingWindow is guaranteed null, so this lambda
        // is never invoked.
        val seqKV = kBatched.shape[2]
        val slidingMask = slidingWindow?.let {
            require(!isCrossAttention) { "slidingWindow + cross-attention is not supported" }
            buildSlidingCausalMask(qSeqLen, seqKV, it, ctx, qBatched.dtype)
        }
        // Cross-attention never applies a causal mask — there's no temporal
        // ordering between decoder query positions and encoder memory frames.
        val useCausalPath = !isCrossAttention && causal && slidingMask == null

        // Scaled dot-product attention
        val attnOut = ops.scaledDotProductAttention(
            query = qBatched,
            key = kBatched,
            value = vBatched,
            mask = slidingMask,
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
        val merged = ops.reshape(swappedBack, Shape(qSeqLen, qDim))

        // Output projection: merged @ wO^T (+ bias if enabled)
        var output = linearProject(ops, merged, wO)
        if (bias) {
            output = ops.add(output, params[oWIdx + 1].value)
        }
        return output
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
    private fun buildSlidingCausalMask(
        seqQ: Int,
        seqKV: Int,
        window: Int,
        ctx: ExecutionContext,
        dtype: KClass<T>
    ): Tensor<T, V> {
        val neg = -1.0e30f
        val data = FloatArray(seqQ * seqKV)
        val qAbsOffset = seqKV - seqQ
        for (qi in 0 until seqQ) {
            val absQ = qAbsOffset + qi
            val lowerExclusive = absQ - window
            for (ki in 0 until seqKV) {
                val allowed = ki in (lowerExclusive + 1)..absQ
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

    private fun repeatKVHeads(t: Tensor<T, V>, repeats: Int, ops: sk.ainet.lang.tensor.ops.TensorOps): Tensor<T, V> {
        if (repeats == 1) return t
        // Repeat each KV head individually so head mapping matches GQA:
        // head h uses KV head h/repeats → [kv0]*repeats ++ [kv1]*repeats ++ ...
        val expanded = mutableListOf<Tensor<T, V>>()
        for (h in 0 until nKVHeads) {
            val headSlice = ops.narrow(t, 0, h, 1) // [1, seqLen, headDim]
            repeat(repeats) { expanded.add(headSlice) }
        }
        return ops.concat(expanded, dim = 0)
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
