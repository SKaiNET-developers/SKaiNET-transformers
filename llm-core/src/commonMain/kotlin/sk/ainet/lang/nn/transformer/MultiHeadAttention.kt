package sk.ainet.lang.nn.transformer

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.normalization.RMSNormalization
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
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
    public val slidingWindow: Int? = null
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
            Any::class as KClass<T>
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
            Any::class as KClass<T>
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
        RMSNormalization(intArrayOf(headDim), name = "$name.q_norm", unitOffset = qkNormUnitOffset)
    } else null

    public val kNorm: RMSNormalization<T, V>? = if (qkNorm) {
        RMSNormalization(intArrayOf(headDim), name = "$name.k_norm", unitOffset = qkNormUnitOffset)
    } else null

    @Suppress("UNCHECKED_CAST")
    override val modules: List<Module<T, V>>
        get() = buildList {
            if (qNorm != null) add(qNorm)
            if (kNorm != null) add(kNorm)
            if (rope != null) add(rope as Module<T, V>)
            if (kvCache != null) add(kvCache as Module<T, V>)
        }

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        val ops = ctx.ops
        val scale = 1.0f / sqrt(headDim.toFloat())

        val wQ = params[qWIdx].value
        val wK = params[kWIdx].value
        val wV = params[vWIdx].value
        val wO = params[oWIdx].value

        // Project Q, K, V: input @ W^T (+ bias if enabled).
        // linearProject handles both the stock [out, in] layout and the
        // [in, out] pre-transposed layout produced by MemSeg conversion.
        var q = linearProject(ops, input, wQ)
        var k = linearProject(ops, input, wK)
        var v = linearProject(ops, input, wV)

        if (bias) {
            q = ops.add(q, params[qWIdx + 1].value)
            k = ops.add(k, params[kWIdx + 1].value)
            v = ops.add(v, params[vWIdx + 1].value)
        }

        // Reshape to multi-head: [seqLen, dim] → [nHeads, seqLen, headDim]
        val seqLen = if (input.rank >= 2) input.shape[input.rank - 2] else 1
        q = ops.reshape(q, Shape(nHeads, seqLen, headDim))
        k = ops.reshape(k, Shape(nKVHeads, seqLen, headDim))
        val vReshaped = ops.reshape(v, Shape(nKVHeads, seqLen, headDim))

        // Optional QK-Norm
        if (qNorm != null && kNorm != null) {
            q = qNorm.forward(q, ctx)
            k = kNorm.forward(k, ctx)
        }

        // Optional RoPE
        val ropeModule = rope
        if (ropeModule != null) {
            val position = kvCache?.position ?: 0
            q = ropeModule.forward(q, position, ctx)
            k = ropeModule.forward(k, position, ctx)
        }

        // Optional KV Cache update
        val (fullK, fullV) = if (kvCache != null) {
            kvCache!!.update(k, vReshaped, ctx)
        } else {
            k to vReshaped
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
        val seqKV = kBatched.shape[2]
        val slidingMask = slidingWindow?.let { buildSlidingCausalMask(seqLen, seqKV, it, ctx, qBatched.dtype) }
        val useCausalPath = causal && slidingMask == null

        // Scaled dot-product attention
        val attnOut = ops.scaledDotProductAttention(
            query = qBatched,
            key = kBatched,
            value = vBatched,
            mask = slidingMask,
            scale = scale,
            causal = useCausalPath
        )

        // Remove batch dim and merge heads: [1, nHeads, seqLen, headDim] → [seqLen, qDim]
        val squeezed = ops.squeeze(attnOut, 0)
        val merged = ops.reshape(squeezed, Shape(seqLen, qDim))

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
        return ctx.fromFloatArray(Shape(1, 1, seqQ, seqKV), dtype, data)
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
}
