package sk.ainet.models.whisper

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.layers.Embedding
import sk.ainet.lang.nn.layers.EmbeddingAdapter
import sk.ainet.lang.nn.normalization.LayerNormalization
import sk.ainet.lang.nn.transformer.VoidDense
import sk.ainet.lang.nn.transformer.linearProject
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.operators.bind
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Whisper DECODER in the SKaiNET NN DSL, shaped for the fixed-masked-KV
 * prefill/step export validated on the Android/Vulkan pipeline:
 *
 * - **prefill** — decode the S-token prompt in one shot; besides logits it
 *   surfaces per-layer self K/V zero-padded to `maxPositions` and the cross K/V
 *   projected once from the encoder features (device-resident thereafter).
 * - **step** — one token at runtime position `pos`. The self-KV cache write is
 *   pure arithmetic (`sk·(1−wf) + k·wf` with a host-computed one-hot `wf`), and
 *   attention masking is a host-computed additive f32 `add_mask` — no `i1`
 *   compare/`Where`, which do not codegen on Vulkan/SPIR-V (the hard-won rule
 *   from the ONNX pipeline).
 *
 * KV tensors are FLAT `[1, seq, dim]` (heads split inside the graph where
 * needed) — the layout of the proven contract. All shapes static. The token
 * embedding is TIED to the output projection (`logits = h @ embed_tokensᵀ`).
 */
public class WhisperDecoderLayer<T : DType, V>(
    cfg: WhisperConfig,
    layer: Int,
    dtype: KClass<T>,
) : Module<T, V>() {

    override val name: String = "dec.$layer"

    private val dim = cfg.dim
    private val audioCtx = cfg.audioCtx

    private val selfNorm = LayerNormalization<T, V>(
        normalizedShape = intArrayOf(dim), eps = cfg.layerNormEps.toDouble(),
        name = "dec.$layer.self_attn_norm", dtype = dtype,
    )
    public val selfAttn: WhisperAttentionProjections<T, V> =
        WhisperAttentionProjections("dec.$layer.self_attn", dim, cfg.nHeads, cfg.headDim, dtype)
    private val crossNorm = LayerNormalization<T, V>(
        normalizedShape = intArrayOf(dim), eps = cfg.layerNormEps.toDouble(),
        name = "dec.$layer.cross_attn_norm", dtype = dtype,
    )
    public val crossAttn: WhisperAttentionProjections<T, V> =
        WhisperAttentionProjections("dec.$layer.cross_attn", dim, cfg.nHeads, cfg.headDim, dtype)
    private val mlpNorm = LayerNormalization<T, V>(
        normalizedShape = intArrayOf(dim), eps = cfg.layerNormEps.toDouble(),
        name = "dec.$layer.mlp_norm", dtype = dtype,
    )
    private val fc1 = VoidDense<T, V>("dec.$layer.fc1", cfg.ffnDim, dim, dtype, addBias = true)
    private val fc2 = VoidDense<T, V>("dec.$layer.fc2", dim, cfg.ffnDim, dtype, addBias = true)

    override val modules: List<Module<T, V>> =
        listOf(selfNorm, selfAttn, crossNorm, crossAttn, mlpNorm, fc1, fc2)

    private fun kOf(attn: WhisperAttentionProjections<T, V>, x: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> =
        linearProject(ctx.ops, x, attn.kProj.params[0].value) // k_proj has no bias

    /**
     * Prefill: [x] `[1, S, dim]` with additive [causalMask] `[1, 1, S, S]`,
     * cross-attending [memory] `[1, audioCtx, dim]`. Returns the layer output
     * plus self K/V padded to [maxPositions] (`[1, maxP, dim]`) and cross K/V
     * (`[1, audioCtx, dim]`).
     */
    public fun forwardPrefill(
        x: Tensor<T, V>,
        memory: Tensor<T, V>,
        causalMask: Tensor<T, V>,
        seq: Int,
        maxPositions: Int,
        zeroPad: Tensor<T, V>, // constant zeros [1, maxPositions - seq, dim]
        ctx: ExecutionContext,
    ): WhisperLayerPrefill<T, V> {
        val ops = ctx.ops
        val sn = selfNorm.forward(x, ctx)
        val sk = kOf(selfAttn, sn, ctx)                       // [1, S, dim]
        val sv = linearProjectBias(selfAttn.vProj, sn, ctx)
        val afterSelf = ops.add(x, selfAttn.attend(sn, sk, sv, seq, seq, causalMask, ctx))

        val cn = crossNorm.forward(afterSelf, ctx)
        val ck = kOf(crossAttn, memory, ctx)                  // [1, audioCtx, dim]
        val cv = linearProjectBias(crossAttn.vProj, memory, ctx)
        val afterCross = ops.add(afterSelf, crossAttn.attend(cn, ck, cv, seq, audioCtx, null, ctx))

        val h = ops.gelu(linearProjectBias(fc1, mlpNorm.forward(afterCross, ctx), ctx))
        val out = ops.add(afterCross, linearProjectBias(fc2, h, ctx))

        val skPadded = ops.concat(listOf(sk, zeroPad), dim = 1) // [1, maxP, dim]
        val svPadded = ops.concat(listOf(sv, zeroPad), dim = 1)
        return WhisperLayerPrefill(out, skPadded, svPadded, ck, cv)
    }

    /**
     * Step: one token [x] `[1, 1, dim]`. [wf] is the host one-hot write vector
     * `[1, maxP, 1]` (`wf[i] = i==pos`), [addMask] the host additive attention
     * mask `[1, 1, 1, maxP]` (`0` for `i<=pos`, `-1e4` beyond). Self K/V caches
     * come in flat `[1, maxP, dim]`; cross K/V `[1, audioCtx, dim]` pass through
     * unchanged on the host side.
     */
    public fun forwardStep(
        x: Tensor<T, V>,
        addMask: Tensor<T, V>,
        wf: Tensor<T, V>,
        selfKIn: Tensor<T, V>,
        selfVIn: Tensor<T, V>,
        crossKIn: Tensor<T, V>,
        crossVIn: Tensor<T, V>,
        maxPositions: Int,
        ctx: ExecutionContext,
    ): WhisperLayerStep<T, V> {
        val ops = ctx.ops
        val sn = selfNorm.forward(x, ctx)
        val kNew = kOf(selfAttn, sn, ctx)                     // [1, 1, dim]
        val vNew = linearProjectBias(selfAttn.vProj, sn, ctx)
        val keep = ops.rsubScalar(1, wf)                      // [1, maxP, 1]
        // Arithmetic cache write at pos: uk = sk*(1-wf) + kNew*wf (broadcast over dim/seq).
        val uk = ops.add(ops.multiply(selfKIn, keep), ops.multiply(kNew, wf))
        val uv = ops.add(ops.multiply(selfVIn, keep), ops.multiply(vNew, wf))
        val afterSelf = ops.add(x, selfAttn.attend(sn, uk, uv, 1, maxPositions, addMask, ctx))

        val cn = crossNorm.forward(afterSelf, ctx)
        val afterCross = ops.add(afterSelf, crossAttn.attend(cn, crossKIn, crossVIn, 1, audioCtx, null, ctx))

        val h = ops.gelu(linearProjectBias(fc1, mlpNorm.forward(afterCross, ctx), ctx))
        val out = ops.add(afterCross, linearProjectBias(fc2, h, ctx))
        return WhisperLayerStep(out, uk, uv)
    }

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> =
        error("WhisperDecoderLayer is driven via forwardPrefill/forwardStep")
}

public class WhisperLayerPrefill<T : DType, V>(
    public val out: Tensor<T, V>,
    public val selfK: Tensor<T, V>,
    public val selfV: Tensor<T, V>,
    public val crossK: Tensor<T, V>,
    public val crossV: Tensor<T, V>,
)

public class WhisperLayerStep<T : DType, V>(
    public val out: Tensor<T, V>,
    public val newSelfK: Tensor<T, V>,
    public val newSelfV: Tensor<T, V>,
)

public class WhisperDecoderModel<T : DType, V>(
    private val cfg: WhisperConfig,
    private val dtype: KClass<T>,
) : Module<T, V>() {

    override val name: String = "whisper_decoder"

    private fun void(vararg dims: Int): Tensor<T, V> = VoidOpsTensor(
        object : TensorData<T, V> {
            override val shape: Shape = Shape(*dims)
            @Suppress("UNCHECKED_CAST")
            override fun get(vararg indices: Int): V = 0.0f as V
            override fun set(vararg indices: Int, value: V) {}
        },
        dtype,
    )

    // Tied token embedding: gather source AND lm_head weight (one tensor, one irpa entry).
    private val embedTokens = Embedding<T, V>(
        numEmbeddings = cfg.vocabSize, embeddingDim = cfg.dim,
        initWeight = void(cfg.vocabSize, cfg.dim), name = "embed_tokens",
    )
    private val embedAdapter = EmbeddingAdapter(embedTokens)

    // Learned positions [maxTextPositions, dim]; gathered at runtime pos in step,
    // statically narrowed in prefill.
    private val posEmbed = Embedding<T, V>(
        numEmbeddings = cfg.maxTextPositions, embeddingDim = cfg.dim,
        initWeight = void(cfg.maxTextPositions, cfg.dim), name = "pos_embed",
    )
    private val posAdapter = EmbeddingAdapter(posEmbed)

    public val layers: List<WhisperDecoderLayer<T, V>> =
        (0 until cfg.decoderLayers).map { WhisperDecoderLayer(cfg, it, dtype) }

    private val lnFinal = LayerNormalization<T, V>(
        normalizedShape = intArrayOf(cfg.dim), eps = cfg.layerNormEps.toDouble(),
        name = "dec_ln", dtype = dtype,
    )

    override val modules: List<Module<T, V>> =
        listOf<Module<T, V>>(embedAdapter, posAdapter) + layers + listOf(lnFinal)

    private fun logitsOf(h: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> =
        linearProject(ctx.ops, lnFinal.forward(h, ctx), embedTokens.params[0].value)

    /**
     * Prefill graph: [promptIds] `[S]` i32 + [memory] `[1, audioCtx, dim]` →
     * logits `[1, S, vocab]` + per-layer padded self K/V + cross K/V.
     * [causalMask] `[1,1,S,S]` and [zeroPad] `[1, maxP-S, dim]` are embedded
     * constants supplied by the caller (eager runtime or export harness).
     */
    public fun forwardPrefill(
        promptIds: Tensor<T, V>,
        memory: Tensor<T, V>,
        causalMask: Tensor<T, V>,
        zeroPad: Tensor<T, V>,
        seq: Int,
        maxPositions: Int,
        ctx: ExecutionContext,
    ): WhisperPrefillOutput<T, V> {
        val ops = ctx.ops
        val mem = memory.bind(ctx)
        val tok = embedAdapter.forward(promptIds, ctx)                       // [S, dim]
        val pos = ops.narrow(posEmbed.params[0].value, 0, 0, seq)           // [S, dim]
        var h = ops.reshape(ops.add(tok, pos), Shape(1, seq, cfg.dim))
        val selfK = ArrayList<Tensor<T, V>>(layers.size)
        val selfV = ArrayList<Tensor<T, V>>(layers.size)
        val crossK = ArrayList<Tensor<T, V>>(layers.size)
        val crossV = ArrayList<Tensor<T, V>>(layers.size)
        for (layer in layers) {
            val r = layer.forwardPrefill(h, mem, causalMask, seq, maxPositions, zeroPad, ctx)
            h = r.out
            selfK += r.selfK; selfV += r.selfV; crossK += r.crossK; crossV += r.crossV
        }
        return WhisperPrefillOutput(logitsOf(h, ctx), selfK, selfV, crossK, crossV)
    }

    /**
     * Step graph: [tokId] `[1]` i32, [posId] `[1]` i32, host masks, per-layer
     * KV caches → logits `[1, 1, vocab]` + per-layer updated self K/V.
     */
    public fun forwardStep(
        tokId: Tensor<T, V>,
        posId: Tensor<T, V>,
        addMask: Tensor<T, V>,
        wf: Tensor<T, V>,
        selfKIn: List<Tensor<T, V>>,
        selfVIn: List<Tensor<T, V>>,
        crossKIn: List<Tensor<T, V>>,
        crossVIn: List<Tensor<T, V>>,
        maxPositions: Int,
        ctx: ExecutionContext,
    ): WhisperStepOutput<T, V> {
        val ops = ctx.ops
        val tok = embedAdapter.forward(tokId, ctx)                          // [1, dim]
        val pos = posAdapter.forward(posId, ctx)                            // [1, dim]
        var h = ops.reshape(ops.add(tok, pos), Shape(1, 1, cfg.dim))
        val nk = ArrayList<Tensor<T, V>>(layers.size)
        val nv = ArrayList<Tensor<T, V>>(layers.size)
        for ((i, layer) in layers.withIndex()) {
            val r = layer.forwardStep(
                h, addMask, wf, selfKIn[i], selfVIn[i], crossKIn[i], crossVIn[i], maxPositions, ctx,
            )
            h = r.out
            // The updated cache also feeds this layer's SDPA, so route the exported copy
            // through a shape-preserving reshape to make it a distinct output node
            // (moonshine forwardWithPast precedent; identity at runtime).
            nk += ops.reshape(r.newSelfK, r.newSelfK.shape)
            nv += ops.reshape(r.newSelfV, r.newSelfV.shape)
        }
        return WhisperStepOutput(logitsOf(h, ctx), nk, nv)
    }
}

public class WhisperPrefillOutput<T : DType, V>(
    public val logits: Tensor<T, V>,
    public val selfK: List<Tensor<T, V>>,
    public val selfV: List<Tensor<T, V>>,
    public val crossK: List<Tensor<T, V>>,
    public val crossV: List<Tensor<T, V>>,
)

public class WhisperStepOutput<T : DType, V>(
    public val logits: Tensor<T, V>,
    public val selfK: List<Tensor<T, V>>,
    public val selfV: List<Tensor<T, V>>,
)
