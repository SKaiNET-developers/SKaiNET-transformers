package sk.ainet.models.gemma

import sk.ainet.apps.llm.HybridTransformerBlock
import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.dsl.NeuralNetworkDslImpl
import sk.ainet.lang.nn.dsl.StageImpl
import sk.ainet.lang.nn.dsl.embedding
import sk.ainet.lang.nn.dsl.geGluFFN
import sk.ainet.lang.nn.dsl.multiHeadAttention
import sk.ainet.lang.nn.dsl.residual
import sk.ainet.lang.nn.dsl.rmsNorm
import sk.ainet.lang.nn.dsl.sequential
import sk.ainet.lang.nn.layers.EmbeddingAdapter
import sk.ainet.lang.nn.normalization.RMSNormalization
import sk.ainet.lang.nn.transformer.LayerScalarMul
import sk.ainet.lang.nn.transformer.OwnerReadOnlyKVCache
import sk.ainet.lang.nn.transformer.PositionalKVCache
import sk.ainet.lang.nn.transformer.RoPEMode
import sk.ainet.lang.nn.transformer.RoPEScaling
import sk.ainet.lang.nn.transformer.VoidDense
import sk.ainet.lang.types.DType

/**
 * Gemma 4 architecture defined via the network DSL.
 *
 * Phase 5b: every Gemma 4 architectural feature the DSL has primitives for
 * is now expressed here, walking [Gemma4ModelMetadata.layerTypes] to pick
 * per-layer configuration:
 *
 * - **Full-attention layers** use proportional (NTK-aware) RoPE with the
 *   global base / scaling factor, the global `head_dim`, and
 *   `partialRotaryFactor = 0.5` (Gemma 4 convention).
 * - **Sliding-attention layers** use the standard RoPE base, the sliding
 *   `head_dim`, full-rotation, and a sliding-window mask of
 *   [Gemma4ModelMetadata.slidingWindow] tokens on the attention step.
 * - **Per-layer FFN width** via [Gemma4ModelMetadata.getIntermediateSize].
 * - **Shared KV cache** across the trailing `kvSharedLayers` layers: an
 *   [AppendKVCache] is built on the owner layer, and each follower gets a
 *   [SharedKVCache] that forwards to it. Follower shape must match the
 *   owner's, which holds for Gemma 4 because shared layers all fall in the
 *   same global/sliding group as the owner.
 *
 * Gemma-specific FFN: GELU-gated (via [geGluFFN]) instead of SwiGLU.
 *
 * @param metadata the parsed Gemma 4 model metadata.
 * @param maxInferenceLen upper bound on sequence length the cos/sin tables
 *   are sized for (and the KV cache capacity). Default caps to 4 096 so
 *   long-context (131 072) checkpoints don't precompute gigabytes of RoPE
 *   tables up front.
 */
public inline fun <reified T : DType, V> gemmaNetwork(
    metadata: Gemma4ModelMetadata,
    maxInferenceLen: Int = minOf(metadata.contextLength, 4096),
    qkNorm: Boolean = true,
    sandwichNorms: Boolean = true,
    layerOutputScale: Boolean = true,
    ple: Boolean = false,
    pleSideChannelOnly: Boolean = false
): Module<T, V> = gemmaNetwork<T, V>(
    metadata, T::class, maxInferenceLen, qkNorm, sandwichNorms,
    layerOutputScale, ple, pleSideChannelOnly
)

/**
 * Non-reified variant of [gemmaNetwork] that takes an explicit `dtype` [KClass].
 * Lets non-reified callers (e.g. `Gemma4Ingestion<T>`) build the DSL network
 * without propagating `reified` through their API.
 *
 * @param qkNorm whether to apply per-head RMSNorm to Q and K before RoPE. True
 *   for real Gemma 4 checkpoints (the GGUF carries `attn_q_norm` /
 *   `attn_k_norm` weights). Callers with synthetic fixtures that don't
 *   provide those weights — or that want to match the hand-coded
 *   [Gemma4AttentionBackend] which never applied QK-Norm — can set this to
 *   false. [GemmaNetworkLoader.fromWeights] auto-detects based on presence
 *   of `blk.*.attn_q_norm.weight` in the tensors map.
 * @param sandwichNorms whether to apply Gemma 2-style post-norms after the
 *   attention and FFN sub-blocks (`post_attention_norm` / `post_ffw_norm`),
 *   before the residual add. True for real Gemma 4 checkpoints. Auto-detected
 *   at `GemmaNetworkLoader.fromWeights` the same way as `qkNorm`.
 * @param layerOutputScale whether to apply `blk.N.layer_output_scale.weight`
 *   (a scalar `[1]`) at the very end of each block. Corresponds to HF's
 *   `self.layer_scalar *= hidden_states`. Auto-detected at
 *   `GemmaNetworkLoader.fromWeights`.
 * @param ple whether to enable Per-Layer Embedding (Gemma 4's
 *   hidden_size_per_layer_input auxiliary signal). Adds a
 *   [PerLayerInputBlockHook] to each block and wraps the whole network
 *   in a [GemmaModel] with a top-level [PerLayerEmbedding]. Default
 *   false — leaving PLE off produces the same observable forward pass
 *   as the pre-5f.5 Sequential wrap. Opt-in only for now; real-model
 *   accuracy is still being validated.
 */
public fun <T : DType, V> gemmaNetwork(
    metadata: Gemma4ModelMetadata,
    dtype: kotlin.reflect.KClass<T>,
    maxInferenceLen: Int = minOf(metadata.contextLength, 4096),
    qkNorm: Boolean = true,
    sandwichNorms: Boolean = true,
    layerOutputScale: Boolean = true,
    ple: Boolean = false,
    pleSideChannelOnly: Boolean = false,
    /**
     * Whether attention applies Gemma 4's parameterless per-head V RMS-norm.
     *
     * **Architecture-dependent, not a Gemma-family constant.** HF `Gemma4TextAttention` declares
     * `v_norm = Gemma4RMSNorm(head_dim, with_scale=False)` and applies it right after `v_proj`;
     * **gemma3 has no such norm** — llama.cpp's gemma3 graph goes `Vcur → MUL_MAT → RESHAPE →
     * VIEW` with nothing in between, verified against `llama-eval-callback`. Because this builder
     * serves both families, forcing it on for everyone silently corrupts gemma3: Q and K stay
     * bit-exact while V is rescaled, so the attention output is only slightly wrong at position 0
     * and compounds as more values enter the weighted sum — matching llama.cpp's greedy argmax at
     * prefill n=3 and diverging by n=8 on FunctionGemma 270M.
     *
     * Default derives from the checkpoint's declared architecture.
     */
    vNorm: Boolean = metadata.architecture.startsWith("gemma4")
): Module<T, V> {
    val dim = metadata.embeddingLength
    val nHeads = metadata.headCount
    val nKVHeads = metadata.kvHeadCount
    val nLayers = metadata.blockCount
    val seqLen = maxInferenceLen
    val vocabSize = metadata.vocabSize
    val eps = metadata.rmsNormEps

    // Gemma 4 rotates only half of head_dim on global (p-RoPE) layers; the
    // metadata field is per-scheme, but the two schemes share this fraction
    // in current checkpoints so we read it off the full params.
    val partialRotaryFactor = metadata.ropeParametersFull.partialRotaryFactor
    val scalingFactor = metadata.ropeParametersFull.factor

    val nnCtx = DefaultNeuralNetworkExecutionContext()
    val dslImpl = NeuralNetworkDslImpl<T, V>(nnCtx, dtype)
    dslImpl.embedding(vocabSize, dim, id = "token_embd")

    // Per HF `Gemma4Attention`, kv-shared layers reuse the K/V tensors of
    // the **last non-shared layer of the same attention type**. So the
    // sliding follower set (layers ≥ firstSharedLayer with type=SLIDING)
    // shares the cache of the last sliding layer in [0, firstSharedLayer);
    // global followers share the last global layer's cache. The two owner
    // caches stay at their respective head_dim (256 / 512) — no cross-type
    // padding is needed because each follower attends to data computed by
    // an owner of the *same* head_dim.
    val firstSharedLayer = nLayers - metadata.kvSharedLayers
    val typeOwners = mutableMapOf<LayerType, PositionalKVCache<T, V>>()
    val typeOwnerLayerIdx = mutableMapOf<LayerType, Int>()
    if (metadata.kvSharedLayers > 0) {
        for (l in 0 until firstSharedLayer) {
            // Track latest of each type seen so far; the LAST entry per
            // type before firstSharedLayer wins.
            typeOwnerLayerIdx[metadata.getLayerType(l)] = l
        }
    }

    for (layer in 0 until nLayers) {
        val layerHeadDim = metadata.getHeadDim(layer)
        val ropeBase = metadata.getRopeBase(layer)
        val layerType = metadata.getLayerType(layer)
        val isGlobal = layerType == LayerType.GLOBAL
        val ropeScaling = if (isGlobal && metadata.ropeParametersFull.ropeType == "proportional") {
            RoPEScaling.PROPORTIONAL
        } else {
            RoPEScaling.NONE
        }
        // Sliding layers get a bounded attention window; global layers see everything.
        val slidingWindow = if (isGlobal) null else metadata.slidingWindow
        val ffnDim = metadata.getIntermediateSize(layer)
        val isInSharedGroup = metadata.kvSharedLayers > 0 && layer >= firstSharedLayer

        val stage = StageImpl<T, V>(nnCtx, "blk.$layer", dtype)
        stage.rmsNorm(dim, eps, id = "attn_norm", unitOffset = false)
        stage.multiHeadAttention(
            dim = dim,
            nHeads = nHeads,
            nKVHeads = nKVHeads,
            causal = true,
            qkNorm = qkNorm, // Gemma 4 per-head RMSNorm on Q and K before RoPE
            qkNormUnitOffset = false, // DIAG: gemma3 gguf may already bake (1+w) into q/k norm
            // gemma3 scales attention by query_pre_attn_scalar^-0.5 = 1/sqrt(head_dim)
            // (HF Gemma3Attention). null => MHA's 1/sqrt(headDim) default. The
            // prior hardcoded 1.0 (a Gemma-4 "q/k-norm makes scale 1.0" claim)
            // over-sharpened softmax for >1 token => parity broke at pos>=1.
            attentionScale = null,
            // Gemma 4 (unlike gemma3) DOES have v_norm: HF Gemma4TextAttention.__init__
            // declares `self.v_norm = Gemma4RMSNorm(self.head_dim, eps=..., with_scale=False)`
            // and forward() calls `value_states = self.v_norm(value_states)` right after
            // v_proj, before it's used in attention (transformers 5.15.0, confirmed against
            // real source — not just a comment/doc claim). No GGUF tensor is needed since
            // with_scale=False is a parameterless per-head RMS normalization, computed
            // on-the-fly below (same as MultiHeadAttention's existing vNormNoScale path).
            // Previously disabled here based on a gemma3-specific finding (gemma3 genuinely
            // has no v_norm) that got incorrectly generalized to gemma4 too — this was THE
            // root cause of the immediate-<eos> decode collapse investigated in
            // GEMMA4_E2B_SKAINET_FINDINGS.md: without it, raw (un-normalized) V values (whose
            // per-head RMS is ~40-50, confirmed matching the real checkpoint via direct
            // ground-truth comparison) flow straight into the softmax-weighted sum, producing
            // an attention output ~35-60x too large, which compounds through 35 layers into a
            // final-logit distribution completely unlike the real trained model's.
            vNormNoScale = vNorm,
            id = "attn",
            slidingWindow = slidingWindow
        ) {
            rope(
                headDim = layerHeadDim,
                maxSeqLen = seqLen,
                // Gemma 4 (and all HF transformers): split-half (NEOX) pairing.
                // Pair dim_i with dim_{i + headDim/2}. The default INTERLEAVED
                // pairing (i, i+1) does NOT match HF / GGUF storage convention
                // for Gemma — see `convert_hf_to_gguf.py` Gemma4Model: weights
                // are stored as-is from HF, and HF uses split-half pairing in
                // `apply_rotary_pos_emb`. Using INTERLEAVED gives correct-by-
                // accident outputs at small N but compounds wrong rotations
                // across positions for N≥3 (especially at full-attention
                // layers, where partial_rotary_factor=0.25 amplifies the
                // pairing mismatch).
                mode = RoPEMode.SPLIT_HALF,
                base = ropeBase,
                scaling = ropeScaling,
                scalingFactor = if (ropeScaling == RoPEScaling.PROPORTIONAL) scalingFactor else 1.0f,
                partialRotaryFactor = if (isGlobal) partialRotaryFactor else 1.0f
            )
            if (!isInSharedGroup) {
                // Non-shared layer: plain per-layer cache at its own head_dim.
                // If this is the LAST non-shared layer of its type, the cache
                // also serves as the read-only delegate for downstream
                // shared-group followers of the same type.
                val own = PositionalKVCache<T, V>(
                    maxSeqLen = seqLen,
                    nKVHeads = nKVHeads,
                    headDim = layerHeadDim,
                    name = "blk.$layer.attn.kv_cache"
                )
                kvCache(own)
                if (typeOwnerLayerIdx[layerType] == layer) {
                    typeOwners[layerType] = own
                }
            } else {
                // Shared follower: discard our k_proj/v_proj output and read
                // the same-type owner's cache. Per HF Gemma4Attention,
                // shared layers don't even have their own k_proj/v_proj —
                // GGUF carries them anyway, but their forward output is
                // ignored by this wrapper.
                val ownerCache = typeOwners[layerType]
                    ?: error(
                        "Gemma: kv-shared layer $layer (type=$layerType) has no " +
                            "non-shared owner of the same type before " +
                            "firstSharedLayer=$firstSharedLayer"
                    )
                require(ownerCache.headDim == layerHeadDim) {
                    "Gemma: kv-shared layer $layer head_dim=$layerHeadDim != " +
                        "owner head_dim=${ownerCache.headDim} (type=$layerType)"
                }
                kvCache(
                    OwnerReadOnlyKVCache(
                        delegate = ownerCache,
                        name = "blk.$layer.attn.kv_cache"
                    )
                )
            }
        }
        if (sandwichNorms) stage.rmsNorm(dim, eps, id = "post_attention_norm", unitOffset = false)
        stage.residual()

        stage.rmsNorm(dim, eps, id = "ffn_norm", unitOffset = false)
        stage.geGluFFN(dim, ffnDim, id = "ffn")
        if (sandwichNorms) stage.rmsNorm(dim, eps, id = "post_ffw_norm", unitOffset = false)
        stage.residual()

        // PLE hook fires between FFN residual and layer_output_scale.
        // Exact order per Gemma4TextDecoderLayer.forward (transformers 5.6.0).
        if (ple) stage.modules += PerLayerInputBlockHook<T, V>(
            hiddenSize = dim,
            perLayerDim = metadata.perLayerEmbeddingLength,
            sideChannelOnly = pleSideChannelOnly,
            name = "blk.$layer.per_layer_input"
        )

        // Gemma 4 tail: scalar-broadcast multiply by layer_output_scale.
        // HF calls this self.layer_scalar = torch.ones(1), applied as
        // `hidden_states *= self.layer_scalar` at the end of the block.
        if (layerOutputScale) stage.modules += LayerScalarMul<T, V>(
            name = "blk.$layer.layer_output_scale",
            dtype = dtype
        )

        dslImpl.modules += HybridTransformerBlock(stage.modules.toList(), name = "blk.$layer")
    }

    dslImpl.rmsNorm(dim, eps, id = "output_norm", unitOffset = false)
    // Void placeholder for output projection — Gemma 4 E2B vocab is 262 144
    // and dense(vocabSize) would eagerly allocate ~1.5 GB of zeros before
    // WeightMapper runs.
    dslImpl.modules += VoidDense<T, V>("output", vocabSize, dim, dtype = dtype)

    // Build the top-level GemmaModel wrapper. When PLE is disabled the model
    // runs the same forward sequence the pre-5f.5 `dslImpl.create()` wrap
    // produced; when PLE is enabled the wrapper threads per_layer_inputs
    // into each block's PerLayerInputBlockHook before running the block.
    @Suppress("UNCHECKED_CAST")
    val tokenEmbedding = dslImpl.modules[0] as EmbeddingAdapter<T, V>
    val blocks = dslImpl.modules.filterIsInstance<HybridTransformerBlock<T, V>>()
    val outputNormModule = dslImpl.modules[dslImpl.modules.size - 2] as RMSNormalization<T, V>
    @Suppress("UNCHECKED_CAST")
    val lmHead = dslImpl.modules[dslImpl.modules.size - 1] as VoidDense<T, V>

    val pleModule: PerLayerEmbedding<T, V>? = if (ple) PerLayerEmbedding<T, V>(
        vocabSize = vocabSize,
        hiddenSize = dim,
        numLayers = nLayers,
        perLayerDim = metadata.perLayerEmbeddingLength,
        rmsEps = eps
    ) else null

    return GemmaModel(
        tokenEmbedding = tokenEmbedding,
        ple = pleModule,
        blocks = blocks,
        outputNorm = outputNormModule,
        lmHead = lmHead,
        dtype = dtype,
        // Gemma 4 scales token embeddings by sqrt(hidden_size) before the trunk
        // (HF: `embed_scale=config.hidden_size**0.5`). Without this, trunk
        // activations are ~1/sqrt(dim) of their trained magnitude and decode
        // collapses to a near-uniform distribution.
        embedScale = kotlin.math.sqrt(dim.toFloat()),
        finalLogitSoftcapping = metadata.finalLogitSoftcapping,
        name = "GemmaModel"
    )
}
