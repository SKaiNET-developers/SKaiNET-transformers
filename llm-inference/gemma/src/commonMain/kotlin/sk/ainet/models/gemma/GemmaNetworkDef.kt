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
import sk.ainet.lang.nn.transformer.PaddedSharedPositionalKVCache
import sk.ainet.lang.nn.transformer.PositionalKVCache
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
    ple: Boolean = false
): Module<T, V> = gemmaNetwork<T, V>(metadata, T::class, maxInferenceLen, qkNorm, sandwichNorms, layerOutputScale, ple)

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
    ple: Boolean = false
): Module<T, V> {
    val dim = metadata.embeddingLength
    val nHeads = metadata.headCount
    val nKVHeads = metadata.kvHeadCount
    val nLayers = metadata.blockCount
    val seqLen = maxInferenceLen
    val vocabSize = metadata.vocabSize
    val eps = 1e-6f

    // Gemma 4 rotates only half of head_dim on global (p-RoPE) layers; the
    // metadata field is per-scheme, but the two schemes share this fraction
    // in current checkpoints so we read it off the full params.
    val partialRotaryFactor = metadata.ropeParametersFull.partialRotaryFactor
    val scalingFactor = metadata.ropeParametersFull.factor

    val nnCtx = DefaultNeuralNetworkExecutionContext()
    val dslImpl = NeuralNetworkDslImpl<T, V>(nnCtx, dtype)
    dslImpl.embedding(vocabSize, dim, id = "token_embd")

    // Owner layers' PositionalKVCache delegates, indexed by their layer
    // number. For the kvSharedLayers trailing group, the delegate's headDim
    // is the MAX across the group — Gemma 4 E2B mixes SLIDING (256) and
    // GLOBAL (512) inside the same group, so the delegate must be 512-wide.
    // Layers in the group (owner and followers both) wrap the delegate in
    // a PaddedSharedPositionalKVCache that pads-on-write and slices-on-read
    // to the layer's own head_dim — matches HeapGemma4KvCache semantics.
    val ownerCaches = mutableMapOf<Int, PositionalKVCache<T, V>>()

    val firstSharedLayer = nLayers - metadata.kvSharedLayers
    val sharedGroupPaddedHeadDim = if (metadata.kvSharedLayers > 0) {
        (firstSharedLayer until nLayers).maxOf { metadata.getHeadDim(it) }
    } else {
        0 // unused
    }

    for (layer in 0 until nLayers) {
        val layerHeadDim = metadata.getHeadDim(layer)
        val ropeBase = metadata.getRopeBase(layer)
        val isGlobal = metadata.getLayerType(layer) == LayerType.GLOBAL
        val ropeScaling = if (isGlobal && metadata.ropeParametersFull.ropeType == "proportional") {
            RoPEScaling.PROPORTIONAL
        } else {
            RoPEScaling.NONE
        }
        // Sliding layers get a bounded attention window; global layers see everything.
        val slidingWindow = if (isGlobal) null else metadata.slidingWindow
        val ffnDim = metadata.getIntermediateSize(layer)
        val cacheOwnerLayer = metadata.getCacheLayerIndex(layer)
        val isInSharedGroup = metadata.kvSharedLayers > 0 && layer >= firstSharedLayer

        val stage = StageImpl<T, V>(nnCtx, "blk.$layer", dtype)
        stage.rmsNorm(dim, eps, id = "attn_norm")
        stage.multiHeadAttention(
            dim = dim,
            nHeads = nHeads,
            nKVHeads = nKVHeads,
            causal = true,
            qkNorm = qkNorm, // Gemma 4 per-head RMSNorm on Q and K before RoPE
            id = "attn",
            slidingWindow = slidingWindow
        ) {
            rope(
                headDim = layerHeadDim,
                maxSeqLen = seqLen,
                base = ropeBase,
                scaling = ropeScaling,
                scalingFactor = if (ropeScaling == RoPEScaling.PROPORTIONAL) scalingFactor else 1.0f,
                partialRotaryFactor = if (isGlobal) partialRotaryFactor else 1.0f
            )
            when {
                !isInSharedGroup -> {
                    // Plain per-layer cache. Own this layer's own head_dim.
                    val own = PositionalKVCache<T, V>(
                        maxSeqLen = seqLen,
                        nKVHeads = nKVHeads,
                        headDim = layerHeadDim,
                        name = "blk.$layer.attn.kv_cache"
                    )
                    kvCache(own)
                }
                cacheOwnerLayer == layer -> {
                    // Owner of the shared group: allocate the padded-max storage
                    // delegate and wrap it with a PaddedSharedPositionalKVCache at
                    // this layer's own head_dim.
                    val delegate = PositionalKVCache<T, V>(
                        maxSeqLen = seqLen,
                        nKVHeads = nKVHeads,
                        headDim = sharedGroupPaddedHeadDim,
                        name = "blk.$layer.attn.kv_cache.storage"
                    )
                    ownerCaches[layer] = delegate
                    kvCache(
                        PaddedSharedPositionalKVCache(
                            delegate = delegate,
                            layerHeadDim = layerHeadDim,
                            name = "blk.$layer.attn.kv_cache"
                        )
                    )
                }
                else -> {
                    // Follower: wrap the shared delegate at this layer's head_dim.
                    val delegate = ownerCaches[cacheOwnerLayer]
                        ?: error("Gemma: layer $layer expects to share KV with layer $cacheOwnerLayer, but owner hasn't been built yet")
                    kvCache(
                        PaddedSharedPositionalKVCache(
                            delegate = delegate,
                            layerHeadDim = layerHeadDim,
                            name = "blk.$layer.attn.kv_cache"
                        )
                    )
                }
            }
        }
        if (sandwichNorms) stage.rmsNorm(dim, eps, id = "post_attention_norm")
        stage.residual()

        stage.rmsNorm(dim, eps, id = "ffn_norm")
        stage.geGluFFN(dim, ffnDim, id = "ffn")
        if (sandwichNorms) stage.rmsNorm(dim, eps, id = "post_ffw_norm")
        stage.residual()

        // PLE hook fires between FFN residual and layer_output_scale.
        // Exact order per Gemma4TextDecoderLayer.forward (transformers 5.6.0).
        if (ple) stage.modules += PerLayerInputBlockHook<T, V>(
            hiddenSize = dim,
            perLayerDim = metadata.perLayerEmbeddingLength,
            name = "blk.$layer.per_layer_input"
        )

        // Gemma 4 tail: scalar-broadcast multiply by layer_output_scale.
        // HF calls this self.layer_scalar = torch.ones(1), applied as
        // `hidden_states *= self.layer_scalar` at the end of the block.
        if (layerOutputScale) stage.modules += LayerScalarMul<T, V>(
            name = "blk.$layer.layer_output_scale"
        )

        dslImpl.modules += HybridTransformerBlock(stage.modules.toList(), name = "blk.$layer")
    }

    dslImpl.rmsNorm(dim, eps, id = "output_norm")
    // Void placeholder for output projection — Gemma 4 E2B vocab is 262 144
    // and dense(vocabSize) would eagerly allocate ~1.5 GB of zeros before
    // WeightMapper runs.
    dslImpl.modules += VoidDense<T, V>("output", vocabSize, dim)

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
        finalLogitSoftcapping = metadata.finalLogitSoftcapping,
        name = "GemmaModel"
    )
}
