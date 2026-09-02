package sk.ainet.models.gemma3n

import sk.ainet.apps.llm.HybridTransformerBlock
import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.dsl.NeuralNetworkDslImpl
import sk.ainet.lang.nn.dsl.StageImpl
import sk.ainet.lang.nn.dsl.embedding
import sk.ainet.lang.nn.dsl.multiHeadAttention
import sk.ainet.lang.nn.dsl.rmsNorm
import sk.ainet.lang.nn.layers.EmbeddingAdapter
import sk.ainet.lang.nn.normalization.RMSNormalization
import sk.ainet.lang.nn.transformer.OwnerReadOnlyKVCache
import sk.ainet.lang.nn.transformer.PositionalKVCache
import sk.ainet.lang.nn.transformer.RoPEMode
import sk.ainet.lang.nn.transformer.VoidDense
import sk.ainet.lang.types.DType
import sk.ainet.models.gemma.LayerType
import sk.ainet.models.gemma.PerLayerEmbedding
import kotlin.reflect.KClass

/**
 * Gemma 3n architecture defined via the network DSL — the #377 DSL migration, replacing the
 * hand-rolled `Gemma3nRuntime`. Faithful to HF `Gemma3nTextModel` (see [Gemma3nModel] for
 * the per-layer flow) and buildable into a compute-graph tape for the StableHLO → IREE
 * mobile path.
 *
 * What gemma3n adds over the gemma-4 lane's `gemmaNetwork()` (which already carries hybrid
 * sliding/global attention, dual RoPE bases, per-layer FFN dims, per-type shared KV, PLE,
 * q/k-norm, parameterless v-norm and attention scale 1.0):
 * [Gemma3nAltUpBlock] (4 parallel streams + router), [Gemma3nLaurelBlock],
 * [Gemma3nSparseGeGluFFN] (Gaussian-top-k on the first layers) and the PLE delta going to
 * the non-active AltUp streams instead of the residual.
 */
public fun <T : DType, V> gemma3nNetwork(
    metadata: Gemma3nModelMetadata,
    dtype: KClass<T>,
    maxInferenceLen: Int = minOf(metadata.contextLength, 4096),
    /** HF `laurel_rank` (64 on real checkpoints; not in the GGUF — the loader derives it
     *  from `blk.0.laurel_l`'s shape). */
    laurelRank: Int = LAUREL_RANK,
    /** Number of layers the PLE tensors cover — normally [Gemma3nModelMetadata.blockCount],
     *  but a layer-truncated export build keeps the FULL table so the stored
     *  `per_layer_*` tensors still bind shape-exact. */
    pleNumLayers: Int = metadata.blockCount,
): Module<T, V> {
    val dim = metadata.embeddingLength
    val nHeads = metadata.headCount
    val nKVHeads = metadata.kvHeadCount
    val nLayers = metadata.blockCount
    val headDim = metadata.headDim
    val seqLen = maxInferenceLen
    val vocabSize = metadata.vocabSize
    val eps = metadata.rmsNormEps

    val nnCtx = DefaultNeuralNetworkExecutionContext()
    val dslImpl = NeuralNetworkDslImpl<T, V>(nnCtx, dtype)
    dslImpl.embedding(vocabSize, dim, id = "token_embd")

    // KV sharing: same owner-per-attention-type scheme as gemma-4 (HF: a shared layer
    // reuses the K/V of the LAST non-shared layer of the same type).
    val firstSharedLayer = nLayers - metadata.kvSharedLayers
    val typeOwners = mutableMapOf<LayerType, PositionalKVCache<T, V>>()
    val typeOwnerLayerIdx = mutableMapOf<LayerType, Int>()
    if (metadata.kvSharedLayers > 0) {
        for (l in 0 until firstSharedLayer) {
            typeOwnerLayerIdx[metadata.getLayerType(l)] = l
        }
    }

    for (layer in 0 until nLayers) {
        val layerType = metadata.getLayerType(layer)
        val isGlobal = layerType == LayerType.GLOBAL
        val ropeBase = metadata.getRopeBase(layer)
        val slidingWindow = if (isGlobal) null else metadata.slidingWindow
        val ffnDim = metadata.feedForwardLengths.getOrElse(layer) { metadata.feedForwardLengths.last() }
        val isInSharedGroup = metadata.kvSharedLayers > 0 && layer >= firstSharedLayer

        val stage = StageImpl<T, V>(nnCtx, "blk.$layer", dtype)
        stage.rmsNorm(dim, eps, id = "attn_norm", unitOffset = false)
        stage.multiHeadAttention(
            dim = dim,
            nHeads = nHeads,
            nKVHeads = nKVHeads,
            causal = true,
            // HF Gemma3nTextAttention: per-head RMSNorm (with scale) on Q and K before RoPE,
            // parameterless per-head RMSNorm on V, attention scaling fixed to 1.0.
            qkNorm = true,
            qkNormUnitOffset = false,
            qkNormEps = eps,
            attentionScale = 1.0f,
            vNormNoScale = true,
            id = "attn",
            slidingWindow = slidingWindow,
        ) {
            rope(
                headDim = headDim,
                maxSeqLen = seqLen,
                mode = RoPEMode.SPLIT_HALF,
                base = ropeBase,
            )
            if (!isInSharedGroup) {
                val own = PositionalKVCache<T, V>(
                    maxSeqLen = seqLen,
                    nKVHeads = nKVHeads,
                    headDim = headDim,
                    name = "blk.$layer.attn.kv_cache",
                )
                kvCache(own)
                if (typeOwnerLayerIdx[layerType] == layer) typeOwners[layerType] = own
            } else {
                val ownerCache = typeOwners[layerType]
                    ?: error(
                        "gemma3n: kv-shared layer $layer (type=$layerType) has no non-shared " +
                            "owner of the same type before firstSharedLayer=$firstSharedLayer",
                    )
                kvCache(OwnerReadOnlyKVCache(delegate = ownerCache, name = "blk.$layer.attn.kv_cache"))
            }
        }
        stage.rmsNorm(dim, eps, id = "post_attention_norm", unitOffset = false)
        stage.rmsNorm(dim, eps, id = "ffn_norm", unitOffset = false)     // pre_feedforward_layernorm
        stage.modules += Gemma3nSparseGeGluFFN<T, V>(
            hiddenSize = dim,
            ffnDim = ffnDim,
            stdMultiplier = metadata.sparsityScaleFor(layer) ?: Float.NEGATIVE_INFINITY,
            dtype = dtype,
            name = "ffn",
        )
        stage.rmsNorm(dim, eps, id = "post_ffw_norm", unitOffset = false)
        stage.modules += Gemma3nLaurelBlock<T, V>(
            hiddenSize = dim,
            laurelRank = laurelRank,
            rmsEps = eps,
            dtype = dtype,
            name = "laurel",
        )
        stage.modules += Gemma3nAltUpBlock<T, V>(
            hiddenSize = dim,
            numInputs = metadata.numAltupInputs,
            activeIdx = metadata.altupActiveIdx,
            rmsEps = eps,
            dtype = dtype,
            name = "altup",
        )
        stage.modules += Gemma3nPerLayerApply<T, V>(
            hiddenSize = dim,
            perLayerDim = metadata.perLayerEmbeddingLength,
            rmsEps = eps,
            dtype = dtype,
            name = "per_layer_input",
        )
        dslImpl.modules += HybridTransformerBlock(stage.modules.toList(), name = "blk.$layer")
    }

    dslImpl.rmsNorm(dim, eps, id = "output_norm", unitOffset = false)
    // Void placeholder — the 262k vocab head would eagerly allocate ~2 GB of zeros otherwise;
    // gemma3n ties the head to token_embd, bound by the loader.
    dslImpl.modules += VoidDense<T, V>("output", vocabSize, dim, dtype = dtype)

    @Suppress("UNCHECKED_CAST")
    val tokenEmbedding = dslImpl.modules[0] as EmbeddingAdapter<T, V>
    val blocks = dslImpl.modules.filterIsInstance<HybridTransformerBlock<T, V>>()
    val outputNorm = dslImpl.modules[dslImpl.modules.size - 2] as RMSNormalization<T, V>
    @Suppress("UNCHECKED_CAST")
    val lmHead = dslImpl.modules[dslImpl.modules.size - 1] as VoidDense<T, V>

    val ple = PerLayerEmbedding<T, V>(
        vocabSize = vocabSize,
        hiddenSize = dim,
        numLayers = pleNumLayers,
        perLayerDim = metadata.perLayerEmbeddingLength,
        rmsEps = eps,
    )

    return Gemma3nModel(
        tokenEmbedding = tokenEmbedding,
        ple = ple,
        altupGlobals = Gemma3nAltUpGlobals(
            hiddenSize = dim,
            numInputs = metadata.numAltupInputs,
            dtype = dtype,
        ),
        blocks = blocks,
        outputNorm = outputNorm,
        lmHead = lmHead,
        dtype = dtype,
        activeIdx = metadata.altupActiveIdx,
        // HF: embed_scale = hidden_size ** 0.5 (bf16-rounded in the reference; we bisect
        // against llama.cpp if the rounding ever matters at parity tolerance).
        embedScale = kotlin.math.sqrt(dim.toFloat()),
    )
}

/** HF `laurel_rank` — constant 64 across gemma-3n checkpoints (not stored in the GGUF). */
public const val LAUREL_RANK: Int = 64
