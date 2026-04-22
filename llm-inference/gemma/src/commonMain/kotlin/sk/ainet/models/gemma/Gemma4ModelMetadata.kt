package sk.ainet.models.gemma

/**
 * Metadata for Gemma 4 models extracted from HuggingFace config.json or GGUF files.
 *
 * Gemma 4 key architectural differences from Gemma 3n:
 * - Proportional RoPE (p-RoPE) for global attention layers
 * - Separate global_head_dim for global attention layers
 * - New control tokens (<|turn>, <turn|>, <|think|>, tool tokens)
 * - Up to 256K context (128K for E2B/E4B, 256K for 31B)
 * - No AltUp or activation sparsity (those are Gemma3n-specific)
 */
public data class Gemma4ModelMetadata(
    val architecture: String,
    val embeddingLength: Int,
    val contextLength: Int,
    val blockCount: Int,
    val headCount: Int,
    val kvHeadCount: Int,
    val intermediateSize: Int,
    val headDim: Int,
    val globalHeadDim: Int,
    val vocabSize: Int,
    val slidingWindow: Int,
    val kvSharedLayers: Int,
    /** Full per-layer list of attention types (not a repeating pattern). */
    val layerTypes: List<String>,
    /** RoPE parameters for full (global) attention layers. */
    val ropeParametersFull: Gemma4RopeConfig,
    /** RoPE parameters for sliding (local) attention layers. */
    val ropeParametersSliding: Gemma4RopeConfig,
    /** Maximum position embeddings (128K for E2B/E4B, 256K for 31B). */
    val maxPositionEmbeddings: Int,
    /** Per-layer embedding dimension (PLE). 0 if not used. */
    val perLayerEmbeddingLength: Int = 0,
    /**
     * Per-layer intermediate (FFN) size. Empty means use the scalar [intermediateSize] for all
     * layers. Gemma 4 E2B/E4B checkpoints encode distinct FFN widths per block (e.g. 6144 for
     * early layers, 12288 for later layers).
     */
    val perLayerIntermediateSize: List<Int> = emptyList(),
    /** BOS token ID. */
    val bosTokenId: Int = 2,
    /** EOS token ID. */
    val eosTokenId: Int = 1,
    /** PAD token ID. */
    val padTokenId: Int = 0
) {
    /**
     * Returns the layer type at the given layer index.
     * Uses the full per-layer list (last layer is always global).
     */
    public fun getLayerType(layerIdx: Int): LayerType {
        val type = layerTypes.getOrElse(layerIdx) { "full_attention" }
        return when (type) {
            "sliding_attention" -> LayerType.SLIDING
            "full_attention" -> LayerType.GLOBAL
            else -> LayerType.SLIDING
        }
    }

    /**
     * Returns the RoPE base frequency for the given layer.
     */
    public fun getRopeBase(layerIdx: Int): Float {
        return when (getLayerType(layerIdx)) {
            LayerType.GLOBAL -> ropeParametersFull.base
            LayerType.SLIDING -> ropeParametersSliding.base
        }
    }

    /**
     * Returns the effective attention window size for the given layer.
     */
    public fun getEffectiveWindow(layerIdx: Int): Int {
        return when (getLayerType(layerIdx)) {
            LayerType.GLOBAL -> contextLength
            LayerType.SLIDING -> slidingWindow
        }
    }

    /**
     * Returns whether the given layer shares its KV cache with another layer.
     */
    public fun isKvShared(layerIdx: Int): Boolean {
        return layerIdx >= (blockCount - kvSharedLayers)
    }

    /**
     * Returns the cache layer index for the given layer.
     * Shared layers map to a single cache slot.
     */
    public fun getCacheLayerIndex(layerIdx: Int): Int {
        return if (isKvShared(layerIdx)) {
            blockCount - kvSharedLayers
        } else {
            layerIdx
        }
    }

    /**
     * Returns the head dimension for the given layer.
     * Global layers use globalHeadDim, sliding layers use headDim.
     */
    public fun getHeadDim(layerIdx: Int): Int {
        return when (getLayerType(layerIdx)) {
            LayerType.GLOBAL -> globalHeadDim
            LayerType.SLIDING -> headDim
        }
    }

    /** Returns the FFN intermediate size for the given layer. */
    public fun getIntermediateSize(layerIdx: Int): Int =
        perLayerIntermediateSize.getOrNull(layerIdx) ?: intermediateSize

    public companion object {
        public const val DEFAULT_SLIDING_WINDOW: Int = 512
        public const val DEFAULT_KV_SHARED_LAYERS: Int = 20
        public const val DEFAULT_HEAD_DIM: Int = 256
        public const val DEFAULT_GLOBAL_HEAD_DIM: Int = 256
    }
}

/**
 * RoPE configuration for a specific attention type (full or sliding).
 */
public data class Gemma4RopeConfig(
    /** Base frequency (theta) for RoPE. */
    val base: Float,
    /** RoPE type: "default" for standard, "proportional" for p-RoPE. */
    val ropeType: String = "default",
    /** Scaling factor for proportional RoPE. */
    val factor: Float = 1.0f,
    /** Original max position embeddings used for RoPE scaling. */
    val originalMaxPositionEmbeddings: Int = 8192,
    /** Fraction of head dimensions to apply rotary embeddings to. 1.0 = all. */
    val partialRotaryFactor: Float = 1.0f
)
