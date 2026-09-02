package sk.ainet.models.gemma3n

import sk.ainet.models.gemma.LayerType

/**
 * Metadata for Gemma 3n models extracted from GGUF files.
 *
 * Gemma 3n has several unique characteristics:
 * - Variable intermediate (FFN) sizes per layer (MatFormer architecture)
 * - Hybrid attention pattern (4 local sliding-window + 1 global)
 * - Per-layer embeddings (optional)
 * - KV cache sharing for the last N layers
 */
public data class Gemma3nModelMetadata(
    val architecture: String,
    val embeddingLength: Int,
    val perLayerEmbeddingLength: Int,
    val contextLength: Int,
    val blockCount: Int,
    val headCount: Int,
    val kvHeadCount: Int,
    val feedForwardLengths: List<Int>,
    val headDim: Int,
    val vocabSize: Int,
    val slidingWindow: Int,
    val ropeBaseLocal: Float,
    val ropeBaseGlobal: Float,
    val kvSharedLayers: Int,
    val layerPattern: List<String>,
    /** Number of AltUp parallel inputs. E4B: 4, E2B: 1 (no-op). */
    val numAltupInputs: Int = 1,
    /** Active input index for AltUp routing. */
    val altupActiveIdx: Int = 0,
    /** Per-layer activation sparsity rates. Empty means no sparsity. */
    val activationSparsityPattern: List<Float> = emptyList(),
    /** Activation sparsity scale factor (from GGUF: gemma3n.activation_sparsity_scale). */
    val activationSparsityScale: Float = 0f,
    /** RMSNorm epsilon (`gemma3n.attention.layer_norm_rms_epsilon`; real E2B/E4B: 1e-6). */
    val rmsNormEps: Float = 1e-6f,
    /**
     * Per-layer activation-sparsity std multipliers (`gemma3n.activation_sparsity_scale`).
     * Real checkpoints store `Φ⁻¹(target_sparsity) ≈ 1.6449` on sparse layers and `-inf`
     * on the rest — non-finite (or empty list) disables sparsity for that layer.
     */
    val activationSparsityScales: List<Float> = emptyList(),
) {
    /** The std multiplier for one layer, or `null` when sparsity is off there. */
    public fun sparsityScaleFor(layerIdx: Int): Float? =
        activationSparsityScales.getOrNull(layerIdx)?.takeIf { it.isFinite() && it > 0f }
    /**
     * Returns the layer type at the given layer index.
     * Pattern repeats: ["sliding", "sliding", "sliding", "sliding", "full"]
     */
    public fun getLayerType(layerIdx: Int): LayerType {
        val patternIdx = layerIdx % layerPattern.size
        return when (layerPattern[patternIdx]) {
            "full", "global" -> LayerType.GLOBAL
            else -> LayerType.SLIDING
        }
    }

    /**
     * Returns the RoPE base frequency for the given layer.
     * Local/sliding layers use ropeBaseLocal (10k), global layers use ropeBaseGlobal (1M).
     */
    public fun getRopeBase(layerIdx: Int): Float {
        return when (getLayerType(layerIdx)) {
            LayerType.GLOBAL -> ropeBaseGlobal
            LayerType.SLIDING -> ropeBaseLocal
        }
    }

    /**
     * Returns the effective attention window size for the given layer.
     * Sliding layers use slidingWindow, global layers use contextLength.
     */
    public fun getEffectiveWindow(layerIdx: Int): Int {
        return when (getLayerType(layerIdx)) {
            LayerType.GLOBAL -> contextLength
            LayerType.SLIDING -> slidingWindow
        }
    }

    /**
     * Returns whether the given layer shares its KV cache with another layer.
     * The last kvSharedLayers layers share their KV cache.
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
     * Returns the feed-forward dimension for the given layer.
     */
    public fun getFeedForwardLength(layerIdx: Int): Int {
        return feedForwardLengths.getOrElse(layerIdx) {
            feedForwardLengths.lastOrNull() ?: (embeddingLength * 4)
        }
    }

    public companion object {
        /** Default layer pattern: 4 sliding + 1 global, repeating */
        public val DEFAULT_LAYER_PATTERN: List<String> = listOf("sliding", "sliding", "sliding", "sliding", "full")

        /** Default sliding window size */
        public const val DEFAULT_SLIDING_WINDOW: Int = 512

        /** Default local RoPE base frequency */
        public const val DEFAULT_ROPE_BASE_LOCAL: Float = 10000f

        /** Default global RoPE base frequency */
        public const val DEFAULT_ROPE_BASE_GLOBAL: Float = 1000000f

        /** Default number of KV shared layers */
        public const val DEFAULT_KV_SHARED_LAYERS: Int = 15
    }
}

