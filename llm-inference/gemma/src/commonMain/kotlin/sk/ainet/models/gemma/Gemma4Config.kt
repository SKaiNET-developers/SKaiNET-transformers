package sk.ainet.models.gemma

/**
 * Configuration for Gemma 4 model inference.
 *
 * Key architecture differences from Gemma 3n:
 * - Proportional RoPE (p-RoPE) for global attention layers
 * - Separate global_head_dim for global layers
 * - Uniform FFN intermediate size (no variable per-layer sizes)
 * - No AltUp or activation sparsity
 * - Full per-layer attention type list (not a repeating pattern)
 * - Up to 256K context window
 */
public data class Gemma4Config(
    val hiddenSize: Int = 2304,
    val numLayers: Int = 34,
    val numAttentionHeads: Int = 8,
    val numKvHeads: Int = 4,
    val headDim: Int = 256,
    val globalHeadDim: Int = 256,
    val intermediateSize: Int = 9216,
    val slidingWindow: Int = 512,
    val layerTypes: List<String> = emptyList(),
    val ropeBaseLocal: Float = 10000f,
    val ropeBaseGlobal: Float = 1000000f,
    val ropeType: String = "proportional",
    val ropeFactor: Float = 1.0f,
    val partialRotaryFactor: Float = 0.5f,
    val originalMaxPositionEmbeddings: Int = 8192,
    val kvSharedLayers: Int = 20,
    val maxPositionEmbeddings: Int = 131072,
    val perLayerHiddenSize: Int = 0
) {
    /** Total query dimension (numAttentionHeads * headDim) for sliding layers. */
    public val queryDim: Int get() = numAttentionHeads * headDim

    /** Total key-value dimension (numKvHeads * headDim) for sliding layers. */
    public val kvDim: Int get() = numKvHeads * headDim

    /** Total query dimension for global layers. */
    public val globalQueryDim: Int get() = numAttentionHeads * globalHeadDim

    /** Total key-value dimension for global layers. */
    public val globalKvDim: Int get() = numKvHeads * globalHeadDim

    /** Number of query heads per KV head for GQA. */
    public val numHeadsPerKv: Int get() = numAttentionHeads / numKvHeads

    /**
     * Returns the layer type at the given layer index.
     */
    public fun getLayerType(layerIdx: Int): LayerType {
        val type = layerTypes.getOrElse(layerIdx) { "full_attention" }
        return when (type) {
            "sliding_attention" -> LayerType.SLIDING
            "full_attention" -> LayerType.GLOBAL
            else -> LayerType.SLIDING
        }
    }

    /** Returns whether the layer uses sliding window attention. */
    public fun isLocalLayer(layerIdx: Int): Boolean = getLayerType(layerIdx) == LayerType.SLIDING

    /** Returns whether the layer uses global full attention. */
    public fun isGlobalLayer(layerIdx: Int): Boolean = getLayerType(layerIdx) == LayerType.GLOBAL

    /**
     * Returns the RoPE base frequency for the given layer.
     */
    public fun getRopeBase(layerIdx: Int): Float {
        return when (getLayerType(layerIdx)) {
            LayerType.GLOBAL -> ropeBaseGlobal
            LayerType.SLIDING -> ropeBaseLocal
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

    /**
     * Returns the Q projection dimension for the given layer.
     */
    public fun getQueryDim(layerIdx: Int): Int = numAttentionHeads * getHeadDim(layerIdx)

    /**
     * Returns the KV projection dimension for the given layer.
     */
    public fun getKvDim(layerIdx: Int): Int = numKvHeads * getHeadDim(layerIdx)

    /**
     * Returns the partial rotary dimension for the given layer.
     */
    public fun getRotaryDim(layerIdx: Int): Int {
        return when (getLayerType(layerIdx)) {
            LayerType.GLOBAL -> (globalHeadDim * partialRotaryFactor).toInt()
            LayerType.SLIDING -> headDim // full rotation for sliding layers
        }
    }

    /** Returns whether the given layer shares its KV cache. */
    public fun isKvShared(layerIdx: Int): Boolean {
        return layerIdx >= (numLayers - kvSharedLayers)
    }

    /** Returns the cache layer index for the given layer. */
    public fun getCacheLayerIndex(layerIdx: Int): Int {
        return if (isKvShared(layerIdx)) {
            numLayers - kvSharedLayers
        } else {
            layerIdx
        }
    }

    /** Returns the effective number of cache layers (accounting for sharing). */
    public val effectiveCacheLayers: Int
        get() = (numLayers - kvSharedLayers) + 1

    public companion object {

        /**
         * Create config from loaded model metadata.
         */
        public fun fromMetadata(metadata: Gemma4ModelMetadata): Gemma4Config {
            return Gemma4Config(
                hiddenSize = metadata.embeddingLength,
                numLayers = metadata.blockCount,
                numAttentionHeads = metadata.headCount,
                numKvHeads = metadata.kvHeadCount,
                headDim = metadata.headDim,
                globalHeadDim = metadata.globalHeadDim,
                intermediateSize = metadata.intermediateSize,
                slidingWindow = metadata.slidingWindow,
                layerTypes = metadata.layerTypes,
                ropeBaseLocal = metadata.ropeParametersSliding.base,
                ropeBaseGlobal = metadata.ropeParametersFull.base,
                ropeType = metadata.ropeParametersFull.ropeType,
                ropeFactor = metadata.ropeParametersFull.factor,
                partialRotaryFactor = metadata.ropeParametersFull.partialRotaryFactor,
                originalMaxPositionEmbeddings = metadata.ropeParametersFull.originalMaxPositionEmbeddings,
                kvSharedLayers = metadata.kvSharedLayers,
                maxPositionEmbeddings = metadata.maxPositionEmbeddings,
                perLayerHiddenSize = metadata.perLayerEmbeddingLength
            )
        }

        /**
         * Default configuration for Gemma 4 E2B model.
         */
        public val E2B_DEFAULT: Gemma4Config = Gemma4Config(
            hiddenSize = 2304,
            numLayers = 34,
            numAttentionHeads = 8,
            numKvHeads = 4,
            headDim = 256,
            globalHeadDim = 256,
            intermediateSize = 9216,
            slidingWindow = 512,
            ropeBaseLocal = 10000f,
            ropeBaseGlobal = 1000000f,
            ropeType = "proportional",
            partialRotaryFactor = 0.5f,
            kvSharedLayers = 20,
            maxPositionEmbeddings = 131072,
            layerTypes = buildE2BLayerTypes()
        )

        /**
         * Default configuration for Gemma 4 E4B model.
         */
        public val E4B_DEFAULT: Gemma4Config = Gemma4Config(
            hiddenSize = 2560,
            numLayers = 42,
            numAttentionHeads = 10,
            numKvHeads = 5,
            headDim = 256,
            globalHeadDim = 256,
            intermediateSize = 10240,
            slidingWindow = 512,
            ropeBaseLocal = 10000f,
            ropeBaseGlobal = 1000000f,
            ropeType = "proportional",
            partialRotaryFactor = 0.5f,
            kvSharedLayers = 20,
            maxPositionEmbeddings = 131072,
            layerTypes = buildLayerTypes(42)
        )

        private fun buildE2BLayerTypes(): List<String> = buildLayerTypes(34)

        private fun buildLayerTypes(blockCount: Int): List<String> {
            return List(blockCount) { idx ->
                if (idx == blockCount - 1) {
                    "full_attention"
                } else if ((idx + 1) % 6 == 0) {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            }
        }
    }
}
