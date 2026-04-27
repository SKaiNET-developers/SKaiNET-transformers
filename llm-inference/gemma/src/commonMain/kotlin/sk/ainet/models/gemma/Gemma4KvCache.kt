package sk.ainet.models.gemma

import sk.ainet.apps.llm.KvCache

/**
 * KV cache interface for Gemma 4 with layer sharing support.
 */
public interface Gemma4KvCache : KvCache

/**
 * Heap-based KV cache implementation for Gemma 4 with layer sharing.
 *
 * Supports KV cache sharing for the last N layers and per-layer
 * varying KV dimensions (global_head_dim vs head_dim).
 *
 * For simplicity, the cache allocates using the maximum KV dimension
 * across all layers. This wastes a small amount of memory for sliding
 * layers when globalHeadDim > headDim, but avoids per-layer dimension
 * tracking complexity.
 *
 * @param nLayers Total number of transformer layers
 * @param seqLen Maximum sequence length
 * @param kvDim Maximum KV dimension (max(nKvHeads * headDim, nKvHeads * globalHeadDim))
 * @param kvSharedLayers Number of last layers that share KV cache
 */
public class HeapGemma4KvCache(
    override val nLayers: Int,
    override val seqLen: Int,
    override val kvDim: Int,
    private val kvSharedLayers: Int
) : Gemma4KvCache {

    private val effectiveLayers = (nLayers - kvSharedLayers) + 1

    private val keyCache = FloatArray(effectiveLayers * seqLen * kvDim)
    private val valueCache = FloatArray(effectiveLayers * seqLen * kvDim)

    private fun getCacheLayerIndex(layerIdx: Int): Int {
        return if (layerIdx >= (nLayers - kvSharedLayers)) {
            nLayers - kvSharedLayers
        } else {
            layerIdx
        }
    }

    override fun store(
        layerIdx: Int,
        position: Int,
        keys: FloatArray,
        keysOffset: Int,
        values: FloatArray,
        valuesOffset: Int
    ) {
        val cacheLayer = getCacheLayerIndex(layerIdx)
        val base = (cacheLayer * seqLen + position) * kvDim
        keys.copyInto(keyCache, base, keysOffset, keysOffset + kvDim)
        values.copyInto(valueCache, base, valuesOffset, valuesOffset + kvDim)
    }

    override fun getKey(layerIdx: Int, position: Int, headOffset: Int, elementIdx: Int): Float {
        val cacheLayer = getCacheLayerIndex(layerIdx)
        val index = (cacheLayer * seqLen + position) * kvDim + headOffset + elementIdx
        return keyCache[index]
    }

    override fun getValue(layerIdx: Int, position: Int, headOffset: Int, elementIdx: Int): Float {
        val cacheLayer = getCacheLayerIndex(layerIdx)
        val index = (cacheLayer * seqLen + position) * kvDim + headOffset + elementIdx
        return valueCache[index]
    }

    override fun reset() {
        keyCache.fill(0f)
        valueCache.fill(0f)
    }

    public val keyArray: FloatArray get() = keyCache
    public val valueArray: FloatArray get() = valueCache

    public companion object {
        public fun fromConfig(config: Gemma4Config, seqLen: Int): HeapGemma4KvCache {
            // Use max of kvDim and globalKvDim to accommodate both layer types
            val maxKvDim = maxOf(config.kvDim, config.globalKvDim)
            return HeapGemma4KvCache(
                nLayers = config.numLayers,
                seqLen = seqLen,
                kvDim = maxKvDim,
                kvSharedLayers = config.kvSharedLayers
            )
        }
    }
}

/**
 * Factory function to create the optimal KV cache for the current platform.
 */
public fun createOptimalGemma4KvCache(config: Gemma4Config, seqLen: Int): Gemma4KvCache {
    return HeapGemma4KvCache.fromConfig(config, seqLen)
}
