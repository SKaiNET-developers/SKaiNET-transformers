@file:Suppress("DEPRECATION") // Gemma4Runtime is deprecated in favour of gemmaNetwork() + OptimizedLLMRuntime (Phase 5d). Migrating this CLI loader is a follow-up — see PLAN-unified-pipeline.md.

package sk.ainet.apps.kgemma

import kotlinx.io.Source
import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.models.gemma.Gemma4AttentionBackend
import sk.ainet.models.gemma.Gemma4Config
import sk.ainet.models.gemma.Gemma4Runtime
import sk.ainet.models.gemma.Gemma4RuntimeWeights
import sk.ainet.models.gemma.createOptimalGemma4KvCache
import sk.ainet.io.model.QuantPolicy
import sk.ainet.models.gemma.loadGemma4RuntimeWeights
import sk.ainet.models.gemma.loadGemma4RuntimeWeightsFromSafeTensors
import sk.ainet.models.gemma.loadGemma4RuntimeWeightsStreaming
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Load configuration for Gemma 4 models.
 */
public data class Gemma4LoadConfig(
    val quantPolicy: QuantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
    val allowQuantized: Boolean = false
)

/**
 * Facade for loading Gemma 4 models from GGUF and SafeTensors files.
 *
 * Mirrors [Gemma3nIngestion] but uses Gemma 4 config, weights, attention,
 * and runtime classes.
 */
public class Gemma4Ingestion<T : DType>(
    private val ctx: ExecutionContext,
    private val dtype: KClass<T>,
    private val config: Gemma4LoadConfig = Gemma4LoadConfig()
) {

    public suspend fun load(sourceProvider: () -> Source): Gemma4RuntimeWeights<T> {
        return loadGemma4RuntimeWeights(
            ctx = ctx,
            sourceProvider = sourceProvider,
            dtype = dtype,
            quantPolicy = config.quantPolicy,
            allowQuantized = config.allowQuantized
        )
    }

    public suspend fun loadStreaming(randomAccessProvider: () -> RandomAccessSource): Gemma4RuntimeWeights<T> {
        return loadGemma4RuntimeWeightsStreaming(
            ctx = ctx,
            randomAccessProvider = randomAccessProvider,
            dtype = dtype,
            quantPolicy = config.quantPolicy,
            allowQuantized = config.allowQuantized
        )
    }

    public suspend fun loadRuntime(sourceProvider: () -> Source): Gemma4Runtime<T> {
        val weights = load(sourceProvider)
        return buildRuntime(weights)
    }

    public suspend fun loadRuntimeStreaming(randomAccessProvider: () -> RandomAccessSource): Gemma4Runtime<T> {
        val weights = loadStreaming(randomAccessProvider)
        return buildRuntime(weights)
    }

    public suspend fun loadFromSafeTensors(indexPath: String): Gemma4RuntimeWeights<T> {
        return loadGemma4RuntimeWeightsFromSafeTensors(ctx, indexPath, dtype)
    }

    public suspend fun loadRuntimeFromSafeTensors(indexPath: String): Gemma4Runtime<T> {
        val weights = loadFromSafeTensors(indexPath)
        return buildRuntime(weights)
    }

    private fun buildRuntime(weights: Gemma4RuntimeWeights<T>): Gemma4Runtime<T> {
        val modelConfig = Gemma4Config.fromMetadata(weights.metadata)
        // Cap KV cache seqLen to avoid OOM on heap — full 128K/256K requires
        // off-heap or memory-mapped caches (future TurboQuant integration)
        val maxHeapSeqLen = 4096
        val seqLen = minOf(weights.metadata.contextLength, maxHeapSeqLen)
        val kvCache = createOptimalGemma4KvCache(modelConfig, seqLen)
        val attentionBackend = Gemma4AttentionBackend(ctx, weights, dtype, modelConfig, kvCache)
        return Gemma4Runtime(ctx, weights, attentionBackend, dtype, modelConfig)
    }
}
