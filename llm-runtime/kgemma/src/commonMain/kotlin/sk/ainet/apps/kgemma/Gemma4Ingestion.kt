@file:Suppress("DEPRECATION") // Retains the Gemma4Runtime path (loadRuntime*) for backwards compat alongside the new DSL path (loadDslRuntime*).

package sk.ainet.apps.kgemma

import kotlin.random.Random
import kotlin.reflect.KClass
import kotlinx.io.Source
import sk.ainet.apps.llm.InferenceRuntime
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import sk.ainet.models.gemma.Gemma4AttentionBackend
import sk.ainet.models.gemma.Gemma4Config
import sk.ainet.models.gemma.Gemma4Runtime
import sk.ainet.models.gemma.Gemma4RuntimeWeights
import sk.ainet.models.gemma.Gemma4SafeTensorsWeightLoader
import sk.ainet.models.gemma.Gemma4WeightLoader
import sk.ainet.models.gemma.Gemma4Weights
import sk.ainet.models.gemma.GemmaNetworkLoader
import sk.ainet.models.gemma.createOptimalGemma4KvCache
import sk.ainet.models.gemma.loadGemma4RuntimeWeights
import sk.ainet.models.gemma.loadGemma4RuntimeWeightsFromSafeTensors
import sk.ainet.models.gemma.loadGemma4RuntimeWeightsStreaming

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
 * Provides two runtime paths:
 *
 * - `loadRuntime*` → [Gemma4Runtime] (hand-coded, **deprecated** but still
 *   the only path that can run Q4/Q8 weights via `NATIVE_OPTIMIZED` policy;
 *   keep using it for RAM-constrained loads of real Gemma 4 checkpoints).
 * - `loadDslRuntime*` → `gemmaNetwork()` + [OptimizedLLMRuntime] (DSL path,
 *   matches the hand-coded runtime at machine precision per
 *   `GemmaRuntimeParityTest`). Requires `DEQUANTIZE_TO_FP32` today and so
 *   needs roughly 20 GB RAM for Gemma 4 E2B; becomes the primary path
 *   once quant-aware DAG kernels land — see `ISSUE-skainet-8b-oom.md`.
 */
public class Gemma4Ingestion<T : DType>(
    private val ctx: ExecutionContext,
    private val dtype: KClass<T>,
    private val config: Gemma4LoadConfig = Gemma4LoadConfig()
) {

    // --- Hand-coded Gemma4Runtime path (kept for RAM-constrained loads) ---

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

    // --- DSL path: gemmaNetwork() + OptimizedLLMRuntime (Phase 5d parity) ---

    /**
     * Load a Gemma 4 GGUF via sequential [Source] and return the DSL-based
     * [InferenceRuntime]. Same weight bytes as [loadRuntime], different
     * execution path. Requires `QuantPolicy.DEQUANTIZE_TO_FP32` at the moment.
     */
    public suspend fun loadDslRuntime(sourceProvider: () -> Source): InferenceRuntime<T> {
        requireDequantPolicy("loadDslRuntime")
        val weights = Gemma4WeightLoader(
            sourceProvider = sourceProvider,
            quantPolicy = config.quantPolicy
        ).loadToMap<T, Float>(ctx, dtype)
        return buildDslRuntime(weights)
    }

    /** Same as [loadDslRuntime] but via a random-access source (models > 2 GB). */
    public suspend fun loadDslRuntimeStreaming(randomAccessProvider: () -> RandomAccessSource): InferenceRuntime<T> {
        requireDequantPolicy("loadDslRuntimeStreaming")
        val weights = Gemma4WeightLoader(
            randomAccessProvider = randomAccessProvider,
            quantPolicy = config.quantPolicy
        ).loadToMapStreaming<T, Float>(ctx, dtype)
        return buildDslRuntime(weights)
    }

    /** Same as [loadDslRuntime] but reads a sharded SafeTensors model by index file. */
    public suspend fun loadDslRuntimeFromSafeTensors(indexPath: String): InferenceRuntime<T> {
        @Suppress("UNCHECKED_CAST")
        val weights = Gemma4SafeTensorsWeightLoader(indexPath).loadToMap(ctx, dtype) as Gemma4Weights<T, Float>
        return buildDslRuntime(weights)
    }

    /**
     * Builds a DSL runtime from pre-loaded raw Gemma weights. Shared with the
     * `loadDslRuntime*` entry points and also useful for synthetic-weight
     * tests where the caller already constructed a [Gemma4Weights] map.
     */
    public fun buildDslRuntime(weights: Gemma4Weights<T, Float>): InferenceRuntime<T> {
        val model = GemmaNetworkLoader.fromWeights(ctx, weights, dtype)
        return OptimizedLLMRuntime(model, ctx, OptimizedLLMMode.DIRECT, dtype, random = Random.Default)
    }

    private fun requireDequantPolicy(caller: String) {
        require(config.quantPolicy == QuantPolicy.DEQUANTIZE_TO_FP32) {
            "$caller currently only supports QuantPolicy.DEQUANTIZE_TO_FP32. " +
                "The DSL / ComputeGraph path does not yet consume quantized tensors directly " +
                "(see ISSUE-skainet-8b-oom.md §Solution C). Use loadRuntime* (Gemma4Runtime) if " +
                "you need NATIVE_OPTIMIZED or RAW_BYTES."
        }
    }
}
