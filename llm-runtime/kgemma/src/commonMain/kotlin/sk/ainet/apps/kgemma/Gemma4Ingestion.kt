package sk.ainet.apps.kgemma

import kotlin.random.Random
import kotlin.reflect.KClass
import kotlinx.io.Source
import sk.ainet.apps.llm.InferenceRuntime
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.types.DType
import sk.ainet.models.gemma.Gemma4RuntimeWeights
import sk.ainet.models.gemma.Gemma4SafeTensorsWeightLoader
import sk.ainet.models.gemma.Gemma4WeightLoader
import sk.ainet.models.gemma.Gemma4Weights
import sk.ainet.models.gemma.GemmaNetworkLoader
import sk.ainet.models.gemma.loadGemma4RuntimeWeights
import sk.ainet.models.gemma.loadGemma4RuntimeWeightsFromSafeTensors
import sk.ainet.models.gemma.loadGemma4RuntimeWeightsStreaming

/**
 * Load configuration for Gemma 4 models.
 *
 * @param weightForm streaming-lane materialization request. `null` (default)
 *   keeps quantized tensors packed in their stored block encoding with
 *   logical `[out, in]` shapes; pass
 *   [sk.ainet.models.gemma.GEMMA_DEQUANTIZE_ALL] for a dense FP32 load
 *   (the export/tracing path). The sequential [Source] lane always
 *   dequantizes.
 */
@OptIn(ExperimentalMemoryApi::class)
public data class Gemma4LoadConfig(
    val weightForm: WeightForm? = null
)

/**
 * Facade for loading Gemma 4 models from GGUF and SafeTensors files.
 *
 * Runtime path: `gemmaNetwork()` + [OptimizedLLMRuntime] (DSL). Numerical
 * parity against the historical hand-coded runtime was proven in Phase 5d
 * (≤ 8.94e-8 across 1-layer global, mixed sliding+global, and shared-KV
 * configurations); the hand-coded path has since been retired.
 *
 * `loadDslRuntime*` route through the engine loader: quantized weights stay
 * packed in their stored block encoding by default and dispatch to the
 * packed matmul kernels via `linearProject` / `matmulWeightTransposed`.
 */
@OptIn(ExperimentalMemoryApi::class)
public class Gemma4Ingestion<T : DType>(
    private val ctx: ExecutionContext,
    private val dtype: KClass<T>,
    private val config: Gemma4LoadConfig = Gemma4LoadConfig()
) {

    // --- Raw weight loading (used by the DSL path and by tests) ---

    public suspend fun load(sourceProvider: () -> Source): Gemma4RuntimeWeights<T> {
        return loadGemma4RuntimeWeights(
            ctx = ctx,
            sourceProvider = sourceProvider,
            dtype = dtype
        )
    }

    public suspend fun loadStreaming(randomAccessProvider: () -> RandomAccessSource): Gemma4RuntimeWeights<T> {
        return loadGemma4RuntimeWeightsStreaming(
            ctx = ctx,
            randomAccessProvider = randomAccessProvider,
            dtype = dtype,
            weightForm = config.weightForm
        )
    }

    public suspend fun loadFromSafeTensors(indexPath: String): Gemma4RuntimeWeights<T> {
        return loadGemma4RuntimeWeightsFromSafeTensors(ctx, indexPath, dtype)
    }

    // --- DSL path: gemmaNetwork() + OptimizedLLMRuntime ---

    /**
     * Load a Gemma 4 GGUF via sequential [Source] (dequantized to dense
     * floats) and return the DSL-based [InferenceRuntime].
     */
    public suspend fun loadDslRuntime(sourceProvider: () -> Source): InferenceRuntime<T> {
        val weights = Gemma4WeightLoader(
            sourceProvider = sourceProvider
        ).loadToMap<T, Float>(ctx, dtype)
        return buildDslRuntime(weights)
    }

    /**
     * Same as [loadDslRuntime] but via a random-access source (any size)
     * through the engine loader — quantized weights stay packed by default
     * (see [Gemma4LoadConfig.weightForm]).
     */
    public suspend fun loadDslRuntimeStreaming(randomAccessProvider: () -> RandomAccessSource): InferenceRuntime<T> {
        val weights = Gemma4WeightLoader(
            randomAccessProvider = randomAccessProvider,
            weightForm = config.weightForm
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
}
