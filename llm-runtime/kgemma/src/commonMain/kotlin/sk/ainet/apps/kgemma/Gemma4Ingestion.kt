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
 */
public data class Gemma4LoadConfig(
    val quantPolicy: QuantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
    val allowQuantized: Boolean = false
)

/**
 * Facade for loading Gemma 4 models from GGUF and SafeTensors files.
 *
 * Runtime path: `gemmaNetwork()` + [OptimizedLLMRuntime] (DSL). Numerical
 * parity against the historical hand-coded runtime was proven in Phase 5d
 * (≤ 8.94e-8 across 1-layer global, mixed sliding+global, and shared-KV
 * configurations); the hand-coded path has since been retired.
 *
 * `loadDslRuntime*` currently requires `DEQUANTIZE_TO_FP32` for K-series
 * quants (~20 GB for Gemma 4 E2B Q4_K_M until a K-kernel lands). Q4_0/Q8_0
 * stay packed via the NATIVE path (see `loadDslRuntimeNative*` in the JVM
 * sibling file).
 */
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

    public suspend fun loadFromSafeTensors(indexPath: String): Gemma4RuntimeWeights<T> {
        return loadGemma4RuntimeWeightsFromSafeTensors(ctx, indexPath, dtype)
    }

    // --- DSL path: gemmaNetwork() + OptimizedLLMRuntime ---

    /**
     * Load a Gemma 4 GGUF via sequential [Source] and return the DSL-based
     * [InferenceRuntime]. Requires `QuantPolicy.DEQUANTIZE_TO_FP32` at the
     * moment for K-series quants.
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
                "The DSL / ComputeGraph path does not yet consume K-series quantized tensors " +
                "directly. Use the NATIVE DSL path (loadDslRuntimeNative*) for Q4_0/Q8_0."
        }
    }
}
