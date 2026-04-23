package sk.ainet.models.gemma

import kotlinx.io.Source
import sk.ainet.apps.llm.weights.LlamaGGUFNameResolver
import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.io.weights.MappingConfig
import sk.ainet.io.weights.WeightMapper
import sk.ainet.io.weights.WeightTensor
import sk.ainet.lang.nn.Module
import sk.ainet.lang.types.DType

/**
 * End-to-end loader that builds a [gemmaNetwork] module and populates it
 * with weights from GGUF or SafeTensors files. Phase 5a of
 * PLAN-unified-pipeline.md — feeds the unified DSL → ComputeGraph → CPU-on-JVM
 * execution path.
 *
 * Uses [LlamaGGUFNameResolver] because Gemma 4 GGUF tensor names follow the
 * standard LLaMA block layout (`blk.N.attn_*`, `blk.N.ffn_*`, etc.) and the
 * DSL module parameter names (`gate_proj`, `up_proj`, `down_proj`, `q_proj`,
 * `k_proj`, `v_proj`, `o_proj`, `attn_norm`, `ffn_norm`, `output_norm`,
 * `output`, `token_embd`) resolve to those canonical names unchanged.
 *
 * Usage:
 * ```kotlin
 * val model = GemmaNetworkLoader.fromGguf(randomAccessProvider = { ras })
 *     .load<FP32, Float>(ctx)
 * ```
 */
public class GemmaNetworkLoader @PublishedApi internal constructor(
    @PublishedApi internal val weightsProvider: WeightsProvider,
    @PublishedApi internal val debug: Boolean = false
) {
    @PublishedApi
    internal sealed interface WeightsProvider {
        data class GgufSource(
            val sourceProvider: () -> Source,
            val quantPolicy: QuantPolicy
        ) : WeightsProvider

        data class GgufRandomAccess(
            val randomAccessProvider: () -> RandomAccessSource,
            val quantPolicy: QuantPolicy
        ) : WeightsProvider

        data class SafeTensorsIndex(
            val indexPath: String
        ) : WeightsProvider

        data class Preloaded<T : DType, V>(
            val weights: Gemma4Weights<T, V>
        ) : WeightsProvider
    }

    public companion object {
        /** Load from a GGUF file via sequential Source (models under 2GB). */
        public fun fromGguf(
            sourceProvider: () -> Source,
            quantPolicy: QuantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
            debug: Boolean = false
        ): GemmaNetworkLoader = GemmaNetworkLoader(
            WeightsProvider.GgufSource(sourceProvider, quantPolicy), debug
        )

        /** Load from a GGUF file via streaming RandomAccessSource (any size). */
        @kotlin.jvm.JvmName("fromGgufRandomAccess")
        public fun fromGguf(
            randomAccessProvider: () -> RandomAccessSource,
            quantPolicy: QuantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
            debug: Boolean = false
        ): GemmaNetworkLoader = GemmaNetworkLoader(
            WeightsProvider.GgufRandomAccess(randomAccessProvider, quantPolicy), debug
        )

        /** Load from HuggingFace SafeTensors with index file. */
        public fun fromSafeTensors(
            indexPath: String,
            debug: Boolean = false
        ): GemmaNetworkLoader = GemmaNetworkLoader(
            WeightsProvider.SafeTensorsIndex(indexPath), debug
        )

        /** Build from pre-loaded [Gemma4Weights]. */
        public inline fun <reified T : DType, V> fromWeights(
            ctx: ExecutionContext,
            weights: Gemma4Weights<T, V>,
            debug: Boolean = false
        ): Module<T, V> = fromWeights(ctx, weights, T::class, debug)

        /**
         * Non-reified variant — explicit [dtype] so non-reified callers
         * (e.g. `Gemma4Ingestion<T>`) can build the DSL network without
         * propagating reification through their public API.
         */
        public fun <T : DType, V> fromWeights(
            ctx: ExecutionContext,
            weights: Gemma4Weights<T, V>,
            dtype: kotlin.reflect.KClass<T>,
            debug: Boolean = false
        ): Module<T, V> = applyWeightsToNetworkNonReified(ctx, weights, dtype, debug)
    }

    /**
     * Load weights and build a fully initialized DSL network.
     */
    public suspend inline fun <reified T : DType, V> load(
        ctx: ExecutionContext
    ): Module<T, V> {
        val weights: Gemma4Weights<T, V> = when (val wp = weightsProvider) {
            is WeightsProvider.GgufSource -> {
                val loader = Gemma4WeightLoader(wp.sourceProvider, quantPolicy = wp.quantPolicy)
                loader.loadToMap<T, V>(ctx)
            }
            is WeightsProvider.GgufRandomAccess -> {
                val loader = Gemma4WeightLoader(wp.randomAccessProvider, quantPolicy = wp.quantPolicy)
                loader.loadToMapStreaming<T, V>(ctx)
            }
            is WeightsProvider.SafeTensorsIndex -> {
                val loader = Gemma4SafeTensorsWeightLoader(wp.indexPath)
                @Suppress("UNCHECKED_CAST")
                loader.loadToMap(ctx, T::class) as Gemma4Weights<T, V>
            }
            is WeightsProvider.Preloaded<*, *> -> {
                @Suppress("UNCHECKED_CAST")
                wp.weights as Gemma4Weights<T, V>
            }
        }

        return applyWeightsToNetwork(ctx, weights)
    }

    @PublishedApi
    internal inline fun <reified T : DType, V> applyWeightsToNetwork(
        ctx: ExecutionContext,
        weights: Gemma4Weights<T, V>
    ): Module<T, V> = applyWeightsToNetworkNonReified(ctx, weights, T::class, debug)
}

/** Shared non-reified impl used by both the inline-reified companion helpers
 *  and the DSL-ingestion entry points that only have a runtime `KClass<T>`. */
@PublishedApi
internal fun <T : DType, V> applyWeightsToNetworkNonReified(
    ctx: ExecutionContext,
    weights: Gemma4Weights<T, V>,
    dtype: kotlin.reflect.KClass<T>,
    debug: Boolean
): Module<T, V> {
    // Enable optional Gemma 4 features iff the checkpoint actually carries
    // their weights. Real Gemma 4 GGUFs do; synthetic toy-model tests do not,
    // and forcing them on would make the WeightMapper strict-check fail.
    val hasQKNorm = weights.tensors.keys.any { it.endsWith(".attn_q_norm.weight") }
    val hasSandwichNorms = weights.tensors.keys.any { it.endsWith(".post_attention_norm.weight") }
    // Phase 5f.6 TODO: layer_output_scale and PLE both REGRESS real-model
    // output from real English words ("Hi relieved desired…" after 5f.3)
    // to degenerate repetition ("Hi ? ?" with scale only, "Hi pełni" with
    // PLE). One or both has an implementation bug — diagnosed in
    // commits/logs (search for DIAG flags). Disable both by default on
    // real-model checkpoints until a parity-level fix lands.
    val hasLayerOutputScale = false
    val hasPle = false
    val model = gemmaNetwork<T, V>(
        weights.metadata,
        dtype,
        qkNorm = hasQKNorm,
        sandwichNorms = hasSandwichNorms,
        layerOutputScale = hasLayerOutputScale,
        ple = hasPle
    )

    val weightTensors = weights.tensors.map { (name, tensor) ->
        WeightTensor(
            name = name,
            shape = tensor.shape.dimensions.toList(),
            tensor = tensor
        )
    }

    val config = MappingConfig(
        usePathBasedMatching = false,
        fallbackToShapeMatching = false,
        debug = debug,
        nameResolver = LlamaGGUFNameResolver()
    )

    val result = WeightMapper.applyWeights(model, weightTensors, config)

    // Gemma has no bias tensors. Allow unmapped bias params from dense().
    val unmappedNonBias = result.missingParams.filter { !it.contains(".bias") }
    require(unmappedNonBias.isEmpty()) {
        buildString {
            appendLine("Failed to map ${unmappedNonBias.size} weight parameters:")
            unmappedNonBias.forEach { appendLine("  - $it") }
            if (result.unusedTensors.isNotEmpty()) {
                appendLine("Unused tensors (${result.unusedTensors.size}):")
                result.unusedTensors.take(10).forEach { appendLine("  - $it") }
            }
        }.trim()
    }

    return model
}
