package sk.ainet.models.llama

import kotlinx.io.Source
import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.io.weights.LlamaGGUFNameResolver
import sk.ainet.io.weights.MappingConfig
import sk.ainet.io.weights.WeightMapper
import sk.ainet.io.weights.WeightTensor
import sk.ainet.lang.nn.Module
import sk.ainet.lang.types.DType
import kotlin.jvm.JvmName

/**
 * End-to-end loader that builds a `llamaNetwork()` module and populates it
 * with weights from GGUF or SafeTensors files via [WeightMapper] + [LlamaGGUFNameResolver].
 *
 * Both formats are normalized to GGUF-canonical tensor names internally,
 * so the same resolver handles both.
 *
 * Usage:
 * ```kotlin
 * // From GGUF (streaming)
 * val model = LlamaNetworkLoader.fromGguf(randomAccessProvider = { rasSource })
 *     .load<FP32, Float>(ctx)
 *
 * // From GGUF (sequential, models <2GB)
 * val model = LlamaNetworkLoader.fromGguf(sourceProvider = { fileSource })
 *     .load<FP32, Float>(ctx)
 *
 * // From SafeTensors
 * val model = LlamaNetworkLoader.fromSafeTensors(metadata, randomAccessProvider = { rasSource })
 *     .load<FP32, Float>(ctx)
 *
 * // From pre-loaded weights
 * val model = LlamaNetworkLoader.fromWeights(llamaWeights, debug = true)
 * ```
 */
public class LlamaNetworkLoader @PublishedApi internal constructor(
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

        data class SafeTensors(
            val randomAccessProvider: () -> RandomAccessSource,
            val metadata: LlamaModelMetadata,
            val tiedEmbeddings: Boolean
        ) : WeightsProvider

        data class Preloaded<T : DType, V>(
            val weights: DecoderGgufWeights<T, V>
        ) : WeightsProvider
    }

    public companion object {
        /** Load from a GGUF file via sequential Source (models under 2GB). */
        public fun fromGguf(
            sourceProvider: () -> Source,
            quantPolicy: QuantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
            debug: Boolean = false
        ): LlamaNetworkLoader = LlamaNetworkLoader(
            WeightsProvider.GgufSource(sourceProvider, quantPolicy), debug
        )

        /** Load from a GGUF file via streaming RandomAccessSource (any size). */
        @JvmName("fromGgufRandomAccess")
        public fun fromGguf(
            randomAccessProvider: () -> RandomAccessSource,
            quantPolicy: QuantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
            debug: Boolean = false
        ): LlamaNetworkLoader = LlamaNetworkLoader(
            WeightsProvider.GgufRandomAccess(randomAccessProvider, quantPolicy), debug
        )

        /** Load from a SafeTensors file. Requires metadata (not embedded in SafeTensors). */
        public fun fromSafeTensors(
            metadata: LlamaModelMetadata,
            randomAccessProvider: () -> RandomAccessSource,
            tiedEmbeddings: Boolean = false,
            debug: Boolean = false
        ): LlamaNetworkLoader = LlamaNetworkLoader(
            WeightsProvider.SafeTensors(randomAccessProvider, metadata, tiedEmbeddings), debug
        )

        /** Build from already-loaded [DecoderGgufWeights] (GGUF-canonical tensor names). */
        public inline fun <reified T : DType, V> fromWeights(
            weights: DecoderGgufWeights<T, V>,
            debug: Boolean = false
        ): Module<T, V> = LlamaNetworkLoader(
            WeightsProvider.Preloaded(weights), debug
        ).applyWeightsToNetwork(weights)
    }

    /**
     * Load weights and build a fully initialized DSL network.
     *
     * @return A [Module] with all weights populated from the model file.
     * @throws IllegalArgumentException if required weights could not be mapped.
     */
    public suspend inline fun <reified T : DType, V> load(
        ctx: ExecutionContext
    ): Module<T, V> {
        val weights: DecoderGgufWeights<T, V> = when (val wp = weightsProvider) {
            is WeightsProvider.GgufSource -> {
                val loader = DecoderGgufWeightLoader(wp.sourceProvider, quantPolicy = wp.quantPolicy)
                loader.loadToMap<T, V>(ctx)
            }
            is WeightsProvider.GgufRandomAccess -> {
                val loader = DecoderGgufWeightLoader(wp.randomAccessProvider, quantPolicy = wp.quantPolicy)
                loader.loadToMapStreaming<T, V>(ctx)
            }
            is WeightsProvider.SafeTensors -> {
                val loader = DecoderSafeTensorsLoader<T>(ctx, T::class, wp.metadata, wp.tiedEmbeddings)
                @Suppress("UNCHECKED_CAST")
                loader.loadToMap(wp.randomAccessProvider) as DecoderGgufWeights<T, V>
            }
            is WeightsProvider.Preloaded<*, *> -> {
                @Suppress("UNCHECKED_CAST")
                wp.weights as DecoderGgufWeights<T, V>
            }
        }

        return applyWeightsToNetwork(weights)
    }

    /**
     * Build the DSL network from metadata and map all weights.
     */
    @PublishedApi
    internal inline fun <reified T : DType, V> applyWeightsToNetwork(
        weights: DecoderGgufWeights<T, V>
    ): Module<T, V> {
        val model = llamaNetwork<T, V>(weights.metadata)

        val weightTensors = weights.tensors.map { (name, tensor) ->
            WeightTensor(
                name = name,
                shape = tensor.shape.dimensions.toList(),
                tensor = tensor
            )
        }

        // Both GGUF and SafeTensors loaders normalize to GGUF-canonical names
        val config = MappingConfig(
            usePathBasedMatching = false,
            fallbackToShapeMatching = false,
            debug = debug,
            nameResolver = LlamaGGUFNameResolver()
        )

        val result = WeightMapper.applyWeights(model, weightTensors, config)

        // Llama has no bias tensors. The DSL's dense() creates Linear modules
        // with zero-initialized bias params — these are expected to be unmapped.
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
}
