package sk.ainet.models.apertus

import kotlinx.io.Source
import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.io.weights.LlamaGGUFNameResolver
import sk.ainet.io.weights.MappingConfig
import sk.ainet.io.weights.WeightMapper
import sk.ainet.io.weights.WeightTensor
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.DType

/**
 * End-to-end loader that builds an `apertusNetwork()` module and populates it
 * with weights from GGUF or SafeTensors files.
 *
 * Handles Apertus-specific concerns:
 * - xIELU scalar parameters (stored separately in [ApertusWeights.xieluParams])
 *   are converted to scalar tensors for the weight mapper
 * - QK-Norm weights are mapped via the resolver
 * - Ungated FFN (no gate_proj)
 *
 * Usage:
 * ```kotlin
 * val model = ApertusNetworkLoader.fromGguf(randomAccessProvider = { ras })
 *     .load<FP32, Float>(ctx)
 * ```
 */
public class ApertusNetworkLoader private constructor(
    private val weightsProvider: WeightsProvider,
    private val debug: Boolean = false
) {
    private sealed interface WeightsProvider {
        data class GgufSource(
            val sourceProvider: () -> Source,
            val quantPolicy: QuantPolicy
        ) : WeightsProvider

        data class GgufRandomAccess(
            val randomAccessProvider: () -> RandomAccessSource,
            val quantPolicy: QuantPolicy
        ) : WeightsProvider

        data class SafeTensorsSingle(
            val randomAccessProvider: () -> RandomAccessSource,
            val metadata: ApertusModelMetadata
        ) : WeightsProvider

        data class Preloaded<T : DType, V>(
            val weights: ApertusWeights<T, V>
        ) : WeightsProvider
    }

    public companion object {
        /** Load from a GGUF file via sequential Source (models under 2GB). */
        public fun fromGguf(
            sourceProvider: () -> Source,
            quantPolicy: QuantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
            debug: Boolean = false
        ): ApertusNetworkLoader = ApertusNetworkLoader(
            WeightsProvider.GgufSource(sourceProvider, quantPolicy), debug
        )

        /** Load from a GGUF file via streaming RandomAccessSource (any size). */
        public fun fromGguf(
            randomAccessProvider: () -> RandomAccessSource,
            quantPolicy: QuantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
            debug: Boolean = false
        ): ApertusNetworkLoader = ApertusNetworkLoader(
            WeightsProvider.GgufRandomAccess(randomAccessProvider, quantPolicy), debug
        )

        /** Load from a single SafeTensors file. */
        public fun fromSafeTensors(
            metadata: ApertusModelMetadata,
            randomAccessProvider: () -> RandomAccessSource,
            debug: Boolean = false
        ): ApertusNetworkLoader = ApertusNetworkLoader(
            WeightsProvider.SafeTensorsSingle(randomAccessProvider, metadata), debug
        )

        /** Build from pre-loaded [ApertusWeights]. */
        public inline fun <reified T : DType, V> fromWeights(
            ctx: ExecutionContext,
            weights: ApertusWeights<T, V>,
            debug: Boolean = false
        ): Module<T, V> = ApertusNetworkLoader(
            WeightsProvider.Preloaded(weights), debug
        ).applyWeightsToNetwork(ctx, weights)
    }

    /**
     * Load weights and build a fully initialized DSL network.
     */
    public suspend inline fun <reified T : DType, V> load(
        ctx: ExecutionContext
    ): Module<T, V> {
        val weights: ApertusWeights<T, V> = when (val wp = weightsProvider) {
            is WeightsProvider.GgufSource -> {
                val loader = ApertusWeightLoader.fromSource(wp.sourceProvider, wp.quantPolicy)
                loader.loadToMap<T, V>(ctx)
            }
            is WeightsProvider.GgufRandomAccess -> {
                val loader = ApertusWeightLoader.fromRandomAccess(wp.randomAccessProvider, wp.quantPolicy)
                loader.loadToMap<T, V>(ctx)
            }
            is WeightsProvider.SafeTensorsSingle -> {
                val loader = ApertusSingleSafeTensorsLoader<T>(ctx, T::class, wp.metadata)
                @Suppress("UNCHECKED_CAST")
                loader.loadToMap(wp.randomAccessProvider) as ApertusWeights<T, V>
            }
            is WeightsProvider.Preloaded<*, *> -> {
                @Suppress("UNCHECKED_CAST")
                wp.weights as ApertusWeights<T, V>
            }
        }

        return applyWeightsToNetwork(ctx, weights)
    }

    @PublishedApi
    internal inline fun <reified T : DType, V> applyWeightsToNetwork(
        ctx: ExecutionContext,
        weights: ApertusWeights<T, V>
    ): Module<T, V> {
        val model = apertusNetwork<T, V>(weights.metadata)

        val weightTensors = weights.tensors.map { (name, tensor) ->
            WeightTensor(
                name = name,
                shape = tensor.shape.dimensions.toList(),
                tensor = tensor
            )
        }.toMutableList()

        // Convert xIELU scalar params to tensors with GGUF-canonical names.
        // The resolver expects tensors named "blk.N.mlp.act_fn.{alpha_p,alpha_n,beta,eps}".
        weights.xieluParams.forEach { (layer, params) ->
            fun addScalar(paramSuffix: String, value: Float) {
                val name = "blk.$layer.mlp.act_fn.$paramSuffix"
                val tensor = ctx.fromFloatArray<T, Float>(Shape(1), T::class, floatArrayOf(value))
                @Suppress("UNCHECKED_CAST")
                weightTensors += WeightTensor(
                    name = name,
                    shape = listOf(1),
                    tensor = tensor as sk.ainet.lang.tensor.Tensor<T, V>
                )
            }
            addScalar("alpha_p", params.alphaP)
            addScalar("alpha_n", params.alphaN)
            addScalar("beta", params.beta)
            addScalar("eps", params.eps)
        }

        val config = MappingConfig(
            usePathBasedMatching = false,
            fallbackToShapeMatching = false,
            debug = debug,
            nameResolver = LlamaGGUFNameResolver()
        )

        val result = WeightMapper.applyWeights(model, weightTensors, config)

        // Apertus has no bias tensors. Allow unmapped bias params from dense().
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
