package sk.ainet.models.bitnet

import kotlinx.io.Source
import sk.ainet.apps.llm.DTypePolicyValidation
import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.io.weights.MappingConfig
import sk.ainet.io.weights.WeightMapper
import sk.ainet.io.weights.WeightTensor
import sk.ainet.lang.nn.Module
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.DTypePolicy
import sk.ainet.models.llama.DECODER_NARROW_KEEP_NATIVE
import sk.ainet.models.llama.DecoderGgufWeightLoader
import sk.ainet.models.llama.DecoderGgufWeights
import kotlin.jvm.JvmName

/**
 * End-to-end loader that builds a [bitnetNetwork] module and populates it with weights from a
 * GGUF file via [WeightMapper] + [BitNetGGUFNameResolver].
 *
 * BitNet uses the Llama-family GGUF layout plus per-layer `attn_sub_norm` / `ffn_sub_norm`
 * tensors, so loading delegates to [DecoderGgufWeightLoader] with the BitNet name resolver.
 * 2B4T ties `output.weight` to `token_embd.weight`; the decoder loader's tied-embeddings
 * fallback covers that.
 *
 * Baseline scope (transformers#336): F32/F16/BF16 GGUFs load exactly. A ternary **I2_S** GGUF is
 * the engine loader's job — `StreamingGgufParametersLoader(i2sLayout = …)` materializes packed
 * `BITNET_B1_58` tensors (SKaiNET#1140) that dispatch straight to the ternary kernels; wiring
 * that path through this loader (and `RequantizeTo(BITNET_PLANES)` for `output.weight`) is
 * transformers#337.
 */
@PublishedApi
internal val BITNET_ARCHITECTURES: Set<String> = setOf("bitnet", "bitnet-25", "bitnet-b1.58")

public class BitNetNetworkLoader @PublishedApi internal constructor(
    @PublishedApi internal val weightsProvider: WeightsProvider,
    @PublishedApi internal val debug: Boolean = false
) {
    /** See [LlamaNetworkLoader.dtypePolicy]. */
    public var dtypePolicy: DTypePolicy = DTypePolicy.Any
        private set

    /** See [LlamaNetworkLoader.withDtypePolicy]. */
    public fun withDtypePolicy(policy: DTypePolicy): BitNetNetworkLoader {
        DTypePolicyValidation.validate(
            policy, "BitNetNetworkLoader.withDtypePolicy", keepNative = DECODER_NARROW_KEEP_NATIVE,
        )
        this.dtypePolicy = policy
        return this
    }

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
        ): BitNetNetworkLoader = BitNetNetworkLoader(
            WeightsProvider.GgufSource(sourceProvider, quantPolicy), debug
        )

        /** Load from a GGUF file via streaming RandomAccessSource (any size). */
        @JvmName("fromGgufRandomAccess")
        public fun fromGguf(
            randomAccessProvider: () -> RandomAccessSource,
            quantPolicy: QuantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
            debug: Boolean = false
        ): BitNetNetworkLoader = BitNetNetworkLoader(
            WeightsProvider.GgufRandomAccess(randomAccessProvider, quantPolicy), debug
        )

        /** Build from already-loaded [DecoderGgufWeights] (GGUF-canonical tensor names). */
        public inline fun <reified T : DType, V> fromWeights(
            weights: DecoderGgufWeights<T, V>,
            debug: Boolean = false
        ): Module<T, V> = BitNetNetworkLoader(
            WeightsProvider.Preloaded(weights), debug
        ).applyWeightsToNetwork(weights)
    }

    /**
     * Load weights and build a fully initialized DSL network.
     *
     * @throws IllegalArgumentException if required weights could not be mapped.
     */
    public suspend inline fun <reified T : DType, V> load(
        ctx: ExecutionContext
    ): Module<T, V> {
        val weights: DecoderGgufWeights<T, V> = when (val wp = weightsProvider) {
            is WeightsProvider.GgufSource -> {
                val loader = DecoderGgufWeightLoader(
                    wp.sourceProvider,
                    quantPolicy = wp.quantPolicy,
                    acceptedArchitectures = BITNET_ARCHITECTURES,
                    dtypePolicy = dtypePolicy,
                )
                loader.loadToMap<T, V>(ctx)
            }
            is WeightsProvider.GgufRandomAccess -> {
                val loader = DecoderGgufWeightLoader(
                    wp.randomAccessProvider,
                    quantPolicy = wp.quantPolicy,
                    acceptedArchitectures = BITNET_ARCHITECTURES,
                    dtypePolicy = dtypePolicy,
                )
                loader.loadToMapStreaming<T, V>(ctx)
            }
            is WeightsProvider.Preloaded<*, *> -> {
                @Suppress("UNCHECKED_CAST")
                wp.weights as DecoderGgufWeights<T, V>
            }
        }

        return applyWeightsToNetwork(weights)
    }

    /** Build the DSL network from metadata and map all weights. */
    @PublishedApi
    internal inline fun <reified T : DType, V> applyWeightsToNetwork(
        weights: DecoderGgufWeights<T, V>
    ): Module<T, V> {
        val model = bitnetNetwork<T, V>(weights.metadata)

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
            nameResolver = BitNetGGUFNameResolver()
        )

        val result = WeightMapper.applyWeights(model, weightTensors, config)

        // BitNet has no bias tensors; zero-initialized bias params stay unmapped by design.
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
