package sk.ainet.models.qwen

import kotlinx.io.Source
import sk.ainet.apps.llm.DTypePolicyValidation
import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.io.weights.MappingConfig
import sk.ainet.io.weights.WeightMapper
import sk.ainet.io.weights.WeightTensor
import sk.ainet.lang.nn.Module
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.DTypePolicy
import sk.ainet.lang.nn.dsl.decoder.GgufDecoderMetadata
import sk.ainet.lang.nn.dsl.decoder.DecoderSafeTensorsLoader
import sk.ainet.lang.nn.dsl.decoder.DecoderGgufWeights
import sk.ainet.lang.nn.dsl.decoder.DECODER_NARROW_KEEP_NATIVE
import kotlin.jvm.JvmName

/**
 * End-to-end loader that builds a `qwenNetwork()` module and populates it
 * with weights from GGUF or SafeTensors files via [WeightMapper] + [QwenGGUFNameResolver].
 *
 * Qwen3 uses the same GGUF tensor naming (`blk.N.*`) and weight structure as LLaMA,
 * so loading delegates to [QwenWeightLoader] for GGUF and [DecoderSafeTensorsLoader]
 * for SafeTensors. The network is built via [qwenNetwork] which delegates to [llamaNetwork].
 *
 * Usage:
 * ```kotlin
 * // From GGUF (streaming)
 * val model = QwenNetworkLoader.fromGguf(randomAccessProvider = { rasSource })
 *     .load<FP32, Float>(ctx)
 *
 * // From GGUF (sequential, models <2GB)
 * val model = QwenNetworkLoader.fromGguf(sourceProvider = { fileSource })
 *     .load<FP32, Float>(ctx)
 *
 * // From pre-loaded weights
 * val model = QwenNetworkLoader.fromWeights(llamaWeights)
 * ```
 */
@OptIn(ExperimentalMemoryApi::class)
public class QwenNetworkLoader @PublishedApi internal constructor(
    @PublishedApi internal val weightsProvider: WeightsProvider,
    @PublishedApi internal val debug: Boolean = false
) {
    /** See [LlamaNetworkLoader.dtypePolicy]. */
    public var dtypePolicy: DTypePolicy = DTypePolicy.Any
        private set

    /** See [LlamaNetworkLoader.withDtypePolicy]. */
    public fun withDtypePolicy(policy: DTypePolicy): QwenNetworkLoader {
        DTypePolicyValidation.validate(
            policy, "QwenNetworkLoader.withDtypePolicy", keepNative = DECODER_NARROW_KEEP_NATIVE,
        )
        this.dtypePolicy = policy
        return this
    }

    @PublishedApi
    internal sealed interface WeightsProvider {
        data class GgufSource(
            val sourceProvider: () -> Source
        ) : WeightsProvider

        data class GgufRandomAccess(
            val randomAccessProvider: () -> RandomAccessSource,
            val weightForm: WeightForm?
        ) : WeightsProvider

        data class SafeTensors(
            val randomAccessProvider: () -> RandomAccessSource,
            val metadata: GgufDecoderMetadata,
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
            debug: Boolean = false
        ): QwenNetworkLoader = QwenNetworkLoader(
            WeightsProvider.GgufSource(sourceProvider), debug
        )

        /** Load from a GGUF file via streaming RandomAccessSource (any size). */
        @JvmName("fromGgufRandomAccess")
        public fun fromGguf(
            randomAccessProvider: () -> RandomAccessSource,
            debug: Boolean = false,
            weightForm: WeightForm? = null
        ): QwenNetworkLoader = QwenNetworkLoader(
            WeightsProvider.GgufRandomAccess(randomAccessProvider, weightForm), debug
        )

        /** Load from a SafeTensors file. Requires metadata (not embedded in SafeTensors). */
        public fun fromSafeTensors(
            metadata: GgufDecoderMetadata,
            randomAccessProvider: () -> RandomAccessSource,
            tiedEmbeddings: Boolean = false,
            debug: Boolean = false
        ): QwenNetworkLoader = QwenNetworkLoader(
            WeightsProvider.SafeTensors(randomAccessProvider, metadata, tiedEmbeddings), debug
        )

        /** Build from already-loaded [DecoderGgufWeights] (GGUF-canonical tensor names). */
        public inline fun <reified T : DType, V> fromWeights(
            weights: DecoderGgufWeights<T, V>,
            debug: Boolean = false
        ): Module<T, V> = QwenNetworkLoader(
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
            is WeightsProvider.GgufSource -> QwenWeightLoader.loadToMap<T, V>(
                ctx, wp.sourceProvider, dtypePolicy,
            )
            is WeightsProvider.GgufRandomAccess -> QwenWeightLoader.loadToMapStreaming<T, V>(
                ctx, wp.randomAccessProvider, wp.weightForm, dtypePolicy,
            )
            is WeightsProvider.SafeTensors -> {
                val loader = DecoderSafeTensorsLoader<T>(ctx, T::class, wp.metadata, wp.tiedEmbeddings, dtypePolicy)
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
     *
     * QK-Norm is enabled iff any `*.attn_q_norm.weight` tensor is present in
     * the loaded weights. Real Qwen3 GGUFs always carry these (it's a
     * defining feature of Qwen3). Synthetic test fixtures that omit them
     * get a Llama-shape network so weight mapping doesn't reject unmapped
     * `q_norm`/`k_norm` parameters.
     */
    @PublishedApi
    internal inline fun <reified T : DType, V> applyWeightsToNetwork(
        weights: DecoderGgufWeights<T, V>
    ): Module<T, V> {
        val hasQkNorm = weights.tensors.keys.any { it.endsWith(".attn_q_norm.weight") }
        // Qwen2/Qwen2.5 ship attention projection biases; Qwen3 does not. Structural
        // presence in the file decides, like qkNorm — without this the loaded bias
        // tensors never bind and qwen2 logits are silently garbage (#338 arc find).
        val hasAttnBias = weights.tensors.keys.any { it.endsWith(".attn_q.bias") }
        val model = qwenNetwork<T, V>(weights.metadata, qkNorm = hasQkNorm, attnBias = hasAttnBias)

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
            nameResolver = QwenGGUFNameResolver()
        )

        val result = WeightMapper.applyWeights(model, weightTensors, config)

        // Qwen3 has no bias tensors — the DSL's zero-initialized bias params are expected to be
        // unmapped THERE. But a bias tensor present in the file that failed to bind is the
        // silent-garbage failure #352 diagnosed: fail loudly on it.
        val unboundFileBiases = result.unusedTensors.filter { it.endsWith(".bias") }
        require(unboundFileBiases.isEmpty()) {
            "Bias tensors present in the GGUF but never bound (zero-initialized params would " +
                "silently stand in): $unboundFileBiases"
        }
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
