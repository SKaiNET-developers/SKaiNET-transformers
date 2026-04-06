package sk.ainet.models.voxtral

import kotlinx.io.Source
import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.io.weights.MappingConfig
import sk.ainet.io.weights.WeightMapper
import sk.ainet.io.weights.WeightTensor
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.models.llama.LlamaModelMetadata
import sk.ainet.models.llama.LlamaSafeTensorsLoader
import sk.ainet.models.llama.LlamaWeightLoader
import sk.ainet.models.llama.LlamaWeights
import kotlin.jvm.JvmName
import kotlin.reflect.KClass

/**
 * End-to-end loader that builds Voxtral network modules and populates them
 * with weights from GGUF or SafeTensors files.
 *
 * Voxtral's text backbone uses the same LLaMA architecture and GGUF tensor naming
 * as Mistral/LLaMA, so weight loading delegates to [LlamaWeightLoader] for GGUF
 * and [LlamaSafeTensorsLoader] for SafeTensors. Tensor names are mapped via
 * [VoxtralHfTensorNameMapper] for SafeTensors and [VoxtralGGUFNameResolver] for
 * DSL weight mapping.
 *
 * Usage:
 * ```kotlin
 * // Load backbone from GGUF (streaming)
 * val backbone = VoxtralNetworkLoader.fromGguf(randomAccessProvider = { rasSource })
 *     .loadBackbone<FP32, Float>(ctx)
 *
 * // Load backbone from SafeTensors
 * val backbone = VoxtralNetworkLoader.fromSafeTensors(
 *     metadata = VoxtralDefaults.BACKBONE,
 *     randomAccessProvider = { rasSource },
 *     tiedEmbeddings = true
 * ).loadBackbone<FP32, Float>(ctx)
 *
 * // Load from pre-loaded weights
 * val backbone = VoxtralNetworkLoader.backboneFromWeights(llamaWeights)
 * ```
 */
public class VoxtralNetworkLoader @PublishedApi internal constructor(
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
            val weights: LlamaWeights<T, V>
        ) : WeightsProvider
    }

    public companion object {
        /** Load from a GGUF file via sequential Source (models under 2GB). */
        public fun fromGguf(
            sourceProvider: () -> Source,
            quantPolicy: QuantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
            debug: Boolean = false
        ): VoxtralNetworkLoader = VoxtralNetworkLoader(
            WeightsProvider.GgufSource(sourceProvider, quantPolicy), debug
        )

        /** Load from a GGUF file via streaming RandomAccessSource (any size). */
        @JvmName("fromGgufRandomAccess")
        public fun fromGguf(
            randomAccessProvider: () -> RandomAccessSource,
            quantPolicy: QuantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
            debug: Boolean = false
        ): VoxtralNetworkLoader = VoxtralNetworkLoader(
            WeightsProvider.GgufRandomAccess(randomAccessProvider, quantPolicy), debug
        )

        /** Load from a SafeTensors file. Requires metadata (not embedded in SafeTensors). */
        public fun fromSafeTensors(
            metadata: LlamaModelMetadata,
            randomAccessProvider: () -> RandomAccessSource,
            tiedEmbeddings: Boolean = true,
            debug: Boolean = false
        ): VoxtralNetworkLoader = VoxtralNetworkLoader(
            WeightsProvider.SafeTensors(randomAccessProvider, metadata, tiedEmbeddings), debug
        )

        /** Build backbone from already-loaded [LlamaWeights] (GGUF-canonical tensor names). */
        public inline fun <reified T : DType, V> backboneFromWeights(
            weights: LlamaWeights<T, V>,
            debug: Boolean = false
        ): Module<T, V> = VoxtralNetworkLoader(
            WeightsProvider.Preloaded(weights), debug
        ).applyWeightsToBackbone(weights)

        /** Build acoustic runtime from already-loaded weights. */
        public inline fun <reified T : DType> acousticFromWeights(
            weights: LlamaWeights<T, Float>,
            acousticMetadata: LlamaModelMetadata,
            ctx: ExecutionContext,
            nCodebooks: Int = 36,
            codebookLevels: Int = 21,
            debug: Boolean = false
        ): VoxtralAcousticRuntime<T> {
            val acousticModel = voxtralAcousticNetwork<T, Float>(acousticMetadata)
            return VoxtralNetworkLoader(
                WeightsProvider.Preloaded(weights), debug
            ).buildAcousticRuntime(weights, acousticModel, acousticMetadata, ctx, T::class, nCodebooks, codebookLevels)
        }
    }

    /**
     * Load weights and build the text backbone network.
     *
     * @return A [Module] with all backbone weights populated from the model file.
     * @throws IllegalArgumentException if required weights could not be mapped.
     */
    public suspend inline fun <reified T : DType, V> loadBackbone(
        ctx: ExecutionContext
    ): Module<T, V> {
        val weights: LlamaWeights<T, V> = loadWeights(ctx)
        return applyWeightsToBackbone(weights)
    }

    /**
     * Load weights from the configured source.
     */
    @PublishedApi
    internal suspend inline fun <reified T : DType, V> loadWeights(
        ctx: ExecutionContext
    ): LlamaWeights<T, V> {
        return when (val wp = weightsProvider) {
            is WeightsProvider.GgufSource -> {
                val loader = LlamaWeightLoader(wp.sourceProvider, quantPolicy = wp.quantPolicy)
                loader.loadToMap<T, V>(ctx)
            }
            is WeightsProvider.GgufRandomAccess -> {
                val loader = LlamaWeightLoader(wp.randomAccessProvider, quantPolicy = wp.quantPolicy)
                loader.loadToMapStreaming<T, V>(ctx)
            }
            is WeightsProvider.SafeTensors -> {
                val loader = LlamaSafeTensorsLoader<T>(ctx, T::class, wp.metadata, wp.tiedEmbeddings)
                @Suppress("UNCHECKED_CAST")
                loader.loadToMap(wp.randomAccessProvider) as LlamaWeights<T, V>
            }
            is WeightsProvider.Preloaded<*, *> -> {
                @Suppress("UNCHECKED_CAST")
                wp.weights as LlamaWeights<T, V>
            }
        }
    }

    /**
     * Build the backbone DSL network from metadata and map all weights.
     */
    @PublishedApi
    internal inline fun <reified T : DType, V> applyWeightsToBackbone(
        weights: LlamaWeights<T, V>
    ): Module<T, V> {
        val model = voxtralBackboneNetwork<T, V>(weights.metadata)

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
            nameResolver = VoxtralGGUFNameResolver()
        )

        val result = WeightMapper.applyWeights(model, weightTensors, config)

        // Voxtral uses tied embeddings and has no bias tensors. The DSL's dense()
        // creates Linear modules with zero-initialized bias params — expected to be unmapped.
        // Also filter out acoustic/codec tensors that belong to other components.
        val unmappedNonBias = result.missingParams.filter { !it.contains(".bias") }
        require(unmappedNonBias.isEmpty()) {
            buildString {
                appendLine("Failed to map ${unmappedNonBias.size} backbone weight parameters:")
                unmappedNonBias.forEach { appendLine("  - $it") }
                if (result.unusedTensors.isNotEmpty()) {
                    appendLine("Unused tensors (${result.unusedTensors.size}):")
                    result.unusedTensors.take(10).forEach { appendLine("  - $it") }
                }
            }.trim()
        }

        return model
    }

    /**
     * Load weights and build the acoustic flow-matching runtime.
     *
     * @param acousticMetadata Metadata for the acoustic transformer (3 layers).
     * @param nCodebooks Number of acoustic codebooks (default: 36).
     * @param codebookLevels FSQ levels per codebook (default: 21).
     * @return A [VoxtralAcousticRuntime] ready for flow-matching inference.
     */
    public suspend inline fun <reified T : DType> loadAcoustic(
        ctx: ExecutionContext,
        acousticMetadata: LlamaModelMetadata,
        nCodebooks: Int = 36,
        codebookLevels: Int = 21
    ): VoxtralAcousticRuntime<T> {
        val weights: LlamaWeights<T, Float> = loadWeights(ctx)
        val acousticModel = voxtralAcousticNetwork<T, Float>(acousticMetadata)
        return buildAcousticRuntime(weights, acousticModel, acousticMetadata, ctx, T::class, nCodebooks, codebookLevels)
    }

    /**
     * Build the acoustic runtime from loaded weights.
     *
     * 1. Build acoustic transformer DSL network and map weights
     * 2. Extract input/output projection tensors
     * 3. Assemble into [VoxtralAcousticRuntime]
     */
    @PublishedApi
    internal fun <T : DType> buildAcousticRuntime(
        weights: LlamaWeights<T, Float>,
        acousticModel: Module<T, Float>,
        acousticMetadata: LlamaModelMetadata,
        ctx: ExecutionContext,
        dtype: KClass<T>,
        nCodebooks: Int,
        codebookLevels: Int
    ): VoxtralAcousticRuntime<T> {
        val dim = acousticMetadata.embeddingLength
        val acousticDim = nCodebooks * codebookLevels

        // Map acoustic transformer weights
        val weightTensors = weights.tensors
            .filter { (name, _) -> name.startsWith("acoustic.blk.") || name.startsWith("acoustic.output_norm") }
            .map { (name, tensor) ->
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
            nameResolver = VoxtralGGUFNameResolver()
        )

        if (weightTensors.isNotEmpty()) {
            val result = WeightMapper.applyWeights(acousticModel, weightTensors, config)
            val unmappedNonBias = result.missingParams.filter { !it.contains(".bias") }
            if (unmappedNonBias.isNotEmpty() && debug) {
                println("Acoustic model unmapped params (${unmappedNonBias.size}):")
                unmappedNonBias.forEach { println("  - $it") }
            }
        }

        // Extract projection tensors (these are raw linear layers, not part of the DSL module)
        val inputProj = weights.tensors[VoxtralTensorNames.ACOUSTIC_INPUT_PROJ]
            ?: createZeroTensor(ctx, dtype, Shape(dim, acousticDim))
        val outputProj = weights.tensors[VoxtralTensorNames.ACOUSTIC_OUTPUT_PROJ]
            ?: createZeroTensor(ctx, dtype, Shape(acousticDim, dim))
        val inputProjBias = weights.tensors[VoxtralTensorNames.ACOUSTIC_INPUT_PROJ_BIAS]
        val outputProjBias = weights.tensors[VoxtralTensorNames.ACOUSTIC_OUTPUT_PROJ_BIAS]

        return VoxtralAcousticRuntime(
            acousticTransformer = acousticModel,
            inputProj = inputProj,
            outputProj = outputProj,
            inputProjBias = inputProjBias,
            outputProjBias = outputProjBias,
            ctx = ctx,
            dtype = dtype,
            nCodebooks = nCodebooks,
            codebookLevels = codebookLevels,
            dim = dim
        )
    }

    private fun <T : DType> createZeroTensor(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        shape: Shape
    ): Tensor<T, Float> {
        val data = FloatArray(shape.volume)
        @Suppress("UNCHECKED_CAST")
        val result = ctx.fromFloatArray<T, Float>(shape, dtype, data)
        return result as Tensor<T, Float>
    }
}
