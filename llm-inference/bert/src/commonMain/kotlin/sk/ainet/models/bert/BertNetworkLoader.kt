package sk.ainet.models.bert

import sk.ainet.context.ExecutionContext
import sk.ainet.io.ParametersLoader
import sk.ainet.io.weights.BertSafeTensorsNameResolver
import sk.ainet.io.weights.MappingConfig
import sk.ainet.io.weights.WeightMapper
import sk.ainet.io.weights.WeightTensor
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * End-to-end loader that builds a `bertNetwork()` module and populates it
 * with weights from SafeTensors (HuggingFace) format via [WeightMapper] + [BertSafeTensorsNameResolver].
 *
 * Usage:
 * ```kotlin
 * val model = BertNetworkLoader.fromTensorMap(config, tensors)
 *
 * // Or from existing BertRuntimeWeights
 * val model = BertNetworkLoader.fromRuntimeWeights(config, runtimeWeights)
 * ```
 *
 * Note: BERT's position and token_type embeddings are not part of the DSL network
 * (they require additive lookups during forward pass). Only word embeddings,
 * LayerNorm, attention, and FFN weights are mapped.
 */
public object BertNetworkLoader {

    /**
     * Build a BERT network from a flat tensor map with HuggingFace tensor names.
     *
     * @param config BERT model configuration
     * @param tensors Map of HF tensor names to tensors (e.g. from SafeTensors/ONNX loading)
     * @param debug Whether to print debug mapping information
     */
    public inline fun <reified T : DType, V> fromTensorMap(
        config: BertModelConfig,
        tensors: Map<String, Tensor<T, V>>,
        debug: Boolean = false
    ): Module<T, V> {
        val model = bertNetwork<T, V>(config)

        val weightTensors = tensors.map { (name, tensor) ->
            WeightTensor(
                name = name,
                shape = tensor.shape.dimensions.toList(),
                tensor = tensor
            )
        }

        val mappingConfig = MappingConfig(
            usePathBasedMatching = false,
            fallbackToShapeMatching = false,
            debug = debug,
            nameResolver = BertSafeTensorsNameResolver()
        )

        val result = WeightMapper.applyWeights(model, weightTensors, mappingConfig)

        // BERT tensors include position_embeddings, token_type_embeddings, pooler, etc.
        // that are not in the DSL model — these will appear as unused tensors, which is fine.
        // But all DSL parameters must be mapped.
        require(result.mapped == result.total) {
            buildString {
                appendLine("Failed to map ${result.total - result.mapped}/${result.total} BERT parameters:")
                result.missingParams.forEach { appendLine("  - $it") }
                if (result.unusedTensors.isNotEmpty()) {
                    appendLine("Unused tensors (${result.unusedTensors.size}):")
                    result.unusedTensors.take(10).forEach { appendLine("  - $it") }
                }
            }.trim()
        }

        return model
    }

    /**
     * Build a BERT network from existing [BertRuntimeWeights].
     *
     * Converts the structured weights back to a flat HF-named tensor map
     * and feeds it through the standard mapping pipeline.
     */
    public inline fun <reified T : DType, V> fromRuntimeWeights(
        weights: BertRuntimeWeights<T>,
        debug: Boolean = false
    ): Module<T, V> {
        val tensors = mutableMapOf<String, Tensor<T, V>>()

        @Suppress("UNCHECKED_CAST")
        fun put(name: String, t: Tensor<T, *>) { tensors[name] = t as Tensor<T, V> }

        // Embeddings
        put(BertTensorNames.WORD_EMBEDDINGS, weights.wordEmbeddings)
        put(BertTensorNames.EMBEDDING_LN_WEIGHT, weights.embeddingLayerNormWeight)
        put(BertTensorNames.EMBEDDING_LN_BIAS, weights.embeddingLayerNormBias)

        // Per-layer weights
        weights.layers.forEachIndexed { i, layer ->
            put(BertTensorNames.queryWeight(i), layer.queryWeight)
            put(BertTensorNames.queryBias(i), layer.queryBias)
            put(BertTensorNames.keyWeight(i), layer.keyWeight)
            put(BertTensorNames.keyBias(i), layer.keyBias)
            put(BertTensorNames.valueWeight(i), layer.valueWeight)
            put(BertTensorNames.valueBias(i), layer.valueBias)
            put(BertTensorNames.attnOutputWeight(i), layer.attnOutputWeight)
            put(BertTensorNames.attnOutputBias(i), layer.attnOutputBias)
            put(BertTensorNames.attnLayerNormWeight(i), layer.attnLayerNormWeight)
            put(BertTensorNames.attnLayerNormBias(i), layer.attnLayerNormBias)
            put(BertTensorNames.intermediateWeight(i), layer.intermediateWeight)
            put(BertTensorNames.intermediateBias(i), layer.intermediateBias)
            put(BertTensorNames.outputWeight(i), layer.outputWeight)
            put(BertTensorNames.outputBias(i), layer.outputBias)
            put(BertTensorNames.outputLayerNormWeight(i), layer.outputLayerNormWeight)
            put(BertTensorNames.outputLayerNormBias(i), layer.outputLayerNormBias)
        }

        return fromTensorMap(weights.config, tensors, debug)
    }
}
