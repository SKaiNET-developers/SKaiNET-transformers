package sk.ainet.models.bert

import sk.ainet.apps.llm.weights.BertSafeTensorsNameResolver
import sk.ainet.context.ExecutionContext
import sk.ainet.io.ParametersLoader
import sk.ainet.io.weights.MappingConfig
import sk.ainet.io.weights.WeightMapper
import sk.ainet.io.weights.WeightTensor
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * End-to-end loader that builds a `bertNetwork()` module and populates it
 * with weights from SafeTensors (HuggingFace) format via [WeightMapper] +
 * [BertSafeTensorsNameResolver].
 *
 * Usage:
 * ```kotlin
 * val tensors = BertNetworkLoader.loadWeightTensors(loaders, ctx, FP32::class)
 * val model = BertNetworkLoader.fromWeightTensors<FP32, Float>(config, tensors)
 * ```
 */
public object BertNetworkLoader {

    /** The sentence-transformers dense-projection tensor names (2_Dense/model.safetensors). */
    public const val PROJECTION_WEIGHT: String = "linear.weight"
    public const val PROJECTION_BIAS: String = "linear.bias"

    /**
     * Collect all tensors from [loaders] (base checkpoint plus the optional
     * `2_Dense/model.safetensors` projection file) into a flat [WeightTensor]
     * list with normalized HF names.
     *
     * Name normalization: plain BERT checkpoints prefix every tensor with
     * `bert.`; sentence-transformers checkpoints (e.g. MongoDB/mdbr-leaf) ship
     * bare names. When no key carries the prefix, every tensor except the
     * projection pair gains it, so [BertSafeTensorsNameResolver]'s
     * `bert.`-prefixed output matches either layout.
     */
    public suspend fun <T : DType> loadWeightTensors(
        loaders: List<ParametersLoader>,
        ctx: ExecutionContext,
        dtype: KClass<T>,
    ): List<WeightTensor<T, Float>> {
        val collected = mutableListOf<WeightTensor<T, Float>>()
        loaders.forEach { loader ->
            loader.load<T, Float>(ctx, dtype) { name, tensor ->
                collected += WeightTensor(
                    name = name,
                    shape = tensor.shape.dimensions.toList(),
                    tensor = tensor,
                )
            }
        }
        return normalizeTensorNames(collected)
    }

    /**
     * Build a BERT network and map [tensors] (as produced by
     * [loadWeightTensors]) into it. Position/token_type/pooler tensors that
     * the checkpoint carries beyond the DSL parameters are reported as unused,
     * which is expected; every DSL parameter must be mapped.
     */
    public inline fun <reified T : DType, V> fromWeightTensors(
        config: BertModelConfig,
        tensors: List<WeightTensor<T, V>>,
        debug: Boolean = false,
    ): Module<T, V> {
        val model = bertNetwork<T, V>(config)

        val mappingConfig = MappingConfig(
            // Path-based ONNX-style matching never applies to HF BERT names;
            // shape fallback stays off so same-shaped Q/K/V can't cross-wire.
            usePathBasedMatching = false,
            fallbackToShapeMatching = false,
            debug = debug,
            nameResolver = BertSafeTensorsNameResolver()
        )

        val result = WeightMapper.applyWeights(model, tensors, mappingConfig)

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
     * Build a BERT network from a flat tensor map with HuggingFace tensor names.
     */
    public inline fun <reified T : DType, V> fromTensorMap(
        config: BertModelConfig,
        tensors: Map<String, Tensor<T, V>>,
        debug: Boolean = false
    ): Module<T, V> {
        val weightTensors = normalizeTensorNames(
            tensors.map { (name, tensor) ->
                WeightTensor(
                    name = name,
                    shape = tensor.shape.dimensions.toList(),
                    tensor = tensor
                )
            }
        )
        return fromWeightTensors(config, weightTensors, debug)
    }

    /** See [loadWeightTensors] for the normalization contract. */
    public fun <T : DType, V> normalizeTensorNames(
        tensors: List<WeightTensor<T, V>>,
    ): List<WeightTensor<T, V>> {
        if (tensors.any { it.name.startsWith("bert.") }) return tensors
        return tensors.map { wt ->
            when (wt.name) {
                PROJECTION_WEIGHT, PROJECTION_BIAS -> wt
                else -> wt.copy(name = "bert.${wt.name}")
            }
        }
    }
}
