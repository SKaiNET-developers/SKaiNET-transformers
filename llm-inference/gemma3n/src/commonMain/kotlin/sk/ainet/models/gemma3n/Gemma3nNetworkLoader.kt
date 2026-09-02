package sk.ainet.models.gemma3n

import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.weights.MappingConfig
import sk.ainet.io.weights.WeightMapper
import sk.ainet.io.weights.WeightTensor
import sk.ainet.lang.nn.Module
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * End-to-end loader for the Gemma 3n DSL path (#377): loads a real gemma3n GGUF through the
 * engine-delegated [Gemma3nWeightLoader] (packed/MAPPED by default; the PLE table stays
 * packed with row-dequant), builds [gemma3nNetwork] and binds every weight via
 * [WeightMapper] + [Gemma3nGGUFNameResolver].
 */
public object Gemma3nNetworkLoader {

    public suspend inline fun <reified T : DType, V> fromGguf(
        ctx: ExecutionContext,
        noinline randomAccessProvider: () -> RandomAccessSource,
        maxInferenceLen: Int? = null,
        debug: Boolean = false,
    ): Module<T, V> {
        val weights = Gemma3nWeightLoader(randomAccessProvider).loadToMapStreaming<T, V>(ctx)
        return fromWeights(ctx, weights, T::class, maxInferenceLen, debug)
    }

    public fun <T : DType, V> fromWeights(
        ctx: ExecutionContext,
        weights: Gemma3nWeights<T, V>,
        dtype: KClass<T>,
        maxInferenceLen: Int? = null,
        debug: Boolean = false,
    ): Module<T, V> {
        val md = weights.metadata
        // laurel_rank is not a GGUF field — read it off the checkpoint's own tensor.
        val laurelRank = weights.tensors["blk.0.laurel_l.weight"]?.shape?.get(0) ?: LAUREL_RANK

        val model = gemma3nNetwork<T, V>(
            md,
            dtype,
            maxInferenceLen = maxInferenceLen ?: minOf(md.contextLength, 4096),
            laurelRank = laurelRank,
        )

        val weightTensors = weights.tensors.map { (name, tensor) ->
            WeightTensor(name = name, shape = tensor.shape.dimensions.toList(), tensor = tensor)
        }
        val config = MappingConfig(
            usePathBasedMatching = false,
            fallbackToShapeMatching = false,
            debug = debug,
            nameResolver = Gemma3nGGUFNameResolver(),
        )
        val result = WeightMapper.applyWeights(model, weightTensors, config)

        // gemma3n has no bias tensors; every non-bias DSL param must bind, and no loaded
        // tensor may silently go unused (the qwen-bias lesson, transformers#352).
        val unmappedNonBias = result.missingParams.filter { !it.contains(".bias") }
        require(unmappedNonBias.isEmpty()) {
            buildString {
                appendLine("gemma3n: failed to map ${unmappedNonBias.size} weight parameters:")
                unmappedNonBias.take(20).forEach { appendLine("  - $it") }
                if (result.unusedTensors.isNotEmpty()) {
                    appendLine("Unused tensors (${result.unusedTensors.size}):")
                    result.unusedTensors.take(20).forEach { appendLine("  - $it") }
                }
            }.trim()
        }
        require(result.unusedTensors.isEmpty()) {
            "gemma3n: tensors present in the GGUF but never bound: ${result.unusedTensors.take(20)}"
        }
        return model
    }

    public inline fun <reified T : DType, V> fromWeights(
        ctx: ExecutionContext,
        weights: Gemma3nWeights<T, V>,
        maxInferenceLen: Int? = null,
        debug: Boolean = false,
    ): Module<T, V> = fromWeights(ctx, weights, T::class, maxInferenceLen, debug)
}
