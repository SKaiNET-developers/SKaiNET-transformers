package sk.ainet.models.bert

import sk.ainet.apps.llm.weights.BertSafeTensorsNameResolver
import sk.ainet.context.ExecutionContext
import sk.ainet.io.weights.MappingConfig
import sk.ainet.io.weights.WeightMapper
import sk.ainet.io.weights.WeightNameResolver
import sk.ainet.io.weights.WeightTensor
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.div
import sk.ainet.lang.tensor.matmul
import sk.ainet.lang.tensor.mean
import sk.ainet.lang.tensor.plus
import sk.ainet.lang.tensor.sqrt
import sk.ainet.lang.tensor.sum
import sk.ainet.lang.tensor.t
import sk.ainet.lang.tensor.times
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/** Execution strategy for [BertEncoderRuntime]. */
public enum class BertExecutionMode {
    /** Run the module tree eagerly — the primary JVM path. */
    DIRECT,

    /** Trace the encoder into an optimized ComputeGraph and execute the graph. */
    OPTIMIZED,
}

/**
 * Encoder runtime for BERT sentence embeddings on the DSL path.
 *
 * Wraps a [bertNetwork] module (a complete `tokens → hidden-states` encoder)
 * and adds what sentence embedding needs on top of the pure encoder graph:
 * masked mean pooling, the optional sentence-transformers dense projection
 * (`2_Dense`), and L2 normalization. Pooling and projection stay outside the
 * DSL network on purpose — the traced/exported graph remains a clean encoder,
 * and the pooling mask is dynamic per call.
 *
 * Intended use is one unpadded sequence per [encode] call; the attention mask
 * only affects pooling (attention itself is bidirectional over the full
 * sequence, exactly like the eager runtime this class replaces).
 *
 * Construction goes through [createBertEncoderRuntime], which maps checkpoint
 * tensors into the module via [WeightMapper].
 */
public class BertEncoderRuntime<T : DType>(
    private val model: Module<T, Float>,
    public val config: BertModelConfig,
    private val ctx: ExecutionContext,
    private val dtype: KClass<T>,
    private val projectionWeight: Tensor<T, Float>? = null,
    private val projectionBias: Tensor<T, Float>? = null,
    private val mode: BertExecutionMode = BertExecutionMode.DIRECT,
) {

    /** Output dimensionality of [encode]: projection out-features when present, else hidden size. */
    public val dimensions: Int get() = config.projectionDim ?: config.hiddenSize

    /**
     * Full-sequence encoder forward: `[L]` token ids → `[L, hiddenSize]`
     * hidden states.
     */
    public fun forward(tokenIds: IntArray): Tensor<T, Float> = ctx.scratch.scope {
        require(tokenIds.isNotEmpty()) { "BertEncoderRuntime: tokenIds must not be empty" }
        when (mode) {
            BertExecutionMode.DIRECT -> model.forward(tokenTensor(tokenIds), ctx)
            BertExecutionMode.OPTIMIZED -> forwardOptimized(tokenIds)
        }
    }

    /**
     * Encode tokens into a single embedding vector: forward → mean pooling
     * (mask-weighted when [attentionMask] is given) → optional dense
     * projection → L2 normalization.
     *
     * @param tokenIds token IDs including `[CLS]` and `[SEP]`
     * @param attentionMask 1 for real tokens, 0 for padding; affects pooling only
     * @return normalized vector of size [dimensions]
     */
    public fun encode(tokenIds: IntArray, attentionMask: IntArray? = null): FloatArray {
        val hiddenStates = forward(tokenIds)
        val seqLen = tokenIds.size

        var pooled = if (attentionMask != null) {
            require(attentionMask.size == seqLen) {
                "BertEncoderRuntime: attentionMask size ${attentionMask.size} != tokenIds size $seqLen"
            }
            val maskTensor = ctx.fromFloatArray<T, Float>(
                Shape(seqLen, 1), dtype,
                FloatArray(seqLen) { attentionMask[it].toFloat() }
            )
            val masked = hiddenStates * maskTensor
            val summed = masked.sum(dim = 0)
            val count = attentionMask.sumOf { it }.toFloat().coerceAtLeast(1f)
            summed / count
        } else {
            hiddenStates.mean(dim = 0)
        }

        if (projectionWeight != null) {
            // sentence-transformers Dense head; bias is optional (LEAF models
            // ship bias=false). The legacy eager runtime required both tensors
            // and silently skipped the projection on bias-free heads — fixed here.
            pooled = pooled.matmul(projectionWeight.t())
            if (projectionBias != null) pooled = pooled + projectionBias
        }

        pooled = l2Normalize(pooled)

        val out = FloatArray(pooled.volume)
        for (i in out.indices) out[i] = pooled.data[i]
        return out
    }

    private fun forwardOptimized(tokenIds: IntArray): Tensor<T, Float> {
        // Wired in the OPTIMIZED-mode change; DIRECT is the default path.
        throw UnsupportedOperationException(
            "BertExecutionMode.OPTIMIZED is not available yet — use DIRECT"
        )
    }

    private fun tokenTensor(tokenIds: IntArray): Tensor<T, Float> =
        ctx.fromFloatArray(
            Shape(tokenIds.size), dtype,
            FloatArray(tokenIds.size) { tokenIds[it].toFloat() },
        )

    private fun l2Normalize(tensor: Tensor<T, Float>): Tensor<T, Float> {
        val squared = tensor * tensor
        val sumSquared = squared.sum()
        val norm = (sumSquared + 1e-12).sqrt()
        return tensor / norm
    }
}

/**
 * Build a [BertEncoderRuntime] from checkpoint tensors: constructs
 * `bertNetwork(config)`, maps every DSL parameter by name via [resolver]
 * (strict — shape fallback stays off so same-shaped Q/K/V tensors can't
 * cross-wire), and pulls the optional `2_Dense` projection pair
 * (`linear.weight` / `linear.bias`) out of [tensors] for the runtime.
 *
 * @param tensors as produced by [BertNetworkLoader.loadWeightTensors]
 */
public inline fun <reified T : DType> createBertEncoderRuntime(
    config: BertModelConfig,
    tensors: List<WeightTensor<T, Float>>,
    ctx: ExecutionContext,
    resolver: WeightNameResolver = BertSafeTensorsNameResolver(),
    mode: BertExecutionMode = BertExecutionMode.DIRECT,
    debug: Boolean = false,
): BertEncoderRuntime<T> {
    val model = bertNetwork<T, Float>(config)

    val result = WeightMapper.applyWeights(
        model, tensors,
        MappingConfig(
            usePathBasedMatching = false,
            fallbackToShapeMatching = false,
            debug = debug,
            nameResolver = resolver,
        )
    )
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

    val projectionWeight = tensors.firstOrNull { it.name == BertNetworkLoader.PROJECTION_WEIGHT }?.tensor
    val projectionBias = tensors.firstOrNull { it.name == BertNetworkLoader.PROJECTION_BIAS }?.tensor
    if (config.projectionDim != null) {
        requireNotNull(projectionWeight) {
            "config.projectionDim=${config.projectionDim} but no ${BertNetworkLoader.PROJECTION_WEIGHT} tensor was provided"
        }
    }
    require(projectionBias == null || projectionWeight != null) {
        "${BertNetworkLoader.PROJECTION_BIAS} provided without ${BertNetworkLoader.PROJECTION_WEIGHT}"
    }

    return BertEncoderRuntime(
        model = model,
        config = config,
        ctx = ctx,
        dtype = T::class,
        projectionWeight = projectionWeight,
        projectionBias = projectionBias,
        mode = mode,
    )
}
