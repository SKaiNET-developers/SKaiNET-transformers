package sk.ainet.models.vec2text

import sk.ainet.context.ExecutionContext
import sk.ainet.io.ParametersLoader
import sk.ainet.lang.tensor.Tensor
import sk.ainet.models.t5.T5Config
import sk.ainet.models.t5.T5WeightMapper
import sk.ainet.models.t5.T5Weights
import sk.ainet.models.t5.loadTensorMap
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/** The inversion ("hypothesizer") model's weights: one embedding-transform MLP + a T5 seq2seq. */
public class InversionWeights<T : DType>(
    public val transform: EmbeddingTransform<T>,
    public val t5: T5Weights<T>,
    public val config: T5Config,
    public val numRepeatTokens: Int,
)

/** The corrector model's weights: three embedding-transform MLPs, a LayerNorm, and a T5 seq2seq. */
public class CorrectorWeights<T : DType>(
    public val transform1: EmbeddingTransform<T>,
    public val transform2: EmbeddingTransform<T>,
    public val transform3: EmbeddingTransform<T>,
    public val layerNormWeight: Tensor<T, Float>,
    public val layerNormBias: Tensor<T, Float>,
    public val t5: T5Weights<T>,
    public val config: T5Config,
    public val numRepeatTokens: Int,
)

/** Shared loading helpers for the vec2text checkpoints (T5 under the `encoder_decoder.` scope). */
public object Vec2TextWeightLoader {

    private const val PREFIX = "encoder_decoder."

    private fun <T : DType> transform(
        tensors: Map<String, Tensor<T, Float>>,
        name: String,
        numRepeatTokens: Int,
        dModel: Int,
    ): EmbeddingTransform<T> = EmbeddingTransform(
        w0 = tensors.getValue("$name.0.weight"),
        b0 = tensors.getValue("$name.0.bias"),
        w3 = tensors.getValue("$name.3.weight"),
        b3 = tensors.getValue("$name.3.bias"),
        numRepeatTokens = numRepeatTokens,
        dModel = dModel,
    )

    public suspend fun <T : DType> loadInversion(
        loader: ParametersLoader,
        ctx: ExecutionContext,
        dtype: KClass<T>,
        config: T5Config = T5Config(),
        numRepeatTokens: Int = 16,
    ): InversionWeights<T> {
        val tensors = loadTensorMap(loader, ctx, dtype)
        val t5 = T5WeightMapper.map(tensors, config, PREFIX, withDecoder = true)
        return InversionWeights(
            transform = transform(tensors, "embedding_transform", numRepeatTokens, config.dModel),
            t5 = t5,
            config = config,
            numRepeatTokens = numRepeatTokens,
        )
    }

    public suspend fun <T : DType> loadCorrector(
        loader: ParametersLoader,
        ctx: ExecutionContext,
        dtype: KClass<T>,
        config: T5Config = T5Config(),
        numRepeatTokens: Int = 16,
    ): CorrectorWeights<T> {
        val tensors = loadTensorMap(loader, ctx, dtype)
        val t5 = T5WeightMapper.map(tensors, config, PREFIX, withDecoder = true)
        return CorrectorWeights(
            transform1 = transform(tensors, "embedding_transform_1", numRepeatTokens, config.dModel),
            transform2 = transform(tensors, "embedding_transform_2", numRepeatTokens, config.dModel),
            transform3 = transform(tensors, "embedding_transform_3", numRepeatTokens, config.dModel),
            layerNormWeight = tensors.getValue("layernorm.weight"),
            layerNormBias = tensors.getValue("layernorm.bias"),
            t5 = t5,
            config = config,
            numRepeatTokens = numRepeatTokens,
        )
    }
}
