package sk.ainet.models.t5

import sk.ainet.context.ExecutionContext
import sk.ainet.io.ParametersLoader
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/** One T5 self- or cross-attention block's projection weights (all `[·, ·]`, no bias). */
public class T5AttentionWeights<T : DType>(
    public val q: Tensor<T, Float>,
    public val k: Tensor<T, Float>,
    public val v: Tensor<T, Float>,
    public val o: Tensor<T, Float>,
)

/** One T5 feed-forward block: `wo(relu(wi(x)))` (t5-base, un-gated). */
public class T5FeedForwardWeights<T : DType>(
    public val wi: Tensor<T, Float>,
    public val wo: Tensor<T, Float>,
)

/** One encoder layer: self-attention + FFN, each preceded by an RMS (T5) layer norm. */
public class T5EncoderLayerWeights<T : DType>(
    public val selfAttn: T5AttentionWeights<T>,
    public val selfAttnNorm: Tensor<T, Float>,
    public val ff: T5FeedForwardWeights<T>,
    public val ffNorm: Tensor<T, Float>,
)

/** One decoder layer: self-attn + cross-attn + FFN, each preceded by an RMS layer norm. */
public class T5DecoderLayerWeights<T : DType>(
    public val selfAttn: T5AttentionWeights<T>,
    public val selfAttnNorm: Tensor<T, Float>,
    public val crossAttn: T5AttentionWeights<T>,
    public val crossAttnNorm: Tensor<T, Float>,
    public val ff: T5FeedForwardWeights<T>,
    public val ffNorm: Tensor<T, Float>,
)

/**
 * Fully-parsed T5 weights. [decoderLayers] and [decoderFinalNorm] are null when only the
 * encoder was loaded (the GTR embedder case). [relativeAttentionBias] tensors are the
 * block-0 tables shared across the whole stack.
 */
public class T5Weights<T : DType>(
    public val config: T5Config,
    public val shared: Tensor<T, Float>,
    public val encoderRelativeBias: Tensor<T, Float>,
    public val encoderLayers: List<T5EncoderLayerWeights<T>>,
    public val encoderFinalNorm: Tensor<T, Float>,
    public val decoderRelativeBias: Tensor<T, Float>? = null,
    public val decoderLayers: List<T5DecoderLayerWeights<T>>? = null,
    public val decoderFinalNorm: Tensor<T, Float>? = null,
)

/**
 * Maps a flat `name -> tensor` map (as produced by any [ParametersLoader]) to typed
 * [T5Weights]. [prefix] is prepended to every key: `""` for a bare T5 checkpoint
 * (gtr_encoder), `"encoder_decoder."` for the vec2text inversion/corrector checkpoints
 * whose T5 lives under that scope.
 */
public object T5WeightMapper {

    public fun <T : DType> map(
        tensors: Map<String, Tensor<T, Float>>,
        config: T5Config,
        prefix: String = "",
        withDecoder: Boolean,
    ): T5Weights<T> {
        fun get(name: String): Tensor<T, Float> =
            tensors["$prefix$name"] ?: error("Missing T5 tensor: $prefix$name")

        fun attn(stack: String, block: Int, sub: Int, module: String): T5AttentionWeights<T> =
            T5AttentionWeights(
                q = get("$stack.block.$block.layer.$sub.$module.q.weight"),
                k = get("$stack.block.$block.layer.$sub.$module.k.weight"),
                v = get("$stack.block.$block.layer.$sub.$module.v.weight"),
                o = get("$stack.block.$block.layer.$sub.$module.o.weight"),
            )

        fun ff(stack: String, block: Int, sub: Int): T5FeedForwardWeights<T> =
            T5FeedForwardWeights(
                wi = get("$stack.block.$block.layer.$sub.DenseReluDense.wi.weight"),
                wo = get("$stack.block.$block.layer.$sub.DenseReluDense.wo.weight"),
            )

        val shared = get("shared.weight")

        val encoderLayers = (0 until config.numLayers).map { i ->
            T5EncoderLayerWeights(
                selfAttn = attn("encoder", i, 0, "SelfAttention"),
                selfAttnNorm = get("encoder.block.$i.layer.0.layer_norm.weight"),
                ff = ff("encoder", i, 1),
                ffNorm = get("encoder.block.$i.layer.1.layer_norm.weight"),
            )
        }
        val encoderRelativeBias = get("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")
        val encoderFinalNorm = get("encoder.final_layer_norm.weight")

        if (!withDecoder) {
            return T5Weights(config, shared, encoderRelativeBias, encoderLayers, encoderFinalNorm)
        }

        val decoderLayers = (0 until config.numDecoderLayers).map { i ->
            T5DecoderLayerWeights(
                selfAttn = attn("decoder", i, 0, "SelfAttention"),
                selfAttnNorm = get("decoder.block.$i.layer.0.layer_norm.weight"),
                crossAttn = attn("decoder", i, 1, "EncDecAttention"),
                crossAttnNorm = get("decoder.block.$i.layer.1.layer_norm.weight"),
                ff = ff("decoder", i, 2),
                ffNorm = get("decoder.block.$i.layer.2.layer_norm.weight"),
            )
        }
        val decoderRelativeBias = get("decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")
        val decoderFinalNorm = get("decoder.final_layer_norm.weight")

        return T5Weights(
            config, shared, encoderRelativeBias, encoderLayers, encoderFinalNorm,
            decoderRelativeBias, decoderLayers, decoderFinalNorm,
        )
    }
}

/** Load a flat tensor map from a [ParametersLoader] (SafeTensors / GGUF / ONNX / …). */
public suspend fun <T : DType> loadTensorMap(
    loader: ParametersLoader,
    ctx: ExecutionContext,
    dtype: KClass<T>,
): Map<String, Tensor<T, Float>> {
    val tensors = mutableMapOf<String, Tensor<T, Float>>()
    loader.load<T, Float>(ctx, dtype) { name, tensor -> tensors[name] = tensor }
    return tensors
}

/** Convenience: load + map T5 weights from a single [ParametersLoader]. */
public suspend fun <T : DType> loadT5Weights(
    loader: ParametersLoader,
    ctx: ExecutionContext,
    dtype: KClass<T>,
    config: T5Config,
    prefix: String = "",
    withDecoder: Boolean,
): T5Weights<T> = T5WeightMapper.map(loadTensorMap(loader, ctx, dtype), config, prefix, withDecoder)
