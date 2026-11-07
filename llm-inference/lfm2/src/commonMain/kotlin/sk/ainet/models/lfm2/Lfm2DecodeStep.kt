package sk.ainet.models.lfm2

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

/**
 * One fixed-capacity single-token step from an already embedded token: the decoder of
 * [Lfm2CachedTensorNetwork] with the embedding lookup left to the host and the output projection
 * over [head], a `[vocabulary, width]` weight of its own. The head may be a differently encoded
 * copy of the tied embedding (e.g. Q4_0), so a block-quant fusion can fold it: the tied tensor
 * itself also serves the lookup and would have two consumers.
 *
 * Inputs and state follow [Lfm2CachedTensorNetwork.forward]: [mask] `[1, capacity + 1]`,
 * [writeMask] `[1, capacity, 1]`, `history_<l>` `[width, kernel]`, `key_<l>`/`value_<l>`
 * `[kvHeads, capacity, headDim]`. [convLowering] defaults to [ShortConvLowering.MULTIPLY_SUM]:
 * the step's convolution is then elementwise arithmetic, with the same result. [attentionLayout] defaults to
 * one batched product over all heads ([AttentionLayoutPolicy.CURRENT_BATCHED]: 12 attention matmuls instead
 * of 192 for LFM2.5-350M, 22 % faster on an SL2610 NPU) rather than the cached network's per-head form.
 */
class Lfm2DecodeStep(
    val dimensions: ModelDimensions,
    weights: Map<String, Tensor<FP32, Float>>,
    val capacity: Int,
    private val head: Tensor<FP32, Float>,
    val convLowering: ShortConvLowering = ShortConvLowering.MULTIPLY_SUM,
    val attentionLayout: AttentionLayoutPolicy = AttentionLayoutPolicy.CURRENT_BATCHED,
) {
    private val layers = Lfm2CachedLayers(dimensions, weights, capacity, convLowering, attentionLayout)

    init {
        require(head.shape == Shape(dimensions.vocabulary, dimensions.width)) { "Output head must be [vocabulary, width]" }
    }

    /** State names and shapes, identical at the step's input and output. */
    fun stateShapes(): LinkedHashMap<String, Shape> = layers.stateShapes()

    /** Logits `[1, vocabulary]` and every next state, for the embedding row [embedded] `[1, width]`. */
    fun forward(
        embedded: Tensor<FP32, Float>,
        cos: Tensor<FP32, Float>,
        sin: Tensor<FP32, Float>,
        mask: Tensor<FP32, Float>,
        writeMask: Tensor<FP32, Float>,
        states: Map<String, Tensor<FP32, Float>>,
        ctx: ExecutionContext,
    ): Lfm2CachedTensorResult {
        val (normalized, nextStates) = layers.forward(embedded, cos, sin, mask, writeMask, states, ctx)
        return Lfm2CachedTensorResult(ctx.ops.matmulWeightTransposed(normalized, head), nextStates)
    }
}
