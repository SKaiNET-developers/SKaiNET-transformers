package sk.ainet.models.vec2text

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.normalization.LayerNormalization
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.minus
import sk.ainet.models.t5.T5Runtime
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * vec2text corrector (`jxm/gtr__nq__32__correct`). Given the target embedding, the current
 * hypothesis embedding and the current hypothesis token ids, it builds the encoder input
 *
 *   `[sep, t1(target), sep, t3(hypEmb), sep, t2(target−hypEmb), sep, embed(hypIds)]`
 *
 * (sep = the EOS token embedded via the shared table), applies a LayerNorm over the whole
 * sequence, runs the T5 encoder, and greedily decodes a refined hypothesis. Layout, MLP
 * assignment and mask are verified against `vec2text/models/corrector_encoder.py`.
 */
public class CorrectorModel<T : DType>(
    private val ctx: ExecutionContext,
    private val weights: CorrectorWeights<T>,
    dtype: KClass<T>,
) {
    private val t5 = T5Runtime(ctx, weights.t5, dtype)
    private val eosId = weights.config.eosTokenId

    private val layerNorm = LayerNormalization<T, Float>(
        normalizedShape = intArrayOf(weights.config.dModel),
        eps = 1e-5,
        elementwiseAffine = true,
        name = "corrector.layernorm",
        initGamma = weights.layerNormWeight,
        initBeta = weights.layerNormBias,
    )

    /** Refine [hypothesisIds] into a new hypothesis (T5 token ids). */
    public fun correct(
        targetEmbedding: Tensor<T, Float>,
        hypothesisEmbedding: Tensor<T, Float>,
        hypothesisIds: IntArray,
        maxLength: Int = 128,
    ): IntArray {
        val diff = targetEmbedding - hypothesisEmbedding
        val pTarget = weights.transform1.project(targetEmbedding)      // [R, d]
        val pHyp = weights.transform3.project(hypothesisEmbedding)     // [R, d]
        val pDiff = weights.transform2.project(diff)                   // [R, d]
        val sep = t5.embed(intArrayOf(eosId))                          // [1, d]
        val hypTokens = t5.embed(hypothesisIds)                        // [hypLen, d]

        val inputsEmbeds = ctx.ops.concat(
            listOf(sep, pTarget, sep, pHyp, sep, pDiff, sep, hypTokens),
            dim = 0,
        )
        val normed = layerNorm.forward(inputsEmbeds, ctx)
        val memory = t5.encode(normed)
        return t5.generate(memory, maxLength)
    }
}
