package sk.ainet.models.vec2text

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.models.t5.T5Runtime
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * vec2text inversion model / "hypothesizer" (`jxm/gtr__nq__32`): projects the target
 * embedding to `numRepeatTokens` pseudo-tokens, feeds them to a T5 encoder as `inputs_embeds`,
 * and greedily decodes the first text hypothesis.
 */
public class InversionModel<T : DType>(
    ctx: ExecutionContext,
    private val weights: InversionWeights<T>,
    dtype: KClass<T>,
) {
    private val t5 = T5Runtime(ctx, weights.t5, dtype)

    /** Produce the initial hypothesis token ids (T5 tokenizer space) from a `[dEmbedder]` target. */
    public fun invert(targetEmbedding: Tensor<T, Float>, maxLength: Int = 128): IntArray {
        val pseudoTokens = weights.transform.project(targetEmbedding) // [R, dModel]
        val memory = t5.encode(pseudoTokens)
        return t5.generate(memory, maxLength)
    }
}
