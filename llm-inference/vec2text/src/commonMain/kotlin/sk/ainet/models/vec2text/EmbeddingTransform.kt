package sk.ainet.models.vec2text

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.gelu
import sk.ainet.lang.tensor.matmul
import sk.ainet.lang.tensor.plus
import sk.ainet.lang.tensor.reshape
import sk.ainet.lang.tensor.t
import sk.ainet.lang.types.DType

/**
 * vec2text's `embedding_transform` MLP: `Linear(d, d) → GELU → Linear(d, d·R)`, reshaped to
 * `R` pseudo-token vectors `[R, d]` fed to the T5 encoder as `inputs_embeds`. (Dropout is
 * inference-time identity and omitted.) The two linears are stored under the PyTorch
 * `Sequential` indices `.0` (first Linear) and `.3` (second Linear).
 *
 * Weights `[out, in]`; applied as `x @ Wᵀ + b`.
 */
public class EmbeddingTransform<T : DType>(
    private val w0: Tensor<T, Float>,
    private val b0: Tensor<T, Float>,
    private val w3: Tensor<T, Float>,
    private val b3: Tensor<T, Float>,
    private val numRepeatTokens: Int,
    private val dModel: Int,
) {
    /** Project a single `[dEmbedder]` embedding to `[numRepeatTokens, dModel]` pseudo-tokens. */
    public fun project(embedding: Tensor<T, Float>): Tensor<T, Float> {
        val x = embedding.reshape(Shape(1, embedding.shape[0])) // [1, dEmbedder]
        val h = (x.matmul(w0.t()) + b0).gelu()                  // [1, d]
        val out = h.matmul(w3.t()) + b3                         // [1, d·R]
        return out.reshape(Shape(numRepeatTokens, dModel))      // [R, d]
    }
}
