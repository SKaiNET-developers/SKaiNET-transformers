package sk.ainet.models.apertus

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Strategy interface for Apertus attention computation.
 *
 * Similar to LLaMA's AttentionBackend but receives Q/K after QK-norm has been applied.
 * Applies RoPE encoding, KV cache management, and GQA attention scoring.
 *
 * Contract:
 * - Input: q [1, dim], k [1, kvDim], v [1, kvDim], layerIdx, position
 * - Output: attention output [1, dim]
 */
public interface ApertusAttentionBackend<T : DType> {

    /**
     * Compute attention for one token at the given position.
     *
     * Q and K have already been QK-normed by the caller.
     * This method applies RoPE, stores k/v in the KV cache,
     * and returns the attention-weighted output.
     */
    public fun attention(
        q: Tensor<T, Float>,
        k: Tensor<T, Float>,
        v: Tensor<T, Float>,
        layerIdx: Int,
        position: Int
    ): Tensor<T, Float>

    /**
     * Compute attention for a batch of tokens starting at [startPos].
     *
     * Returns null if the backend does not support batch attention,
     * in which case the runtime falls back to sequential processing.
     */
    public fun batchAttention(
        q: Tensor<T, Float>,
        k: Tensor<T, Float>,
        v: Tensor<T, Float>,
        layerIdx: Int,
        startPos: Int,
    ): Tensor<T, Float>? = null

    /**
     * Reset internal state (KV caches, position tracking, etc.).
     */
    public fun reset()
}
