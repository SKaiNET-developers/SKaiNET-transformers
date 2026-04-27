package sk.ainet.apps.llm

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Minimal inference runtime interface for the agent layer.
 * Any model runtime (LLaMA, Gemma, Mistral, etc.) that wants agent capabilities
 * implements this interface.
 */
public interface InferenceRuntime<T : DType> {
    /** Reset the runtime state (clear KV cache, rewind position). */
    public fun reset()

    /** Forward one token and return logits. */
    public fun forward(tokenId: Int): Tensor<T, Float>

    /**
     * Batched forward over [tokenIds], advancing the runtime's internal
     * position by `tokenIds.size` and returning the logits **at the
     * last position only** (shape `[vocab]`, same as [forward]).
     *
     * Equivalent in semantics to calling [forward] once per token and
     * keeping only the final logits — but typically much faster for
     * prefill, because the forward graph is constructed/dispatched once
     * over an `[N]`-shaped input instead of N times. Real-world prefill
     * speedups on Gemma 4 with ~250-token chat-template prompts are
     * around 5–10×.
     *
     * Default implementation loops [forward] for runtimes that haven't
     * specialized this — they keep working at single-token speed
     * without code changes.
     */
    public fun forwardBatched(tokenIds: IntArray): Tensor<T, Float> {
        require(tokenIds.isNotEmpty()) { "forwardBatched: tokenIds must not be empty" }
        var last: Tensor<T, Float> = forward(tokenIds[0])
        for (i in 1 until tokenIds.size) {
            last = forward(tokenIds[i])
        }
        return last
    }
}
