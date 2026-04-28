package sk.ainet.apps.llm

/**
 * Selects how an [InferenceRuntime] processes the prompt before sampling
 * begins.
 *
 * The prompt-processing phase ("prefill") and the sampling phase ("decode")
 * have very different cost structures. Decode is necessarily one token at a
 * time — each token depends on the previous sample. Prefill is teacher-
 * forced, so the entire prompt is known up front and can be ingested in one
 * batched forward pass; on a long prompt that's typically 3–10× faster than
 * looping `forward(t)` per token, because graph dispatch and per-layer
 * setup happen once instead of N times.
 *
 * Default is [Autoregressive] — preserves the long-standing one-token-per-
 * forward behavior so existing callers see no change. Opt in to [Batched]
 * to use [InferenceRuntime.forwardBatched] for the prompt phase.
 */
public sealed interface PrefillStrategy {
    /**
     * Process the prompt by calling [InferenceRuntime.forward] once per
     * token. Equivalent to the original `generate()` semantics.
     */
    public object Autoregressive : PrefillStrategy

    /**
     * Process the prompt in chunks of up to [maxBatch] tokens via
     * [InferenceRuntime.forwardBatched]. The last chunk's last-position
     * logits are used to sample the first generated token; subsequent
     * decode steps remain autoregressive.
     *
     * @param maxBatch maximum number of tokens fed to a single
     *   `forwardBatched` call. The whole prompt goes in one batched
     *   forward when its length is `<= maxBatch`. Tune lower if memory
     *   pressure on a long prompt becomes a problem; higher for maximum
     *   throughput on long prompts.
     */
    public data class Batched(val maxBatch: Int = 64) : PrefillStrategy {
        init {
            require(maxBatch > 0) { "PrefillStrategy.Batched: maxBatch must be > 0, got $maxBatch" }
        }
    }
}
