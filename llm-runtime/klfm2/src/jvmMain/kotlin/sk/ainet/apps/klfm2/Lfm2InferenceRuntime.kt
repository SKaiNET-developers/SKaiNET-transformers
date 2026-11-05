package sk.ainet.apps.klfm2

import sk.ainet.models.lfm2.HybridLfm2Model
import sk.ainet.models.lfm2.Lfm2Session
import sk.ainet.apps.llm.InferenceRuntime
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

/**
 * SKaiNET-transformers runtime backed by one fresh LFM2 hybrid-state session.
 *
 * Calls are synchronous and thread-confined. [forwardBatched] consumes the whole
 * input and returns only the last position's logits, with shape `[vocabulary]`.
 * [reset] clears this runtime's convolution/KV state without modifying the model
 * or another runtime. Invalid inputs are rejected before advancing state.
 *
 * The owner must [close] this runtime when finished; close is idempotent, clears
 * state, and permanently rejects later forward/reset calls. It does not dispose
 * shared model weights. The upstream interface itself has no close contract.
 *
 * This token-level adapter does not select a tokenizer, template, sampler, EOS
 * policy or generation loop. The qualified text facade retains those choices.
 */
class Lfm2InferenceRuntime(model: HybridLfm2Model) : InferenceRuntime<FP32>, AutoCloseable {
    private val session = Lfm2Session(model)

    /** Number of tokens forwarded into this runtime's current state. */
    val position: Int get() = session.position

    override fun forward(tokenId: Int): Tensor<FP32, Float> = session.forward(tokenId)

    override fun forwardBatched(tokenIds: IntArray): Tensor<FP32, Float> = session.forwardBatched(tokenIds)

    override fun reset() = session.reset()

    override fun close() = session.close()
}
