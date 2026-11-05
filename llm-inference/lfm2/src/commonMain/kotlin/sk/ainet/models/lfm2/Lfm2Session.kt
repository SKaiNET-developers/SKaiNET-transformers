package sk.ainet.models.lfm2

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

/**
 * A caller-owned, thread-confined token session. Each session has independent hybrid state.
 * Both forward methods return the last position's logits as [vocabulary], matching the
 * shape contract used by SKaiNET-transformers InferenceRuntime. This class does not yet
 * implement that upstream interface or render/tokenize messages.
 */
class Lfm2Session(private val model: HybridLfm2Model) : AutoCloseable {
    private val state = HybridModelState()
    private var closed = false
    val position: Int get() = state.position

    fun forward(tokenId: Int): Tensor<FP32, Float> = forwardBatched(intArrayOf(tokenId))

    fun forwardBatched(tokenIds: IntArray): Tensor<FP32, Float> {
        check(!closed) { "LFM2 session is closed" }
        val result = model.forward(tokenIds, state)
        return model.ctx.ops.reshape(
            model.ctx.ops.narrow(result.logits, dim = 0, start = tokenIds.size - 1, length = 1),
            Shape(model.dimensions.vocabulary),
        )
    }

    fun reset() {
        check(!closed) { "LFM2 session is closed" }
        state.reset()
    }

    override fun close() {
        state.reset()
        closed = true
    }
}
