package sk.ainet.models.t5

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.mean
import sk.ainet.lang.types.DType

/**
 * GTR (`sentence-transformers/gtr-t5-base`) sentence embedder as used by vec2text: the
 * raw T5 **encoder** followed by mask-weighted mean pooling — **no** sentence-transformers
 * Dense projection and **no** L2 normalization (verified against
 * `vec2text/models/model_utils.py::load_embedder_and_tokenizer` +
 * `inversion.py::_process_embedder_output`).
 *
 * Batch size 1: callers pass the tokenized ids (with the trailing EOS `</s>`, truncated to
 * `config.maxSeqLength`) and no padding, so plain mean over the sequence equals vec2text's
 * masked mean pool.
 */
public class GtrEmbedder<T : DType>(
    private val t5: T5Runtime<T>,
) {
    /** Embed already-tokenized ids into a `[dModel]` sentence embedding. */
    public fun embed(tokenIds: IntArray): Tensor<T, Float> {
        val memory = t5.encode(t5.embed(tokenIds)) // [seqLen, dModel]
        return memory.mean(dim = 0)                // [dModel]
    }
}
