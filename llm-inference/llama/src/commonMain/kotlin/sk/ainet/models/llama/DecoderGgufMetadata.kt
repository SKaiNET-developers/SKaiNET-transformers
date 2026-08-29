package sk.ainet.models.llama

import sk.ainet.io.gguf.StreamingTensorInfo

/**
 * [LlamaModelMetadata] from the llama.cpp-convention GGUF KV fields
 * (`{arch}.embedding_length`, …) plus the header's tensor directory for the
 * values small checkpoints omit (embedding/vocab off `token_embd.weight`'s
 * shape, head_dim off `blk.0.attn_q.weight`'s).
 *
 * The single metadata parser for every decoder-shaped family (#346):
 * [DecoderGgufWeightLoader] (llama/qwen/mistral/smollm2) and the BitNet
 * loaders both read it — the per-loader private copies this replaces had
 * already drifted in their numeric coercions and inference fallbacks.
 */
public fun decoderMetadataFromGguf(
    fields: Map<String, Any?>,
    tensors: List<StreamingTensorInfo>,
): LlamaModelMetadata {
    val arch = (fields["general.architecture"] as? String) ?: "unknown"
    val prefix = arch

    val embeddingLength = fields["$prefix.embedding_length"]?.toIntValue()
        ?: inferEmbeddingFromTensors(tensors)
    val contextLength = fields["$prefix.context_length"]?.toIntValue() ?: 0
    val blockCount = fields["$prefix.block_count"]?.toIntValue() ?: 0
    val headCount = fields["$prefix.attention.head_count"]?.toIntValue() ?: 0
    val kvHeadCount = fields["$prefix.attention.head_count_kv"]?.toIntValue() ?: headCount
    val feedForwardLength = fields["$prefix.feed_forward_length"]?.toIntValue() ?: 0
    var ropeDim = fields["$prefix.rope.dimension_count"]?.toIntValue()
    val vocabSize = fields["$prefix.vocab_size"]?.toIntValue()
        ?: inferVocabFromTensors(tensors)
    val ropeFreqBase = fields["$prefix.rope.freq_base"]?.toFloatValue() ?: 10_000f
    val rmsNormEps = fields["$prefix.attention.layer_norm_rms_epsilon"]?.toFloatValue() ?: 1e-5f
    val bosTokenId = fields["tokenizer.ggml.bos_token_id"]?.toIntValue() ?: 1
    val eosTokenId = fields["tokenizer.ggml.eos_token_id"]?.toIntValue() ?: 2

    // Infer head_dim from the Q weight shape when the rope dimension is not
    // set: Q is [q_dim, dim] with q_dim = nHeads * headDim.
    if (ropeDim == null && headCount > 0) {
        val qTensor = tensors.firstOrNull { it.name == "blk.0.attn_q.weight" }
        if (qTensor != null && qTensor.shape.size == 2) {
            val qDim = qTensor.shape.firstOrNull { it.toInt() != embeddingLength }?.toInt()
                ?: qTensor.shape[0].toInt() // square weight: both dims equal embeddingLength
            val inferredHeadDim = qDim / headCount
            if (inferredHeadDim > 0 && inferredHeadDim * headCount == qDim) {
                ropeDim = inferredHeadDim
            }
        }
        // Last resort (no Q tensor in the directory): the MHA default.
        if (ropeDim == null && embeddingLength % headCount == 0) {
            ropeDim = embeddingLength / headCount
        }
    }

    return LlamaModelMetadata(
        architecture = arch,
        embeddingLength = embeddingLength,
        contextLength = contextLength,
        blockCount = blockCount,
        headCount = headCount,
        kvHeadCount = kvHeadCount,
        feedForwardLength = feedForwardLength,
        ropeDimensionCount = ropeDim,
        vocabSize = vocabSize,
        ropeFreqBase = ropeFreqBase,
        rmsNormEps = rmsNormEps,
        bosTokenId = bosTokenId,
        eosTokenId = eosTokenId,
    )
}

private fun Any?.toIntValue(): Int? = when (this) {
    is Int -> this
    is UInt -> this.toInt()
    is Long -> this.toInt()
    is ULong -> this.toInt()
    is Short -> this.toInt()
    is UShort -> this.toInt()
    is Byte -> this.toInt()
    is UByte -> this.toInt()
    else -> null
}

private fun Any?.toFloatValue(): Float? = when (this) {
    is Float -> this
    is Double -> this.toFloat()
    is Int -> this.toFloat()
    is UInt -> this.toFloat()
    is Long -> this.toFloat()
    is ULong -> this.toFloat()
    else -> null
}

private fun inferEmbeddingFromTensors(tensors: List<StreamingTensorInfo>): Int {
    val token = tensors.firstOrNull { it.name == LlamaTensorNames.TOKEN_EMBEDDINGS }
        ?: error("Cannot infer embedding length without token embeddings tensor")
    return token.shape.map { it.toInt() }.minOrNull()
        ?: error("Cannot infer embedding length from tensor shape ${token.shape}")
}

private fun inferVocabFromTensors(tensors: List<StreamingTensorInfo>): Int {
    val token = tensors.firstOrNull { it.name == LlamaTensorNames.TOKEN_EMBEDDINGS }
        ?: error("Cannot infer vocab size without token embeddings tensor")
    return token.shape.map { it.toInt() }.maxOrNull()
        ?: error("Cannot infer vocab size from tensor shape ${token.shape}")
}
