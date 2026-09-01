package sk.ainet.lang.nn.dsl.decoder

/**
 * Architecture-neutral metadata for decoder-only transformer models.
 *
 * Captures the structural shape (dim, heads, layers, FFN size, vocab, context)
 * plus the architecture-specific knobs that show up consistently across
 * decoder LLMs (RoPE base, RMSNorm epsilon, BOS/EOS token ids).
 *
 * Per-architecture metadata types (`GgufDecoderMetadata`, `VoxtralModelMetadata`,
 * etc.) implement this so that [decoderTransformerNetwork] can build a network
 * from any of them without depending on a specific model module.
 *
 * QK-norm and FFN kind are not on this interface — they are *behavioral*
 * choices passed explicitly to [decoderTransformerNetwork] by each model's
 * `xNetwork(metadata)` function.
 */
public interface DecoderModelMetadata {
    /** Hidden / embedding dimension (`d_model`). */
    public val embeddingLength: Int

    /** Maximum sequence length the model was trained for. */
    public val contextLength: Int

    /** Number of transformer layers. */
    public val blockCount: Int

    /** Number of attention heads in Q. */
    public val headCount: Int

    /** Number of KV heads (for GQA / MQA — equal to [headCount] for MHA). */
    public val kvHeadCount: Int

    /** FFN intermediate dimension. */
    public val feedForwardLength: Int

    /**
     * Per-head dimension if explicitly specified by the model file;
     * otherwise null, in which case `embeddingLength / headCount` is used.
     */
    public val ropeDimensionCount: Int?

    /** Vocabulary size. */
    public val vocabSize: Int

    /** RoPE frequency base (typically 10_000 for Llama, 1_000_000 for Qwen3). */
    public val ropeFreqBase: Float

    /** RMSNorm epsilon. */
    public val rmsNormEps: Float

    /** Beginning-of-sequence token id. */
    public val bosTokenId: Int

    /** End-of-sequence token id. */
    public val eosTokenId: Int
}
