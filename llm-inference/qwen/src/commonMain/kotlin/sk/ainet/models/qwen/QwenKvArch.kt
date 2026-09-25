package sk.ainet.models.qwen

/**
 * Architecture facts the qwen-kv-v1 export/runtime contract needs, DERIVED from a loaded
 * checkpoint at export time (see [deriveFrom]) rather than hardcoded per model — unlike
 * `FunctionGemmaSpec`, which describes exactly one checkpoint, this same [QwenKvArch] class
 * (and the rest of [QwenKvContract] / `QwenExportHarness`) serves both Qwen2.5-0.5B-Instruct
 * (arch `qwen2`, GQA 14/2 heads, headDim 64, attention bias, no QK-norm) and Qwen3-0.6B (arch
 * `qwen3`, GQA 16/8 heads, headDim 128, QK-norm, no bias) — SKaiNET-transformers#411.
 *
 * `[qwen25_05bInstruct]`/`[qwen3_06b]` mirror `IreeKvSession.IreeKvSpec`'s own factories of the
 * same names (`llm-runtime/iree-android`) for tests and fixtures that need the numbers without a
 * checkpoint on disk; the harness always derives the real thing via [deriveFrom].
 */
public data class QwenKvArch(
    public val architecture: String,
    public val nLayers: Int,
    public val headDim: Int,
    public val nKvHeads: Int,
    public val nHeads: Int,
    public val hiddenSize: Int,
    public val vocabSize: Int,
    public val ropeBase: Float,
    public val rmsNormEps: Float,
    /** Qwen3: true (`attn_q_norm`/`attn_k_norm` present). Qwen2.5: false. */
    public val qkNorm: Boolean,
    /** Qwen2.5 (arch `qwen2`): true (`attn_q.bias` etc. present). Qwen3: false. */
    public val attnBias: Boolean,
    /** `<|endoftext|>` — both models' `tokenizer.ggml.eos_token_id`. */
    public val eos: Int,
    /** `<|im_end|>` — the ChatML turn-end token both models actually stop generation on. */
    public val eot: Int,
) {
    init {
        require(nLayers > 0 && headDim > 0 && nKvHeads > 0 && nHeads > 0) { "bad architecture constants" }
        require(nHeads % nKvHeads == 0) { "nHeads ($nHeads) must be a multiple of nKvHeads ($nKvHeads)" }
        require(hiddenSize > 0 && vocabSize > 0) { "bad architecture constants" }
    }

    public companion object {
        /** `<|endoftext|>` — the id both Qwen2.5 and Qwen3 GGUFs share for `tokenizer.ggml.eos_token_id`. */
        public const val ENDOFTEXT: Int = 151643

        /** `<|im_end|>` — the actual per-turn stop token (the tokenizer's nominal `eos_token_id` is `<|endoftext|>`,
         *  but the ChatML template only ever emits `<|im_end|>` at the end of a turn). */
        public const val IM_END: Int = 151645

        /** Qwen2.5-0.5B-Instruct (arch `qwen2`), read off `Qwen/Qwen2.5-0.5B-Instruct-GGUF`'s metadata. */
        public fun qwen25_05bInstruct(): QwenKvArch = QwenKvArch(
            architecture = "qwen2", nLayers = 24, headDim = 64, nKvHeads = 2, nHeads = 14,
            hiddenSize = 896, vocabSize = 151936, ropeBase = 1_000_000f, rmsNormEps = 1e-6f,
            qkNorm = false, attnBias = true, eos = ENDOFTEXT, eot = IM_END,
        )

        /** Qwen3-0.6B (arch `qwen3`), read off `Qwen/Qwen3-0.6B-GGUF`'s metadata. */
        public fun qwen3_06b(): QwenKvArch = QwenKvArch(
            architecture = "qwen3", nLayers = 28, headDim = 128, nKvHeads = 8, nHeads = 16,
            hiddenSize = 1024, vocabSize = 151936, ropeBase = 1_000_000f, rmsNormEps = 1e-6f,
            qkNorm = true, attnBias = false, eos = ENDOFTEXT, eot = IM_END,
        )
    }
}
