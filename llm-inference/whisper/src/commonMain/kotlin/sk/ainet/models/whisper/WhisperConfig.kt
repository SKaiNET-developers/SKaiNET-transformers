package sk.ainet.models.whisper

/**
 * Whisper architecture parameters. Defaults are whisper-tiny **multilingual**
 * (`n_vocab=51865` — e.g. `primeline/whisper-tiny-german-1224`); English-only
 * checkpoints differ only in `vocabSize=51864`.
 *
 * Encoder: conv1 (nMels→dim, k=3, s=1, p=1) → GELU → conv2 (dim→dim, k=3, s=2, p=1)
 * → GELU → +sinusoid positions → [encoderLayers] pre-norm blocks (biased MHA except
 * k_proj, GELU MLP) → final LayerNorm (`ln_post`).
 *
 * Decoder: learned positions, [decoderLayers] pre-norm blocks of causal self-attn →
 * cross-attn(encoder memory) → GELU MLP, final LayerNorm, logits via the TIED token
 * embedding (`logits = h @ embed_tokensᵀ`; there is no separate lm_head tensor).
 *
 * [audioCtx] is the encoder sequence length AFTER the stride-2 conv — melFrames/2.
 * The stock model uses 1500 (30 s); the Android pipeline builds at 200 (4 s): the
 * sinusoid positional embedding is deterministic, so constructing at a short context
 * is exact (validated ≥3 s; below is out-of-distribution and breaks decoding).
 */
public data class WhisperConfig(
    val nMels: Int = 80,
    val dim: Int = 384,             // n_audio_state == n_text_state (tiny)
    val nHeads: Int = 6,
    val headDim: Int = 64,          // 6 * 64 = 384
    val encoderLayers: Int = 4,
    val decoderLayers: Int = 4,
    val ffnDim: Int = 1536,         // 4 * dim
    val vocabSize: Int = 51865,     // multilingual; 51864 for *.en
    val maxTextPositions: Int = 448, // decoder learned positional table rows
    val audioCtx: Int = 200,        // encoder positions; melFrames = 2 * audioCtx
    val layerNormEps: Float = 1e-5f,
) {
    val melFrames: Int get() = audioCtx * 2
    val isMultilingual: Boolean get() = vocabSize >= 51865
}
