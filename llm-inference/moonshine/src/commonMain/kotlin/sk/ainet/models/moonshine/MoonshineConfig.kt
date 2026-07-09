package sk.ainet.models.moonshine

/**
 * Moonshine-tiny architecture parameters (onnx-community/moonshine-tiny-ONNX).
 *
 * Encoder: a 3-conv audio frontend (raw 16 kHz audio → features) followed by
 * [encoderLayers] pre-LayerNorm transformer layers with RoPE self-attention (no
 * bias) and a plain GELU MLP. Decoder (later): self-attention + cross-attention
 * to the encoder memory.
 *
 * Verified against `enc_frontend.mlir` / `enc_xformer.mlir`:
 *   conv1: 1→[dim]      k=127 s=64   (tanh)
 *   conv2: [dim]→[2dim] k=7   s=3    (gelu)
 *   conv3: [2dim]→[dim] k=3   s=2    (gelu)  → [1, frames, dim]
 *   layer: LayerNorm → MHA(RoPE, non-causal, no bias) → +res
 *          LayerNorm → Linear(dim→ffn) → GELU → Linear(ffn→dim) → +res
 */
public data class MoonshineConfig(
    val dim: Int = 288,
    val encoderLayers: Int = 6,
    val decoderLayers: Int = 6,
    val nHeads: Int = 8,
    val headDim: Int = 36,          // 8 * 36 = 288
    val ffnDim: Int = 1152,         // 4 * dim (Moonshine-tiny MLP hidden)
    val vocabSize: Int = 32768,
    val maxAudioSamples: Int = 64000, // 4 s @ 16 kHz → 165 frames
    val maxFrames: Int = 165,         // encoder sequence length after the conv frontend
    val maxDecodeTokens: Int = 194,   // decoder RoPE table size = config max_position_embeddings (board uses ≤30)
    val ropeBase: Float = 10000.0f,
    // Moonshine uses PARTIAL rotary: only rotaryDim = headDim*partialRotaryFactor = 36*0.9 = 32
    // head dims are rotated (rotate-half / SPLIT_HALF), the trailing 4 pass through. Verified
    // against enc_xformer.onnx (cos/sin are [·,·,32] while head_dim is 36).
    val partialRotaryFactor: Float = 0.9f,
    val layerNormEps: Float = 1e-5f,
) {
    // Conv frontend (channels/kernel/stride), from the ONNX frontend graph.
    val conv1Out: Int get() = dim
    val conv2Out: Int get() = dim * 2
    val conv1Kernel: Int get() = 127
    val conv1Stride: Int get() = 64
    val conv2Kernel: Int get() = 7
    val conv2Stride: Int get() = 3
    val conv3Kernel: Int get() = 3
    val conv3Stride: Int get() = 2
}
