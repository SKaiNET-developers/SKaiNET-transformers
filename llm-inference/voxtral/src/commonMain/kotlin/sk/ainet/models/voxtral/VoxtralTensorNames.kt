package sk.ainet.models.voxtral

import sk.ainet.models.llama.LlamaTensorNames

/**
 * Canonical tensor names for the Voxtral TTS model.
 *
 * Voxtral has three sets of tensors:
 * 1. **Backbone** — uses standard LLaMA GGUF naming (`token_embd`, `blk.N.*`, `output`)
 * 2. **Acoustic model** — prefixed with `acoustic.` to avoid collisions
 * 3. **Codec** — prefixed with `codec.` (convolutional + transformer layers)
 *
 * The backbone tensor names are identical to [LlamaTensorNames] since Voxtral's
 * text backbone is a Ministral-3B (LLaMA architecture).
 */
public object VoxtralTensorNames {

    // ========== Backbone (delegates to LlamaTensorNames) ==========

    public const val TOKEN_EMBEDDINGS: String = "token_embd.weight"
    public const val OUTPUT_NORM: String = "output_norm.weight"
    public const val OUTPUT_WEIGHT: String = "output.weight"

    public fun attnNorm(layer: Int): String = LlamaTensorNames.attnNorm(layer)
    public fun attnQ(layer: Int): String = LlamaTensorNames.attnQ(layer)
    public fun attnK(layer: Int): String = LlamaTensorNames.attnK(layer)
    public fun attnV(layer: Int): String = LlamaTensorNames.attnV(layer)
    public fun attnOut(layer: Int): String = LlamaTensorNames.attnOut(layer)
    public fun ffnNorm(layer: Int): String = LlamaTensorNames.ffnNorm(layer)
    public fun ffnGate(layer: Int): String = LlamaTensorNames.ffnGate(layer)
    public fun ffnDown(layer: Int): String = LlamaTensorNames.ffnDown(layer)
    public fun ffnUp(layer: Int): String = LlamaTensorNames.ffnUp(layer)

    // ========== Acoustic Model ==========

    public const val ACOUSTIC_NORM: String = "acoustic.output_norm.weight"

    public fun acousticAttnNorm(layer: Int): String = "acoustic.blk.$layer.attn_norm.weight"
    public fun acousticAttnQ(layer: Int): String = "acoustic.blk.$layer.attn_q.weight"
    public fun acousticAttnK(layer: Int): String = "acoustic.blk.$layer.attn_k.weight"
    public fun acousticAttnV(layer: Int): String = "acoustic.blk.$layer.attn_v.weight"
    public fun acousticAttnOut(layer: Int): String = "acoustic.blk.$layer.attn_output.weight"
    public fun acousticFfnNorm(layer: Int): String = "acoustic.blk.$layer.ffn_norm.weight"
    public fun acousticFfnGate(layer: Int): String = "acoustic.blk.$layer.ffn_gate.weight"
    public fun acousticFfnDown(layer: Int): String = "acoustic.blk.$layer.ffn_down.weight"
    public fun acousticFfnUp(layer: Int): String = "acoustic.blk.$layer.ffn_up.weight"

    // ========== Codec ==========

    public const val CODEC_SEMANTIC_CODEBOOK: String = "codec.semantic_codebook.weight"
    public const val CODEC_ACOUSTIC_CODEBOOK: String = "codec.acoustic_codebook.weight"
    public const val CODEC_PATCH_PROJ: String = "codec.patch_proj.weight"
    public const val CODEC_PATCH_PROJ_BIAS: String = "codec.patch_proj.bias"
    public const val CODEC_OUTPUT_PROJ: String = "codec.output_proj.weight"
    public const val CODEC_OUTPUT_PROJ_BIAS: String = "codec.output_proj.bias"

    public fun codecDecoderConvWeight(stage: Int): String = "codec.decoder.conv.$stage.weight"
    public fun codecDecoderConvBias(stage: Int): String = "codec.decoder.conv.$stage.bias"
    public fun codecDecoderTransformerAttnNorm(stage: Int, layer: Int): String =
        "codec.decoder.transformer.$stage.blk.$layer.attn_norm.weight"
    public fun codecDecoderTransformerAttnQ(stage: Int, layer: Int): String =
        "codec.decoder.transformer.$stage.blk.$layer.attn_q.weight"
    public fun codecDecoderTransformerAttnK(stage: Int, layer: Int): String =
        "codec.decoder.transformer.$stage.blk.$layer.attn_k.weight"
    public fun codecDecoderTransformerAttnV(stage: Int, layer: Int): String =
        "codec.decoder.transformer.$stage.blk.$layer.attn_v.weight"
    public fun codecDecoderTransformerAttnOut(stage: Int, layer: Int): String =
        "codec.decoder.transformer.$stage.blk.$layer.attn_output.weight"
}
