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

    public const val ACOUSTIC_INPUT_PROJ: String = "acoustic.input_proj.weight"
    public const val ACOUSTIC_INPUT_PROJ_BIAS: String = "acoustic.input_proj.bias"
    public const val ACOUSTIC_OUTPUT_PROJ: String = "acoustic.output_proj.weight"
    public const val ACOUSTIC_OUTPUT_PROJ_BIAS: String = "acoustic.output_proj.bias"
    public const val ACOUSTIC_NORM: String = "acoustic.output_norm.weight"
    public const val ACOUSTIC_LLM_PROJ: String = "acoustic.llm_proj.weight"
    public const val ACOUSTIC_TIME_PROJ: String = "acoustic.time_proj.weight"
    public const val ACOUSTIC_SEMANTIC_OUTPUT: String = "acoustic.semantic_output.weight"

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

    // Codebook
    public const val CODEC_SEMANTIC_CODEBOOK: String = "codec.semantic_codebook.weight"

    // Output projection (weight-normalized conv1d)
    public const val CODEC_OUTPUT_PROJ_G: String = "codec.output_proj.weight_g"
    public const val CODEC_OUTPUT_PROJ_V: String = "codec.output_proj.weight_v"
    public const val CODEC_OUTPUT_PROJ_BIAS: String = "codec.output_proj.bias"
    // Fallback: pre-composed weight (for non-weight-norm models or pre-composed weights)
    public const val CODEC_OUTPUT_PROJ: String = "codec.output_proj.weight"

    // Decoder block convolutions (weight-normalized).
    // Blocks are flat-indexed 0–7: even blocks [0,2,4,6] are convolutions,
    // odd blocks [1,3,5,7] are transformer stages.
    public fun codecBlockConvG(block: Int): String = "codec.decoder_blocks.$block.conv.weight_g"
    public fun codecBlockConvV(block: Int): String = "codec.decoder_blocks.$block.conv.weight_v"
    public fun codecBlockConvBias(block: Int): String = "codec.decoder_blocks.$block.conv.bias"
    // Fallback: pre-composed conv weight
    public fun codecBlockConvWeight(block: Int): String = "codec.decoder_blocks.$block.conv.weight"

    // Decoder block transformer layers (blocks 1,3,5,7 each have 2 layers)
    // Attention
    public fun codecTransformerAttnNorm(block: Int, layer: Int): String =
        "codec.decoder_blocks.$block.layers.$layer.attention_norm.weight"
    public fun codecTransformerAttnQ(block: Int, layer: Int): String =
        "codec.decoder_blocks.$block.layers.$layer.attention.wq.weight"
    public fun codecTransformerAttnK(block: Int, layer: Int): String =
        "codec.decoder_blocks.$block.layers.$layer.attention.wk.weight"
    public fun codecTransformerAttnV(block: Int, layer: Int): String =
        "codec.decoder_blocks.$block.layers.$layer.attention.wv.weight"
    public fun codecTransformerAttnOut(block: Int, layer: Int): String =
        "codec.decoder_blocks.$block.layers.$layer.attention.wo.weight"
    // QK norm
    public fun codecTransformerQNorm(block: Int, layer: Int): String =
        "codec.decoder_blocks.$block.layers.$layer.attention.q_norm.weight"
    public fun codecTransformerKNorm(block: Int, layer: Int): String =
        "codec.decoder_blocks.$block.layers.$layer.attention.k_norm.weight"
    // Layer scale
    public fun codecTransformerAttnScale(block: Int, layer: Int): String =
        "codec.decoder_blocks.$block.layers.$layer.attention_scale"
    public fun codecTransformerFfnScale(block: Int, layer: Int): String =
        "codec.decoder_blocks.$block.layers.$layer.ffn_scale"
    // FFN
    public fun codecTransformerFfnNorm(block: Int, layer: Int): String =
        "codec.decoder_blocks.$block.layers.$layer.ffn_norm.weight"
    public fun codecTransformerFfnGate(block: Int, layer: Int): String =
        "codec.decoder_blocks.$block.layers.$layer.feed_forward.w1.weight"
    public fun codecTransformerFfnDown(block: Int, layer: Int): String =
        "codec.decoder_blocks.$block.layers.$layer.feed_forward.w2.weight"
    public fun codecTransformerFfnUp(block: Int, layer: Int): String =
        "codec.decoder_blocks.$block.layers.$layer.feed_forward.w3.weight"
}
