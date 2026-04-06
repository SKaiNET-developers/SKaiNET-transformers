package sk.ainet.models.voxtral

import sk.ainet.models.llama.LlamaTensorNames

/**
 * Maps HuggingFace/Mistral SafeTensors names for Voxtral to GGUF canonical names.
 *
 * Voxtral uses Mistral's SafeTensors naming convention which differs from HuggingFace:
 * - `tok_embeddings.weight` (not `model.embed_tokens.weight`)
 * - `layers.N.attention.wq.weight` (not `model.layers.N.self_attn.q_proj.weight`)
 * - `layers.N.feed_forward.w1.weight` (gate), `w2` (down), `w3` (up)
 * - `norm.weight` (not `model.norm.weight`)
 * - `output.weight` (not `lm_head.weight`)
 *
 * Acoustic model tensors are prefixed with `acoustic_model.`.
 * Codec tensors are prefixed with `codec.`.
 */
public object VoxtralHfTensorNameMapper {

    private val BACKBONE_LAYER_PATTERN = Regex("""layers\.(\d+)\.(.+)""")
    private val ACOUSTIC_LAYER_PATTERN = Regex("""acoustic_model\.layers\.(\d+)\.(.+)""")
    private val CODEC_DECODER_PATTERN = Regex("""codec\.(.+)""")

    /**
     * Convert a Mistral/HuggingFace tensor name to its GGUF canonical equivalent.
     * Returns null if the tensor should be skipped.
     */
    public fun toCanonical(hfName: String): String? {
        // Acoustic model tensors
        val acousticMatch = ACOUSTIC_LAYER_PATTERN.matchEntire(hfName)
        if (acousticMatch != null) {
            return mapAcousticLayer(
                acousticMatch.groupValues[1].toInt(),
                acousticMatch.groupValues[2]
            )
        }

        // Acoustic model global tensors
        if (hfName == "acoustic_model.norm.weight") {
            return VoxtralTensorNames.ACOUSTIC_NORM
        }

        // Codec tensors
        val codecMatch = CODEC_DECODER_PATTERN.matchEntire(hfName)
        if (codecMatch != null) {
            return mapCodecTensor(hfName)
        }

        // Backbone global tensors (Mistral naming)
        return when (hfName) {
            "tok_embeddings.weight" -> LlamaTensorNames.TOKEN_EMBEDDINGS
            "norm.weight" -> LlamaTensorNames.OUTPUT_NORM
            "output.weight" -> LlamaTensorNames.OUTPUT_WEIGHT
            // HuggingFace naming fallback
            "model.embed_tokens.weight" -> LlamaTensorNames.TOKEN_EMBEDDINGS
            "model.norm.weight" -> LlamaTensorNames.OUTPUT_NORM
            "lm_head.weight" -> LlamaTensorNames.OUTPUT_WEIGHT
            else -> {
                // Backbone layer tensors
                val match = BACKBONE_LAYER_PATTERN.matchEntire(hfName) ?: return null
                mapBackboneLayer(match.groupValues[1].toInt(), match.groupValues[2])
            }
        }
    }

    /**
     * Map backbone layer tensor names.
     * Supports both Mistral naming (attention.wq) and HuggingFace naming (self_attn.q_proj).
     */
    private fun mapBackboneLayer(layer: Int, suffix: String): String? {
        return when (suffix) {
            // Mistral naming
            "attention_norm.weight" -> LlamaTensorNames.attnNorm(layer)
            "attention.wq.weight" -> LlamaTensorNames.attnQ(layer)
            "attention.wk.weight" -> LlamaTensorNames.attnK(layer)
            "attention.wv.weight" -> LlamaTensorNames.attnV(layer)
            "attention.wo.weight" -> LlamaTensorNames.attnOut(layer)
            "ffn_norm.weight" -> LlamaTensorNames.ffnNorm(layer)
            "feed_forward.w1.weight" -> LlamaTensorNames.ffnGate(layer)
            "feed_forward.w2.weight" -> LlamaTensorNames.ffnDown(layer)
            "feed_forward.w3.weight" -> LlamaTensorNames.ffnUp(layer)
            // HuggingFace naming fallback
            "input_layernorm.weight" -> LlamaTensorNames.attnNorm(layer)
            "self_attn.q_proj.weight" -> LlamaTensorNames.attnQ(layer)
            "self_attn.k_proj.weight" -> LlamaTensorNames.attnK(layer)
            "self_attn.v_proj.weight" -> LlamaTensorNames.attnV(layer)
            "self_attn.o_proj.weight" -> LlamaTensorNames.attnOut(layer)
            "post_attention_layernorm.weight" -> LlamaTensorNames.ffnNorm(layer)
            "mlp.gate_proj.weight" -> LlamaTensorNames.ffnGate(layer)
            "mlp.down_proj.weight" -> LlamaTensorNames.ffnDown(layer)
            "mlp.up_proj.weight" -> LlamaTensorNames.ffnUp(layer)
            else -> null
        }
    }

    /**
     * Map acoustic model layer tensor names (same structure as backbone but prefixed).
     */
    private fun mapAcousticLayer(layer: Int, suffix: String): String? {
        return when (suffix) {
            "attention_norm.weight" -> VoxtralTensorNames.acousticAttnNorm(layer)
            "attention.wq.weight" -> VoxtralTensorNames.acousticAttnQ(layer)
            "attention.wk.weight" -> VoxtralTensorNames.acousticAttnK(layer)
            "attention.wv.weight" -> VoxtralTensorNames.acousticAttnV(layer)
            "attention.wo.weight" -> VoxtralTensorNames.acousticAttnOut(layer)
            "ffn_norm.weight" -> VoxtralTensorNames.acousticFfnNorm(layer)
            "feed_forward.w1.weight" -> VoxtralTensorNames.acousticFfnGate(layer)
            "feed_forward.w2.weight" -> VoxtralTensorNames.acousticFfnDown(layer)
            "feed_forward.w3.weight" -> VoxtralTensorNames.acousticFfnUp(layer)
            else -> null
        }
    }

    /**
     * Map codec tensor names. Returns the name prefixed with `codec.` for canonical form.
     * Codec tensors are passed through with their original naming since the codec
     * architecture uses convolutional layers not yet in the standard DSL.
     */
    private fun mapCodecTensor(hfName: String): String? {
        // Pass through codec tensors with canonical prefix
        return hfName
    }
}
