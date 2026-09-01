package sk.ainet.models.qwen

import sk.ainet.lang.nn.dsl.decoder.DecoderTensorNames

/**
 * Maps HuggingFace SafeTensors names for Qwen2 to GGUF canonical names.
 *
 * Qwen2 uses the same HuggingFace naming convention as LLaMA:
 * `model.layers.N.self_attn.q_proj.weight`, `model.embed_tokens.weight`, etc.
 *
 * This mapper handles the identical mapping, reusing [DecoderTensorNames] constants.
 */
public object QwenHfTensorNameMapper {

    private val LAYER_PATTERN = Regex("""model\.layers\.(\d+)\.(.+)""")

    /**
     * Convert a HuggingFace tensor name to its GGUF canonical equivalent.
     * Returns null if the tensor should be skipped.
     */
    public fun toCanonical(hfName: String): String? {
        return when (hfName) {
            "model.embed_tokens.weight" -> DecoderTensorNames.TOKEN_EMBEDDINGS
            "model.norm.weight" -> DecoderTensorNames.OUTPUT_NORM
            "lm_head.weight" -> DecoderTensorNames.OUTPUT_WEIGHT
            else -> {
                val match = LAYER_PATTERN.matchEntire(hfName) ?: return null
                val layer = match.groupValues[1].toInt()
                when (match.groupValues[2]) {
                    "input_layernorm.weight" -> DecoderTensorNames.attnNorm(layer)
                    "self_attn.q_proj.weight" -> DecoderTensorNames.attnQ(layer)
                    "self_attn.k_proj.weight" -> DecoderTensorNames.attnK(layer)
                    "self_attn.v_proj.weight" -> DecoderTensorNames.attnV(layer)
                    "self_attn.o_proj.weight" -> DecoderTensorNames.attnOut(layer)
                    "post_attention_layernorm.weight" -> DecoderTensorNames.ffnNorm(layer)
                    "mlp.gate_proj.weight" -> DecoderTensorNames.ffnGate(layer)
                    "mlp.down_proj.weight" -> DecoderTensorNames.ffnDown(layer)
                    "mlp.up_proj.weight" -> DecoderTensorNames.ffnUp(layer)
                    else -> null
                }
            }
        }
    }
}
