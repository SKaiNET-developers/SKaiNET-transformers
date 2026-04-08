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
    private val ACOUSTIC_LAYER_PATTERN = Regex("""acoustic_(?:model|transformer)\.layers\.(\d+)\.(.+)""")
    private val CODEC_DECODER_PATTERN = Regex("""codec\.(.+)""")
    private val AUDIO_TOKENIZER_PATTERN = Regex("""audio_tokenizer\.(.+)""")

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

        // Acoustic model global tensors (both acoustic_model.* and acoustic_transformer.* prefixes)
        when (hfName) {
            "acoustic_model.norm.weight",
            "acoustic_transformer.norm.weight" -> return VoxtralTensorNames.ACOUSTIC_NORM
            "acoustic_model.input_projection.weight",
            "acoustic_transformer.input_projection.weight" -> return VoxtralTensorNames.ACOUSTIC_INPUT_PROJ
            "acoustic_model.input_projection.bias",
            "acoustic_transformer.input_projection.bias" -> return VoxtralTensorNames.ACOUSTIC_INPUT_PROJ_BIAS
            "acoustic_model.output_projection.weight",
            "acoustic_transformer.acoustic_codebook_output.weight" -> return VoxtralTensorNames.ACOUSTIC_OUTPUT_PROJ
            "acoustic_model.output_projection.bias",
            "acoustic_transformer.output_projection.bias" -> return VoxtralTensorNames.ACOUSTIC_OUTPUT_PROJ_BIAS
            // Additional acoustic transformer projections
            "acoustic_transformer.llm_projection.weight" -> return "acoustic.llm_proj.weight"
            "acoustic_transformer.time_projection.weight" -> return "acoustic.time_proj.weight"
            "acoustic_transformer.semantic_codebook_output.weight" -> return "acoustic.semantic_output.weight"
        }

        // Codec tensors — audio_tokenizer.* prefix (Mistral naming)
        val audioTokMatch = AUDIO_TOKENIZER_PATTERN.matchEntire(hfName)
        if (audioTokMatch != null) {
            return mapAudioTokenizerTensor(audioTokMatch.groupValues[1])
        }

        // Codec tensors — codec.* prefix (already canonical)
        val codecMatch = CODEC_DECODER_PATTERN.matchEntire(hfName)
        if (codecMatch != null) {
            return mapCodecTensor(hfName)
        }

        // Backbone global tensors (Mistral naming)
        return when (hfName) {
            "tok_embeddings.weight",
            "mm_audio_embeddings.tok_embeddings.weight" -> LlamaTensorNames.TOKEN_EMBEDDINGS
            "norm.weight" -> LlamaTensorNames.OUTPUT_NORM
            "output.weight" -> LlamaTensorNames.OUTPUT_WEIGHT
            // Audio codebook embeddings (backbone-level, for audio token encoding)
            "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight" ->
                "audio_codebook_embeddings.weight"
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
     * Map `audio_tokenizer.*` tensor names from Mistral SafeTensors to canonical `codec.*` names.
     *
     * Mistral naming examples:
     * - `audio_tokenizer.quantizer.semantic_codebook.embedding_sum` → semantic codebook
     * - `audio_tokenizer.decoder_blocks.{b}.conv.parametrizations.weight.original0` → conv weight_g
     * - `audio_tokenizer.decoder_blocks.{b}.conv.parametrizations.weight.original1` → conv weight_v
     * - `audio_tokenizer.decoder_blocks.{b}.layers.{l}.attention.wq.weight` → transformer attn
     * - `audio_tokenizer.output_proj.conv.parametrizations.weight.original0` → output proj weight_g
     */
    private fun mapAudioTokenizerTensor(suffix: String): String? {
        // Semantic codebook
        if (suffix == "quantizer.semantic_codebook.embedding_sum") {
            return VoxtralTensorNames.CODEC_SEMANTIC_CODEBOOK
        }

        // Output projection (weight-normalized conv)
        if (suffix.startsWith("output_proj.conv.")) {
            return when {
                suffix.endsWith("parametrizations.weight.original0") ->
                    VoxtralTensorNames.CODEC_OUTPUT_PROJ_G
                suffix.endsWith("parametrizations.weight.original1") ->
                    VoxtralTensorNames.CODEC_OUTPUT_PROJ_V
                suffix.endsWith(".bias") ->
                    VoxtralTensorNames.CODEC_OUTPUT_PROJ_BIAS
                suffix.endsWith(".weight") ->
                    VoxtralTensorNames.CODEC_OUTPUT_PROJ
                else -> "codec.$suffix"
            }
        }

        // Decoder blocks
        val blockPattern = Regex("""decoder_blocks\.(\d+)\.(.+)""")
        val blockMatch = blockPattern.matchEntire(suffix)
        if (blockMatch != null) {
            val blockIdx = blockMatch.groupValues[1].toInt()
            val blockSuffix = blockMatch.groupValues[2]
            return mapDecoderBlock(blockIdx, blockSuffix)
        }

        // Pass through anything else with codec. prefix
        return "codec.$suffix"
    }

    /**
     * Map a decoder block tensor. Blocks alternate: conv (even) / transformer (odd).
     */
    private fun mapDecoderBlock(block: Int, suffix: String): String? {
        // Convolution weights (with weight normalization)
        if (suffix.startsWith("conv.")) {
            return when {
                suffix.endsWith("parametrizations.weight.original0") ->
                    VoxtralTensorNames.codecBlockConvG(block)
                suffix.endsWith("parametrizations.weight.original1") ->
                    VoxtralTensorNames.codecBlockConvV(block)
                suffix.endsWith(".bias") ->
                    VoxtralTensorNames.codecBlockConvBias(block)
                suffix.endsWith(".weight") ->
                    VoxtralTensorNames.codecBlockConvWeight(block)
                else -> "codec.decoder_blocks.$block.$suffix"
            }
        }

        // Transformer layers
        val layerPattern = Regex("""layers\.(\d+)\.(.+)""")
        val layerMatch = layerPattern.matchEntire(suffix)
        if (layerMatch != null) {
            val layer = layerMatch.groupValues[1].toInt()
            val layerSuffix = layerMatch.groupValues[2]
            return mapCodecTransformerLayer(block, layer, layerSuffix)
        }

        return "codec.decoder_blocks.$block.$suffix"
    }

    /**
     * Map codec transformer layer tensor names to canonical form.
     */
    private fun mapCodecTransformerLayer(block: Int, layer: Int, suffix: String): String? {
        return when (suffix) {
            "attention_norm.weight" -> VoxtralTensorNames.codecTransformerAttnNorm(block, layer)
            "attention.wq.weight" -> VoxtralTensorNames.codecTransformerAttnQ(block, layer)
            "attention.wk.weight" -> VoxtralTensorNames.codecTransformerAttnK(block, layer)
            "attention.wv.weight" -> VoxtralTensorNames.codecTransformerAttnV(block, layer)
            "attention.wo.weight" -> VoxtralTensorNames.codecTransformerAttnOut(block, layer)
            "attention.q_norm.weight" -> VoxtralTensorNames.codecTransformerQNorm(block, layer)
            "attention.k_norm.weight" -> VoxtralTensorNames.codecTransformerKNorm(block, layer)
            "attention_scale" -> VoxtralTensorNames.codecTransformerAttnScale(block, layer)
            "ffn_scale" -> VoxtralTensorNames.codecTransformerFfnScale(block, layer)
            "ffn_norm.weight" -> VoxtralTensorNames.codecTransformerFfnNorm(block, layer)
            "feed_forward.w1.weight" -> VoxtralTensorNames.codecTransformerFfnGate(block, layer)
            "feed_forward.w2.weight" -> VoxtralTensorNames.codecTransformerFfnDown(block, layer)
            "feed_forward.w3.weight" -> VoxtralTensorNames.codecTransformerFfnUp(block, layer)
            else -> "codec.decoder_blocks.$block.layers.$layer.$suffix"
        }
    }

    /**
     * Map codec tensor names. Returns the name prefixed with `codec.` for canonical form.
     * Codec tensors are passed through with their original naming since they are
     * already in canonical form.
     */
    private fun mapCodecTensor(hfName: String): String? {
        return hfName
    }
}
