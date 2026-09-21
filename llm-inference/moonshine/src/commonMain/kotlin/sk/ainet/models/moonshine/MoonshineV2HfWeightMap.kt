package sk.ainet.models.moonshine

/**
 * Where a DSL parameter of the Moonshine **v2** streaming model comes from in a Hugging Face
 * `moonshine_streaming` checkpoint (`model.safetensors`), and what has to happen to the values on the way.
 *
 * The DSL projects through `x @ Wᵀ` with `W` in `[out, in]` — the layout of `torch.nn.Linear` — so the
 * checkpoint's linear weights map **as they are**. Three things are not a plain copy:
 *
 *  - the frontend filterbank is applied as `frames @ W` with `W` in `[frame, dim]`, the transpose of the
 *    checkpoint's `Linear` ([Transform.TRANSPOSE]);
 *  - the **encoder** LayerNorm scales are stored zero-centred (`gamma`): the effective scale is `gamma + 1`
 *    ([Transform.PLUS_ONE]). The decoder norms (`weight`) are plain;
 *  - the model's LayerNorms and projections carry no bias while the DSL modules do: those map to zeros
 *    ([hfName] `null`).
 */
public data class MoonshineV2WeightRef(
    /** Tensor name in the checkpoint; `null` = synthesize zeros of the parameter's shape. */
    val hfName: String?,
    val transform: Transform = Transform.NONE,
) {
    public enum class Transform { NONE, TRANSPOSE, PLUS_ONE }
}

/** DSL parameter name → checkpoint tensor, for every module of the v2 streaming model. */
public object MoonshineV2HfWeightMap {

    /** Token embedding table `[vocab, decoderDim]`; the decode loop looks rows up on the host. */
    public const val EMBED_TOKENS: String = "model.decoder.embed_tokens.weight"

    /**
     * Explicit output-head tensors, in order of preference. When the checkpoint has none, the head is tied to
     * [EMBED_TOKENS]. The config's `tie_word_embeddings` flag is not a reliable indicator; the tensor list is.
     */
    public val OUTPUT_HEADS: List<String> = listOf("proj_out.weight", "lm_head.weight")

    private val ZEROS = MoonshineV2WeightRef(null)
    private val ENC_LAYER = Regex("""^enc\.(\d+)\.(.+)$""")
    private val DEC_LAYER = Regex("""^dec\.(\d+)\.(.+)$""")

    /** [MoonshineV2Frontend] parameters. */
    public fun frontend(dslName: String): MoonshineV2WeightRef? {
        val p = "model.encoder.embedder"
        return when (dslName) {
            "fe_filterbank.weight" -> MoonshineV2WeightRef("$p.linear.weight", MoonshineV2WeightRef.Transform.TRANSPOSE)
            "fe_conv1.weight" -> MoonshineV2WeightRef("$p.conv1.weight")
            "fe_conv1.bias" -> MoonshineV2WeightRef("$p.conv1.bias")
            "fe_conv2.weight" -> MoonshineV2WeightRef("$p.conv2.weight")
            "fe_conv2.bias" -> MoonshineV2WeightRef("$p.conv2.bias")
            "fe_log_k" -> MoonshineV2WeightRef("$p.comp.log_k")
            else -> null
        }
    }

    /** [moonshineV2Encoder] and [MoonshineV2Adapter] parameters (one map: the namespaces do not overlap). */
    public fun encoder(dslName: String): MoonshineV2WeightRef? {
        when (dslName) {
            "enc_out_norm.weight" -> return MoonshineV2WeightRef("model.encoder.final_norm.gamma", MoonshineV2WeightRef.Transform.PLUS_ONE)
            "enc_out_norm.bias" -> return ZEROS
            "v2_adapter.pos_embed.weight" -> return MoonshineV2WeightRef("model.decoder.pos_emb.weight")
            // Split-width checkpoints only: bias-free memory projection [decoderDim, dim].
            "v2_adapter.proj.weight" -> return MoonshineV2WeightRef("model.decoder.proj.weight")
            "v2_adapter.proj.bias" -> return ZEROS
        }
        val m = ENC_LAYER.matchEntire(dslName) ?: return null
        val p = "model.encoder.layers.${m.groupValues[1]}"
        return when (m.groupValues[2]) {
            "attn_norm.weight" -> MoonshineV2WeightRef("$p.input_layernorm.gamma", MoonshineV2WeightRef.Transform.PLUS_ONE)
            "ffn_norm.weight" -> MoonshineV2WeightRef("$p.post_attention_layernorm.gamma", MoonshineV2WeightRef.Transform.PLUS_ONE)
            "attn_norm.bias", "ffn_norm.bias" -> ZEROS
            "attn.q_proj.weight" -> MoonshineV2WeightRef("$p.self_attn.q_proj.weight")
            "attn.k_proj.weight" -> MoonshineV2WeightRef("$p.self_attn.k_proj.weight")
            "attn.v_proj.weight" -> MoonshineV2WeightRef("$p.self_attn.v_proj.weight")
            "attn.o_proj.weight" -> MoonshineV2WeightRef("$p.self_attn.o_proj.weight")
            "ffn_up.weight" -> MoonshineV2WeightRef("$p.mlp.fc1.weight")
            "ffn_up.bias" -> MoonshineV2WeightRef("$p.mlp.fc1.bias")
            "ffn_down.weight" -> MoonshineV2WeightRef("$p.mlp.fc2.weight")
            "ffn_down.bias" -> MoonshineV2WeightRef("$p.mlp.fc2.bias")
            else -> null
        }
    }

    /** [moonshineV2Decoder] parameters. [outputHead] is the tensor chosen from [OUTPUT_HEADS], or [EMBED_TOKENS]. */
    public fun decoder(dslName: String, outputHead: String = EMBED_TOKENS): MoonshineV2WeightRef? {
        when (dslName) {
            "dec_out_norm.weight" -> return MoonshineV2WeightRef("model.decoder.norm.weight")
            "lm_head.weight" -> return MoonshineV2WeightRef(outputHead)
            "dec_out_norm.bias", "lm_head.bias" -> return ZEROS
        }
        val m = DEC_LAYER.matchEntire(dslName) ?: return null
        val p = "model.decoder.layers.${m.groupValues[1]}"
        return when (val leaf = m.groupValues[2]) {
            "self_attn_norm.weight" -> MoonshineV2WeightRef("$p.input_layernorm.weight")
            "cross_attn_norm.weight" -> MoonshineV2WeightRef("$p.post_attention_layernorm.weight")
            "mlp_norm.weight" -> MoonshineV2WeightRef("$p.final_layernorm.weight")
            "self_attn_norm.bias", "cross_attn_norm.bias", "mlp_norm.bias" -> ZEROS
            "self_attn.q_proj.weight", "self_attn.k_proj.weight",
            "self_attn.v_proj.weight", "self_attn.o_proj.weight" -> MoonshineV2WeightRef("$p.$leaf")
            "cross_attn.q_proj.weight", "cross_attn.k_proj.weight",
            "cross_attn.v_proj.weight", "cross_attn.o_proj.weight" ->
                MoonshineV2WeightRef("$p.encoder_attn.${leaf.removePrefix("cross_attn.")}")
            "mlp_fc1.weight" -> MoonshineV2WeightRef("$p.mlp.fc1.weight")
            "mlp_fc1.bias" -> MoonshineV2WeightRef("$p.mlp.fc1.bias")
            "mlp_fc2.weight" -> MoonshineV2WeightRef("$p.mlp.fc2.weight")
            "mlp_fc2.bias" -> MoonshineV2WeightRef("$p.mlp.fc2.bias")
            else -> null
        }
    }
}
