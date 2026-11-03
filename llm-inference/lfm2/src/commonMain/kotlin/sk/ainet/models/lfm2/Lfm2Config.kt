package sk.ainet.models.lfm2

import kotlinx.serialization.json.*

/** Dense LFM2 configuration supported by this FP32 implementation. */
data class Lfm2Config(
    val hiddenSize: Int,
    val kernelSize: Int,
    val layerTypes: List<String>,
    val attentionHeads: Int,
    val kvHeads: Int,
    val effectiveIntermediateSize: Int,
    val vocabularySize: Int,
    val normEps: Float,
    val ropeTheta: Double,
    val maximumPositions: Int,
    val bosTokenId: Int?,
    val eosTokenIds: Set<Int>,
) {
    init {
        require(hiddenSize > 0 && kernelSize > 0 && attentionHeads > 0 && kvHeads > 0)
        require(hiddenSize % attentionHeads == 0 && attentionHeads % kvHeads == 0)
        require(hiddenSize / attentionHeads % 2 == 0) { "Split-half RoPE requires an even head width" }
        require(effectiveIntermediateSize > 0 && vocabularySize > 0 && maximumPositions > 0)
        require(normEps.isFinite() && normEps > 0f && ropeTheta.isFinite() && ropeTheta > 0)
        require(layerTypes.isNotEmpty() && layerTypes.all { it in setOf("conv", "full_attention") })
        require(bosTokenId == null || bosTokenId in 0 until vocabularySize)
        require(eosTokenIds.all { it in 0 until vocabularySize })
    }

    /** The declared model limit is not an evaluation claim; default runtime allocation stays bounded. */
    fun toDimensions(contextLimit: Int = minOf(2048, maximumPositions)): ModelDimensions {
        require(contextLimit in 1..maximumPositions) { "Context limit must be within the configured maximum" }
        return ModelDimensions(hiddenSize, vocabularySize, attentionHeads, kvHeads,
            hiddenSize / attentionHeads, effectiveIntermediateSize, kernelSize, normEps,
            ropeTheta, layerTypes.toList(), contextLimit)
    }

    companion object {
        fun parse(text: String): Lfm2Config {
            val o = Json.parseToJsonElement(text).jsonObject
            fun int(name: String): Int = requireNotNull(o[name]?.jsonPrimitive?.intOrNull) { "Missing or invalid integer $name" }
            fun optionalBoolean(name: String): Boolean? = o[name]?.let {
                requireNotNull(it.jsonPrimitive.booleanOrNull) { "Invalid boolean $name" }
            }
            fun matchingInt(name: String, expected: Int) {
                if (name in o) require(int(name) == expected) { "Unsupported inconsistent $name" }
            }
            require(o["model_type"]?.jsonPrimitive?.content == "lfm2") { "Unsupported model_type; expected lfm2" }
            require(o["architectures"]?.jsonArray?.map { it.jsonPrimitive.content } == listOf("Lfm2ForCausalLM")) {
                "Unsupported architecture; expected dense Lfm2ForCausalLM"
            }
            require(optionalBoolean("conv_bias") == false) { "Only conv_bias=false is supported" }
            require(optionalBoolean("attention_bias") != true && optionalBoolean("mlp_bias") != true) { "Projection biases are unsupported" }
            require(optionalBoolean("block_use_swiglu") == true) { "Only SwiGLU feed-forward blocks are supported" }
            require(optionalBoolean("use_pos_enc") != false) { "This implementation requires rotary position encoding" }
            if ("hidden_act" in o) require(o.getValue("hidden_act").jsonPrimitive.content == "silu") { "Only silu activation is supported" }
            val tied = listOfNotNull(optionalBoolean("tie_embedding"), optionalBoolean("tie_word_embeddings"))
            require(tied.isNotEmpty() && tied.all { it }) { "Only tied embedding/output weights are supported" }
            require(o["rope_scaling"] == null || o["rope_scaling"] == JsonNull) { "Scaled RoPE is unsupported" }

            val hidden = int("hidden_size")
            val heads = int("num_attention_heads")
            val kv = int("num_key_value_heads")
            require(hidden > 0 && heads > 0 && kv > 0 && hidden % heads == 0 && heads % kv == 0) { "Invalid attention geometry" }
            matchingInt("head_dim", hidden / heads)
            matchingInt("block_dim", hidden)
            matchingInt("conv_dim", hidden)
            matchingInt("num_heads", heads)
            val layers = requireNotNull(o["layer_types"]) { "Missing layer_types" }.jsonArray.map { it.jsonPrimitive.content }
            require(layers.size == int("num_hidden_layers")) { "Layer count mismatch" }

            val raw = if ("block_ff_dim" in o) int("block_ff_dim") else int("intermediate_size")
            val adjust = optionalBoolean("block_auto_adjust_ff_dim") ?: false
            val multiple = int("block_multiple_of")
            val multiplierValue = o["block_ffn_dim_multiplier"] ?: JsonPrimitive(1.0)
            val multiplier = multiplierValue.jsonPrimitive.doubleOrNull
            require(multiplierValue == JsonNull || multiplier != null) { "Invalid FFN multiplier" }
            require(raw > 0 && multiple > 0 && (multiplier == null || multiplier.isFinite() && multiplier > 0)) { "Invalid FFN dimensions" }
            // Preserve reference order, including the null-multiplier branch without rounding.
            val adjusted = if (adjust) 2L * raw / 3L else raw.toLong()
            val scaled = if (adjust && multiplier != null) {
                val value = multiplier * adjusted
                require(value.isFinite() && value < Int.MAX_VALUE.toDouble()) { "FFN dimension out of range" }
                value.toLong()
            } else adjusted
            val effective = if (adjust && multiplier != null) multiple * ((scaled + multiple - 1L) / multiple) else scaled
            require(effective in 1..Int.MAX_VALUE.toLong()) { "FFN dimension out of range" }

            val epsilon = requireNotNull((o["norm_eps"] ?: o["block_norm_eps"])?.jsonPrimitive?.floatOrNull) { "Missing or invalid norm epsilon" }
            o["block_norm_eps"]?.let { require(it.jsonPrimitive.floatOrNull == epsilon) { "Distinct block/final norm epsilons are unsupported" } }
            val rope = requireNotNull(o["rope_parameters"]?.jsonObject) { "Explicit rope_parameters are required" }
            require(rope["rope_type"]?.jsonPrimitive?.content == "default") { "Only default rope_parameters are supported" }
            val theta = requireNotNull(rope["rope_theta"]?.jsonPrimitive?.doubleOrNull) { "Missing rope_theta" }
            o["rope_theta"]?.let { require(it.jsonPrimitive.doubleOrNull == theta) { "Inconsistent rope_theta" } }
            val vocabulary = int("vocab_size")
            val bos = o["bos_token_id"]?.takeUnless { it == JsonNull }?.jsonPrimitive?.let {
                requireNotNull(it.intOrNull) { "Invalid bos_token_id" }
            }
            val eos = when (val value = o["eos_token_id"]) {
                null, JsonNull -> emptySet()
                is JsonArray -> value.map { requireNotNull(it.jsonPrimitive.intOrNull) { "Invalid eos_token_id" } }.toSet()
                else -> setOf(requireNotNull(value.jsonPrimitive.intOrNull) { "Invalid eos_token_id" })
            }
            return Lfm2Config(hidden, int("conv_L_cache"), layers, heads, kv, effective.toInt(),
                vocabulary, epsilon, theta, int("max_position_embeddings"), bos, eos)
        }
    }
}
