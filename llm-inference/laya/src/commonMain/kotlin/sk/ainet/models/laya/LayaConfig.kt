package sk.ainet.models.laya

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/** A checkpoint declares something this adapter does not implement (REQ-EXE-010). */
public class UnsupportedLayaCheckpointException(message: String) : IllegalArgumentException(message)

internal val layaJson: Json = Json { ignoreUnknownKeys = true; isLenient = false }

/** Option layouts the reference knows; every published checkpoint uses `sequential`. */
public enum class LayaOptionLayout(public val wire: String) { SEQUENTIAL("sequential"), PARALLEL("parallel") }

/**
 * `rl_agent_config.json`: budgets, temperatures and the encoder the checkpoint was trained on.
 * Nothing in here may be defaulted from another checkpoint (REQ-PRE-004).
 */
@Serializable
public data class LayaAgentConfig(
    val encoder: String,
    @SerialName("head_layers") val headLayers: Int = 2,
    @SerialName("max_len") val maxLen: Int = 512,
    @SerialName("head_max_len") val headMaxLen: Int = 192,
    val temperature: List<Double> = listOf(1.0, 1.0, 1.0),
    @SerialName("temperature_by_options") val temperatureByOptions: Map<String, Double> = emptyMap(),
    @SerialName("act_costs") val actCosts: Map<String, Double> = emptyMap(),
    @SerialName("option_layout") val optionLayoutName: String = "sequential",
    @SerialName("amp_dtype") val ampDtype: String? = null,
    @SerialName("model_name") val modelName: String? = null,
) {
    init {
        require(maxLen > 0 && headMaxLen > 0) { "max_len and head_max_len must be positive, got $maxLen / $headMaxLen" }
        require(temperature.size == 3) { "temperature must hold one value per question type, got ${temperature.size}" }
    }

    /** `n_act = len(act_costs) + 1` in the reference. */
    public val actionCount: Int get() = actCosts.size + 1

    public val optionLayout: LayaOptionLayout
        get() = LayaOptionLayout.entries.firstOrNull { it.wire == optionLayoutName }
            ?: throw UnsupportedLayaCheckpointException("option_layout must be one of sequential, parallel; got '$optionLayoutName'")

    /**
     * Rejects what the adapter cannot run. The parallel option layout (shared position ids plus
     * per-option attention masks) changes the encoder input and is not implemented (REQ-EXE-010).
     */
    public fun requireSupported(): LayaAgentConfig {
        if (optionLayout != LayaOptionLayout.SEQUENTIAL) {
            throw UnsupportedLayaCheckpointException("option_layout '${optionLayout.wire}' is not supported; only the sequential layout is")
        }
        return this
    }

    public companion object {
        public fun parse(json: String): LayaAgentConfig = layaJson.decodeFromString(serializer(), json)
    }
}

/** Attention kind of one encoder layer. */
public enum class LayaLayerKind(public val wire: String) { FULL("full_attention"), SLIDING("sliding_attention") }

@Serializable
public data class LayaRopeParameters(
    @SerialName("rope_theta") val ropeTheta: Double,
    @SerialName("rope_type") val ropeType: String = "default",
)

/**
 * `encoder/config.json` of a ModernBERT-family encoder (mmBERT for `multilingual`). Only the
 * keys execution depends on are read; the checkpoint's own file is authoritative, the public
 * base-encoder config is not consulted.
 */
@Serializable
public data class LayaEncoderConfig(
    @SerialName("model_type") val modelType: String,
    @SerialName("hidden_size") val hiddenSize: Int,
    @SerialName("num_attention_heads") val numAttentionHeads: Int,
    @SerialName("num_hidden_layers") val numHiddenLayers: Int,
    @SerialName("intermediate_size") val intermediateSize: Int,
    @SerialName("vocab_size") val vocabSize: Int,
    @SerialName("hidden_activation") val hiddenActivation: String = "gelu",
    @SerialName("norm_eps") val normEps: Double = 1e-5,
    @SerialName("norm_bias") val normBias: Boolean = false,
    @SerialName("attention_bias") val attentionBias: Boolean = false,
    @SerialName("mlp_bias") val mlpBias: Boolean = false,
    @SerialName("local_attention") val localAttention: Int = 128,
    @SerialName("global_attn_every_n_layers") val globalAttnEveryNLayers: Int = 3,
    @SerialName("layer_types") val layerTypeNames: List<String>? = null,
    @SerialName("rope_parameters") val ropeParameters: Map<String, LayaRopeParameters>? = null,
    @SerialName("global_rope_theta") val globalRopeTheta: Double? = null,
    @SerialName("local_rope_theta") val localRopeTheta: Double? = null,
    @SerialName("position_embedding_type") val positionEmbeddingType: String = "sans_pos",
    @SerialName("max_position_embeddings") val maxPositionEmbeddings: Int = 8192,
    @SerialName("pad_token_id") val padTokenId: Int = 0,
    @SerialName("sliding_window") val slidingWindow: Int? = null,
) {
    public val headDim: Int get() = hiddenSize / numAttentionHeads

    /** Keys each query may see on each side on a sliding layer (`local_attention / 2`, inclusive). */
    public val localHalfWindow: Int get() = localAttention / 2

    /** The attention kind of every layer, from `layer_types` or the `global_attn_every_n_layers` rule. */
    public val layerKinds: List<LayaLayerKind>
        get() = layerTypeNames?.map { name ->
            LayaLayerKind.entries.firstOrNull { it.wire == name }
                ?: throw UnsupportedLayaCheckpointException("unknown layer type '$name' in encoder config")
        } ?: List(numHiddenLayers) { i -> if (i % globalAttnEveryNLayers == 0) LayaLayerKind.FULL else LayaLayerKind.SLIDING }

    /** RoPE base for a layer kind: transformers 5 `rope_parameters`, else the 4.x per-kind thetas, else ModernBERT's defaults. */
    public fun ropeTheta(kind: LayaLayerKind): Double {
        ropeParameters?.get(kind.wire)?.let { return it.ropeTheta }
        return when (kind) {
            LayaLayerKind.FULL -> globalRopeTheta ?: 160000.0
            LayaLayerKind.SLIDING -> localRopeTheta ?: 10000.0
        }
    }

    /** Rejects encoder features outside the ModernBERT graph this adapter implements (REQ-EXE-010). */
    public fun requireSupported(): LayaEncoderConfig {
        if (modelType != "modernbert") throw UnsupportedLayaCheckpointException("encoder model_type '$modelType' is not supported; expected modernbert")
        if (hiddenActivation != "gelu") throw UnsupportedLayaCheckpointException("hidden_activation '$hiddenActivation' is not supported; expected gelu")
        if (positionEmbeddingType !in SUPPORTED_POSITION_EMBEDDING_TYPES) {
            throw UnsupportedLayaCheckpointException(
                "position_embedding_type '$positionEmbeddingType' is not supported; expected one of $SUPPORTED_POSITION_EMBEDDING_TYPES " +
                    "(ModernBERT positions are rotary whatever the field says; transformers 4 writes sans_pos, transformers 5 absolute)",
            )
        }
        if (hiddenSize % numAttentionHeads != 0) throw UnsupportedLayaCheckpointException("hidden_size $hiddenSize is not a multiple of num_attention_heads $numAttentionHeads")
        if (layerKinds.size != numHiddenLayers) throw UnsupportedLayaCheckpointException("layer_types lists ${layerKinds.size} layers, num_hidden_layers is $numHiddenLayers")
        return this
    }

    public companion object {
        /**
         * `position_embedding_type` values ModernBERT configs carry. The model ignores the field (its
         * positions are RoPE in every layer); the exporter version decides the spelling — `sans_pos`
         * in transformers 4 (`laya-multilingual`), `absolute` in transformers 5 (`laya-typed-decisions`).
         */
        public val SUPPORTED_POSITION_EMBEDDING_TYPES: Set<String> = setOf("sans_pos", "absolute")

        public fun parse(json: String): LayaEncoderConfig = layaJson.decodeFromString(serializer(), json)
    }
}

/**
 * The special-token strings of `tokenizer/tokenizer_config.json`. Laya reads `cls_token`,
 * `sep_token`, `mask_token` and `pad_token` from the tokenizer, never from the encoder config:
 * for `multilingual` the two disagree (`<bos>` is the classification token, `<eos>` the separator).
 */
public data class LayaTokenizerConfig(
    val clsToken: String,
    val sepToken: String,
    val maskToken: String,
    val padToken: String,
    val unkToken: String?,
) {
    public companion object {
        public fun parse(json: String): LayaTokenizerConfig {
            val root = layaJson.parseToJsonElement(json).jsonObject
            fun token(key: String): String? = root[key]?.let(::tokenContent)
            return LayaTokenizerConfig(
                clsToken = token("cls_token") ?: throw UnsupportedLayaCheckpointException("tokenizer_config.json has no cls_token"),
                sepToken = token("sep_token") ?: throw UnsupportedLayaCheckpointException("tokenizer_config.json has no sep_token"),
                maskToken = token("mask_token") ?: throw UnsupportedLayaCheckpointException("tokenizer_config.json has no mask_token"),
                padToken = token("pad_token") ?: throw UnsupportedLayaCheckpointException("tokenizer_config.json has no pad_token"),
                unkToken = token("unk_token"),
            )
        }

        /** Hugging Face writes a special token either as a string or as an `AddedToken` object with `content`. */
        private fun tokenContent(element: JsonElement): String? = when (element) {
            is JsonPrimitive -> if (element.isString) element.content else null
            is JsonObject -> element["content"]?.jsonPrimitive?.content
            else -> null
        }
    }
}
