package sk.ainet.models.bert

/**
 * Configuration for a BERT model architecture.
 */
public data class BertModelConfig(
    val vocabSize: Int,
    val hiddenSize: Int,
    val numHiddenLayers: Int,
    val numAttentionHeads: Int,
    val intermediateSize: Int,
    val maxPositionEmbeddings: Int = 512,
    val typeVocabSize: Int = 2,
    val layerNormEps: Double = 1e-12,
    /** If non-null, a dense projection applied after pooling (sentence-transformers). */
    val projectionDim: Int? = null
)

/** Default config for MongoDB/mdbr-leaf-ir (23M-param BERT embedding model). */
public val MDBR_LEAF_IR_CONFIG: BertModelConfig = BertModelConfig(
    vocabSize = 30522,
    hiddenSize = 384,
    numHiddenLayers = 6,
    numAttentionHeads = 12,
    intermediateSize = 1536,
    maxPositionEmbeddings = 512,
    typeVocabSize = 2,
    layerNormEps = 1e-12,
    projectionDim = 768
)

/**
 * Parses a HuggingFace BERT `config.json` (and the optional sentence-transformers
 * `2_Dense/config.json` projection head) into a [BertModelConfig].
 *
 * Deliberately regex-based: the values are flat scalars, and this module stays
 * free of serialization dependencies across its multiplatform targets.
 */
public object BertConfigParser {

    /**
     * @param configJson content of the model's `config.json`
     * @param denseConfigJson content of `2_Dense/config.json` when the snapshot
     *   ships a dense projection head; its `out_features` becomes
     *   [BertModelConfig.projectionDim]
     */
    public fun parse(configJson: String, denseConfigJson: String? = null): BertModelConfig {
        fun extractInt(source: String, key: String, default: Int): Int =
            Regex("\"$key\"\\s*:\\s*(\\d+)").find(source)?.groupValues?.get(1)?.toIntOrNull() ?: default

        fun extractDouble(source: String, key: String, default: Double): Double =
            Regex("\"$key\"\\s*:\\s*([\\d.eE\\-+]+)").find(source)?.groupValues?.get(1)?.toDoubleOrNull() ?: default

        val projectionDim = denseConfigJson
            ?.let { extractInt(it, "out_features", 0) }
            ?.takeIf { it > 0 }

        return BertModelConfig(
            vocabSize = extractInt(configJson, "vocab_size", 30522),
            hiddenSize = extractInt(configJson, "hidden_size", 768),
            numHiddenLayers = extractInt(configJson, "num_hidden_layers", 12),
            numAttentionHeads = extractInt(configJson, "num_attention_heads", 12),
            intermediateSize = extractInt(configJson, "intermediate_size", 3072),
            maxPositionEmbeddings = extractInt(configJson, "max_position_embeddings", 512),
            typeVocabSize = extractInt(configJson, "type_vocab_size", 2),
            layerNormEps = extractDouble(configJson, "layer_norm_eps", 1e-12),
            projectionDim = projectionDim,
        )
    }
}
