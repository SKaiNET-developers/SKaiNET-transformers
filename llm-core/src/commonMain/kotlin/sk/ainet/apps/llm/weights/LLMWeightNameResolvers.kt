package sk.ainet.apps.llm.weights

import sk.ainet.io.weights.WeightNameResolver

/**
 * Resolves network DSL module paths to SafeTensors (HuggingFace) tensor names for LLaMA.
 */
public class LlamaSafeTensorsNameResolver : WeightNameResolver {

    override fun resolve(modulePath: String, paramName: String): String? {
        val pathParts = modulePath.split("/").drop(1)
        val blockPrefix = pathParts.firstOrNull { it.startsWith("blk.") }
        val moduleName = pathParts.lastOrNull() ?: return null
        val layerNum = blockPrefix?.removePrefix("blk.")?.toIntOrNull()

        return when {
            moduleName.contains("embd") || moduleName.contains("Embedding") ->
                "model.embed_tokens.weight"

            paramName.contains("q_proj.weight") && layerNum != null ->
                "model.layers.$layerNum.self_attn.q_proj.weight"
            paramName.contains("k_proj.weight") && layerNum != null ->
                "model.layers.$layerNum.self_attn.k_proj.weight"
            paramName.contains("v_proj.weight") && layerNum != null ->
                "model.layers.$layerNum.self_attn.v_proj.weight"
            paramName.contains("o_proj.weight") && layerNum != null ->
                "model.layers.$layerNum.self_attn.o_proj.weight"

            (moduleName == "attn_norm" || paramName.contains("attn_norm")) && layerNum != null ->
                "model.layers.$layerNum.input_layernorm.weight"

            (moduleName == "ffn_norm" || paramName.contains("ffn_norm")) && layerNum != null ->
                "model.layers.$layerNum.post_attention_layernorm.weight"

            paramName.contains("gate_proj.weight") && layerNum != null ->
                "model.layers.$layerNum.mlp.gate_proj.weight"
            paramName.contains("up_proj.weight") && layerNum != null ->
                "model.layers.$layerNum.mlp.up_proj.weight"
            paramName.contains("down_proj.weight") && layerNum != null ->
                "model.layers.$layerNum.mlp.down_proj.weight"

            moduleName == "output_norm" || paramName.contains("output_norm") ->
                "model.norm.weight"

            moduleName == "output" && paramName.endsWith(".weight") ->
                "lm_head.weight"

            else -> null
        }
    }
}

/**
 * Resolves network DSL module paths to HuggingFace BERT tensor names.
 */
public class BertSafeTensorsNameResolver : WeightNameResolver {

    override fun resolve(modulePath: String, paramName: String): String? {
        val pathParts = modulePath.split("/").drop(1)
        val moduleName = pathParts.lastOrNull() ?: return null
        val layerPart = pathParts.firstOrNull { it.startsWith("encoder.layer.") }
        // Layer blocks may carry a sub-block suffix ("encoder.layer.3.attn") —
        // the layer number is the first segment after the prefix.
        val layerNum = layerPart?.removePrefix("encoder.layer.")?.substringBefore('.')?.toIntOrNull()
        val layerPrefix = if (layerNum != null) "bert.encoder.layer.$layerNum" else null
        val inEmbeddings = pathParts.any { it == "embeddings" }

        return when {
            moduleName == "word_embeddings" && inEmbeddings ->
                "bert.embeddings.word_embeddings.weight"

            // BertEmbeddings' own additive tables: the module path ends at
            // "embeddings"; the param name carries the table identity.
            paramName.endsWith("position_embeddings.weight") && inEmbeddings ->
                "bert.embeddings.position_embeddings.weight"
            paramName.endsWith("token_type_embeddings.weight") && inEmbeddings ->
                "bert.embeddings.token_type_embeddings.weight"

            moduleName == "LayerNorm" && inEmbeddings && paramName.endsWith(".weight") ->
                "bert.embeddings.LayerNorm.weight"
            moduleName == "LayerNorm" && inEmbeddings && paramName.endsWith(".bias") ->
                "bert.embeddings.LayerNorm.bias"

            paramName.contains("q_proj.weight") && layerPrefix != null ->
                "$layerPrefix.attention.self.query.weight"
            paramName.contains("q_proj.bias") && layerPrefix != null ->
                "$layerPrefix.attention.self.query.bias"
            paramName.contains("k_proj.weight") && layerPrefix != null ->
                "$layerPrefix.attention.self.key.weight"
            paramName.contains("k_proj.bias") && layerPrefix != null ->
                "$layerPrefix.attention.self.key.bias"
            paramName.contains("v_proj.weight") && layerPrefix != null ->
                "$layerPrefix.attention.self.value.weight"
            paramName.contains("v_proj.bias") && layerPrefix != null ->
                "$layerPrefix.attention.self.value.bias"

            paramName.contains("o_proj.weight") && layerPrefix != null ->
                "$layerPrefix.attention.output.dense.weight"
            paramName.contains("o_proj.bias") && layerPrefix != null ->
                "$layerPrefix.attention.output.dense.bias"

            moduleName == "attn_ln" && paramName.endsWith(".weight") && layerPrefix != null ->
                "$layerPrefix.attention.output.LayerNorm.weight"
            moduleName == "attn_ln" && paramName.endsWith(".bias") && layerPrefix != null ->
                "$layerPrefix.attention.output.LayerNorm.bias"

            moduleName == "intermediate" && paramName.endsWith(".weight") && layerPrefix != null ->
                "$layerPrefix.intermediate.dense.weight"
            moduleName == "intermediate" && paramName.endsWith(".bias") && layerPrefix != null ->
                "$layerPrefix.intermediate.dense.bias"

            moduleName == "output" && paramName.endsWith(".weight") && layerPrefix != null ->
                "$layerPrefix.output.dense.weight"
            moduleName == "output" && paramName.endsWith(".bias") && layerPrefix != null ->
                "$layerPrefix.output.dense.bias"

            moduleName == "output_ln" && paramName.endsWith(".weight") && layerPrefix != null ->
                "$layerPrefix.output.LayerNorm.weight"
            moduleName == "output_ln" && paramName.endsWith(".bias") && layerPrefix != null ->
                "$layerPrefix.output.LayerNorm.bias"

            else -> null
        }
    }
}
