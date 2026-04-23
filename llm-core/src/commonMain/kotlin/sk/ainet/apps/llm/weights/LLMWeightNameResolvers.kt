package sk.ainet.apps.llm.weights

import sk.ainet.io.weights.WeightNameResolver

/**
 * Resolves network DSL module paths to GGUF tensor names for LLaMA-family models.
 */
public class LlamaGGUFNameResolver : WeightNameResolver {

    override fun resolve(modulePath: String, paramName: String): String? {
        val pathParts = modulePath.split("/").drop(1) // drop "MLP"
        val blockPrefix = pathParts.firstOrNull { it.startsWith("blk.") }
        val moduleName = pathParts.lastOrNull() ?: return null

        return when {
            moduleName.contains("embd") || moduleName.contains("Embedding") ->
                "token_embd.weight"

            paramName.contains("q_proj.weight") ->
                if (blockPrefix != null) "$blockPrefix.attn_q.weight" else null
            paramName.contains("k_proj.weight") ->
                if (blockPrefix != null) "$blockPrefix.attn_k.weight" else null
            paramName.contains("v_proj.weight") ->
                if (blockPrefix != null) "$blockPrefix.attn_v.weight" else null
            paramName.contains("o_proj.weight") ->
                if (blockPrefix != null) "$blockPrefix.attn_output.weight" else null

            paramName.contains("q_norm") ->
                if (blockPrefix != null) "$blockPrefix.attn_q_norm.weight" else null
            paramName.contains("k_norm") ->
                if (blockPrefix != null) "$blockPrefix.attn_k_norm.weight" else null

            // Post-norms must match before the pre-norms below, otherwise the
            // substring check on "attn_norm" / "ffn_norm" would over-match and
            // claim the post variants too.
            moduleName == "post_attention_norm" || paramName.contains("post_attention_norm") ->
                if (blockPrefix != null) "$blockPrefix.post_attention_norm.weight" else null

            moduleName == "post_ffw_norm" || paramName.contains("post_ffw_norm") ->
                if (blockPrefix != null) "$blockPrefix.post_ffw_norm.weight" else null

            moduleName == "layer_output_scale" || paramName.contains("layer_output_scale") ->
                if (blockPrefix != null) "$blockPrefix.layer_output_scale.weight" else null

            // Per-Layer-Embedding top-level tensors. Gemma 4 uses these names
            // at the model root, not per-block — so they have no blockPrefix
            // and land in the general "$paramName contains X" checks below.
            paramName.contains("per_layer") && paramName.endsWith(".embed_tokens.weight") ->
                "per_layer_token_embd.weight"
            paramName.contains("per_layer") && paramName.endsWith(".model_proj.weight") ->
                "per_layer_model_proj.weight"
            paramName.contains("per_layer") && paramName.endsWith(".projection_norm.weight") ->
                "per_layer_proj_norm.weight"

            // Per-Layer-Embedding block-level hook tensors. Scoped via the
            // `per_layer_input` module-path segment so the `proj.weight`
            // match below doesn't collide with `o_proj.weight` in attention.
            paramName.contains("per_layer_input") && paramName.endsWith(".inp_gate.weight") ->
                if (blockPrefix != null) "$blockPrefix.inp_gate.weight" else null
            paramName.contains("per_layer_input") && paramName.endsWith(".proj.weight") ->
                if (blockPrefix != null) "$blockPrefix.proj.weight" else null
            paramName.contains("per_layer_input") && paramName.contains("post_norm") ->
                if (blockPrefix != null) "$blockPrefix.post_norm.weight" else null

            moduleName == "attn_norm" || paramName.contains("attn_norm") ->
                if (blockPrefix != null) "$blockPrefix.attn_norm.weight" else null

            moduleName == "ffn_norm" || paramName.contains("ffn_norm") ->
                if (blockPrefix != null) "$blockPrefix.ffn_norm.weight" else null

            paramName.contains("gate_proj.weight") ->
                if (blockPrefix != null) "$blockPrefix.ffn_gate.weight" else null
            paramName.contains("up_proj.weight") ->
                if (blockPrefix != null) "$blockPrefix.ffn_up.weight" else null
            paramName.contains("down_proj.weight") ->
                if (blockPrefix != null) "$blockPrefix.ffn_down.weight" else null

            paramName.contains("alpha_p") ->
                if (blockPrefix != null) "$blockPrefix.mlp.act_fn.alpha_p" else null
            paramName.contains("alpha_n") ->
                if (blockPrefix != null) "$blockPrefix.mlp.act_fn.alpha_n" else null
            paramName.contains(".beta") ->
                if (blockPrefix != null) "$blockPrefix.mlp.act_fn.beta" else null
            paramName.contains(".eps") ->
                if (blockPrefix != null) "$blockPrefix.mlp.act_fn.eps" else null

            moduleName == "output_norm" || paramName.contains("output_norm") ->
                "output_norm.weight"

            moduleName == "output" && paramName.endsWith(".weight") ->
                "output.weight"
            moduleName == "output" && paramName.endsWith(".bias") ->
                "output.bias"

            moduleName == "ffn_up" && paramName.endsWith(".weight") ->
                if (blockPrefix != null) "$blockPrefix.ffn_up.weight" else null
            moduleName == "ffn_down" && paramName.endsWith(".weight") ->
                if (blockPrefix != null) "$blockPrefix.ffn_down.weight" else null

            else -> null
        }
    }
}

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
        val layerNum = layerPart?.removePrefix("encoder.layer.")?.toIntOrNull()
        val layerPrefix = if (layerNum != null) "bert.encoder.layer.$layerNum" else null
        val inEmbeddings = pathParts.any { it == "embeddings" }

        return when {
            moduleName == "word_embeddings" && inEmbeddings ->
                "bert.embeddings.word_embeddings.weight"

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
