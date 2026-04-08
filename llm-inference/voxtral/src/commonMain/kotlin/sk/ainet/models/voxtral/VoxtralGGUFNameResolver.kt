package sk.ainet.models.voxtral

import sk.ainet.io.weights.WeightNameResolver

/**
 * Resolves network DSL module paths to GGUF tensor names for Voxtral.
 *
 * Handles both the backbone (standard LLaMA naming) and the acoustic model
 * (prefixed with `acoustic.`). The backbone uses the same DSL module structure
 * as LLaMA, so the resolution logic is identical but extended for the acoustic
 * model's `acoustic.blk.N.*` modules.
 */
public class VoxtralGGUFNameResolver : WeightNameResolver {

    override fun resolve(modulePath: String, paramName: String): String? {
        val pathParts = modulePath.split("/").drop(1) // drop "MLP"

        // Check if this is an acoustic model module
        val isAcoustic = pathParts.any { it.startsWith("acoustic.") }

        val blockPrefix = if (isAcoustic) {
            pathParts.firstOrNull { it.startsWith("acoustic.blk.") }
        } else {
            pathParts.firstOrNull { it.startsWith("blk.") }
        }

        val moduleName = pathParts.lastOrNull() ?: return null

        return when {
            // Backbone embedding
            moduleName.contains("embd") || moduleName.contains("Embedding") ->
                "token_embd.weight"

            // Attention projections
            paramName.contains("q_proj.weight") ->
                if (blockPrefix != null) "$blockPrefix.attn_q.weight" else null
            paramName.contains("k_proj.weight") ->
                if (blockPrefix != null) "$blockPrefix.attn_k.weight" else null
            paramName.contains("v_proj.weight") ->
                if (blockPrefix != null) "$blockPrefix.attn_v.weight" else null
            paramName.contains("o_proj.weight") ->
                if (blockPrefix != null) "$blockPrefix.attn_output.weight" else null

            // QK norm (if used)
            paramName.contains("q_norm") ->
                if (blockPrefix != null) "$blockPrefix.attn_q_norm.weight" else null
            paramName.contains("k_norm") ->
                if (blockPrefix != null) "$blockPrefix.attn_k_norm.weight" else null

            // Attention norm
            moduleName == "attn_norm" || paramName.contains("attn_norm") ->
                if (blockPrefix != null) "$blockPrefix.attn_norm.weight" else null

            // FFN norm
            moduleName == "ffn_norm" || paramName.contains("ffn_norm") ->
                if (blockPrefix != null) "$blockPrefix.ffn_norm.weight" else null

            // FFN projections
            paramName.contains("gate_proj.weight") ->
                if (blockPrefix != null) "$blockPrefix.ffn_gate.weight" else null
            paramName.contains("up_proj.weight") ->
                if (blockPrefix != null) "$blockPrefix.ffn_up.weight" else null
            paramName.contains("down_proj.weight") ->
                if (blockPrefix != null) "$blockPrefix.ffn_down.weight" else null

            // Acoustic input/output projections
            moduleName == "acoustic.input_proj" && paramName.endsWith(".weight") ->
                "acoustic.input_proj.weight"
            moduleName == "acoustic.input_proj" && paramName.endsWith(".bias") ->
                "acoustic.input_proj.bias"
            moduleName == "acoustic.output_proj" && paramName.endsWith(".weight") ->
                "acoustic.output_proj.weight"
            moduleName == "acoustic.output_proj" && paramName.endsWith(".bias") ->
                "acoustic.output_proj.bias"

            // Output norm (backbone or acoustic)
            moduleName == "acoustic.output_norm" || paramName.contains("acoustic.output_norm") ->
                "acoustic.output_norm.weight"
            moduleName == "output_norm" || paramName.contains("output_norm") ->
                "output_norm.weight"

            // Output projection (backbone only)
            moduleName == "output" && paramName.endsWith(".weight") ->
                "output.weight"
            moduleName == "output" && paramName.endsWith(".bias") ->
                "output.bias"

            // FFN by module name
            moduleName == "ffn_up" && paramName.endsWith(".weight") ->
                if (blockPrefix != null) "$blockPrefix.ffn_up.weight" else null
            moduleName == "ffn_down" && paramName.endsWith(".weight") ->
                if (blockPrefix != null) "$blockPrefix.ffn_down.weight" else null

            else -> null
        }
    }
}
