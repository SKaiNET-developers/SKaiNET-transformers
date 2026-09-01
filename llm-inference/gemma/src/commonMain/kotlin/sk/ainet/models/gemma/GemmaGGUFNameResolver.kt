package sk.ainet.models.gemma

import sk.ainet.io.weights.LlamaGGUFNameResolver
import sk.ainet.io.weights.WeightNameResolver

/**
 * Resolves network DSL module paths to GGUF tensor names for the Gemma family
 * (transformers#372's resolver dedup): the Gemma-specific rules — sandwich post-norms,
 * `layer_output_scale`, and the Gemma-4 Per-Layer-Embedding tensors — matched first, the
 * Llama-family standard set delegated to the engine's [LlamaGGUFNameResolver].
 *
 * These rules previously lived in a full fork of the llama resolver in `llm-core`
 * (`sk.ainet.apps.llm.weights.LlamaGGUFNameResolver`), drifting from the engine copy every
 * family actually uses — the exact duplication #346 warns about.
 */
public class GemmaGGUFNameResolver : WeightNameResolver {

    private val llama = LlamaGGUFNameResolver()

    override fun resolve(modulePath: String, paramName: String): String? {
        val pathParts = modulePath.split("/").drop(1)
        val blockPrefix = pathParts.firstOrNull { it.startsWith("blk.") }
        val moduleName = pathParts.lastOrNull() ?: return null

        return when {
            // Sandwich post-norms — must match before the llama resolver's substring checks
            // on "attn_norm"/"ffn_norm" would over-match and claim the post variants.
            moduleName == "post_attention_norm" || paramName.contains("post_attention_norm") ->
                if (blockPrefix != null) "$blockPrefix.post_attention_norm.weight" else null
            moduleName == "post_ffw_norm" || paramName.contains("post_ffw_norm") ->
                if (blockPrefix != null) "$blockPrefix.post_ffw_norm.weight" else null

            moduleName == "layer_output_scale" || paramName.contains("layer_output_scale") ->
                if (blockPrefix != null) "$blockPrefix.layer_output_scale.weight" else null

            // Per-Layer-Embedding top-level tensors (Gemma 4): model-root names, no blockPrefix.
            paramName.contains("per_layer") && paramName.endsWith(".embed_tokens.weight") ->
                "per_layer_token_embd.weight"
            paramName.contains("per_layer") && paramName.endsWith(".model_proj.weight") ->
                "per_layer_model_proj.weight"
            paramName.contains("per_layer") && paramName.endsWith(".projection_norm.weight") ->
                "per_layer_proj_norm.weight"

            // Per-Layer-Embedding block-level hook tensors. Scoped via the `per_layer_input`
            // module-path segment so the `proj.weight` match doesn't collide with
            // `o_proj.weight` in attention.
            paramName.contains("per_layer_input") && paramName.endsWith(".inp_gate.weight") ->
                if (blockPrefix != null) "$blockPrefix.inp_gate.weight" else null
            paramName.contains("per_layer_input") && paramName.endsWith(".proj.weight") ->
                if (blockPrefix != null) "$blockPrefix.proj.weight" else null
            paramName.contains("per_layer_input") && paramName.contains("post_norm") ->
                if (blockPrefix != null) "$blockPrefix.post_norm.weight" else null

            else -> llama.resolve(modulePath, paramName)
        }
    }
}
