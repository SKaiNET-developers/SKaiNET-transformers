package sk.ainet.models.qwen

import sk.ainet.io.weights.LlamaGGUFNameResolver
import sk.ainet.io.weights.WeightNameResolver

/**
 * Resolves network DSL module paths to GGUF tensor names for the Qwen family (#352): the
 * Qwen2/Qwen2.5 attention **bias** rules matched first, the Llama-family standard set delegated
 * to the engine's [LlamaGGUFNameResolver].
 *
 * Qwen2/2.5 GGUFs ship real `blk.N.attn_{q,k,v}.bias` tensors (Qwen3 and LLaMA do not). The
 * engine resolver has no `.bias` rules, so those tensors loaded and never bound — the DSL's
 * zero-initialized bias params stood in and qwen2 logits were silently garbage. These rules
 * previously existed only in a since-deleted llm-core fork of the resolver that no qwen code
 * ever used (the #372 dedup surfaced that).
 */
public class QwenGGUFNameResolver : WeightNameResolver {

    private val llama = LlamaGGUFNameResolver()

    override fun resolve(modulePath: String, paramName: String): String? {
        val blockPrefix = modulePath.split("/").drop(1).firstOrNull { it.startsWith("blk.") }
        return when {
            paramName.contains("q_proj.bias") ->
                if (blockPrefix != null) "$blockPrefix.attn_q.bias" else null
            paramName.contains("k_proj.bias") ->
                if (blockPrefix != null) "$blockPrefix.attn_k.bias" else null
            paramName.contains("v_proj.bias") ->
                if (blockPrefix != null) "$blockPrefix.attn_v.bias" else null
            paramName.contains("o_proj.bias") ->
                if (blockPrefix != null) "$blockPrefix.attn_output.bias" else null
            else -> llama.resolve(modulePath, paramName)
        }
    }
}
