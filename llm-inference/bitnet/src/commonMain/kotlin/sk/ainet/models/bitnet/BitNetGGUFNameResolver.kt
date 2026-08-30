package sk.ainet.models.bitnet

import sk.ainet.io.weights.LlamaGGUFNameResolver
import sk.ainet.io.weights.WeightNameResolver

/**
 * Resolves network DSL module paths to GGUF tensor names for BitNet b1.58 models.
 *
 * BitNet uses the Llama-family naming (`blk.N.attn_q.weight`, `blk.N.ffn_gate.weight`, …) plus
 * two per-layer sub-norm tensors of its own:
 *
 * - `attn.sub_norm.weight` (the [sk.ainet.lang.nn.transformer.MultiHeadAttention] `attnSubNorm`
 *   child) → `blk.N.attn_sub_norm.weight`
 * - `ffn.sub_norm.weight` (the [sk.ainet.lang.nn.transformer.BitNetFFN] sub-norm child)
 *   → `blk.N.ffn_sub_norm.weight`
 *
 * The sub-norm cases must match BEFORE delegating: the Llama resolver's substring checks on
 * `attn_norm` / `ffn_norm` don't collide with the dotted `attn.sub_norm` (dot vs underscore),
 * but keeping the specific cases first makes that independence explicit rather than accidental.
 */
public class BitNetGGUFNameResolver : WeightNameResolver {

    private val llama = LlamaGGUFNameResolver()

    override fun resolve(modulePath: String, paramName: String): String? {
        val blockPrefix = modulePath.split("/").drop(1).firstOrNull { it.startsWith("blk.") }
        return when {
            paramName.contains("attn.sub_norm.weight") ->
                if (blockPrefix != null) "$blockPrefix.attn_sub_norm.weight" else null
            paramName.contains("ffn.sub_norm.weight") ->
                if (blockPrefix != null) "$blockPrefix.ffn_sub_norm.weight" else null
            else -> llama.resolve(modulePath, paramName)
        }
    }
}
