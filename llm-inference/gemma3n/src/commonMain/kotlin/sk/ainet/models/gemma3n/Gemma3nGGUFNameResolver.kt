package sk.ainet.models.gemma3n

import sk.ainet.io.weights.WeightNameResolver
import sk.ainet.models.gemma.GemmaGGUFNameResolver

/**
 * Resolves DSL module paths to GGUF tensor names for the Gemma 3n family: the 3n-specific
 * rules (AltUp per-layer + global tensors, Laurel) matched first, everything the gemma-4
 * lane already handles (sandwich norms, PLE names, llama-standard set) delegated to
 * [GemmaGGUFNameResolver].
 */
public class Gemma3nGGUFNameResolver : WeightNameResolver {

    private val gemma = GemmaGGUFNameResolver()

    override fun resolve(modulePath: String, paramName: String): String? {
        val blockPrefix = modulePath.split("/").drop(1).firstOrNull { it.startsWith("blk.") }

        // Per-layer AltUp + Laurel params are named after their GGUF tensors already —
        // "<module>.altup_router.weight" etc. — so the rule is: take the tensor-suffix
        // and prefix the block.
        for (suffix in BLOCK_SUFFIXES) {
            if (paramName.endsWith(".$suffix.weight") || paramName == "$suffix.weight") {
                return if (blockPrefix != null) "$blockPrefix.$suffix.weight" else null
            }
        }
        // Model-level AltUp stream projections (3D tensors, no block prefix).
        if (paramName.endsWith(".altup_proj.weight")) return "altup_proj.weight"
        if (paramName.endsWith(".altup_unembd_proj.weight")) return "altup_unembd_proj.weight"

        return gemma.resolve(modulePath, paramName)
    }

    private companion object {
        val BLOCK_SUFFIXES = listOf(
            "altup_router_norm", "altup_router", "altup_predict_coef", "altup_correct_coef",
            "altup_correct_scale", "laurel_l", "laurel_r", "laurel_post_norm",
        )
    }
}
