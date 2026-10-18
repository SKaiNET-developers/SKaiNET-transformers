@file:Suppress("DEPRECATION")

package sk.ainet.apps.llm

import sk.ainet.apps.llm.families.TransformersFamilies

/**
 * Registry of known model architectures and their capabilities.
 *
 * Since 0.58 (SKEEP-006) the source of truth is [TransformersFamilies.registry]: an immutable
 * `sk.ainet.model.ModelFamilyRegistry` of family *descriptors*. This object maps its answers back
 * onto the [ModelFamily] enum for callers that still switch on it; both are deprecated and go
 * in 0.60.
 *
 * ```kotlin
 * val family = TransformersFamilies.forArchitecture("qwen3")   // QwenFamily, or null
 * val id = TransformersFamilies.registry.identify(view)       // ModelIdentity(family = "qwen", variant = "qwen3")
 * ```
 */
@Deprecated("Use TransformersFamilies.registry / ModelIdentity (SKEEP-006); removed in 0.60.", ReplaceWith("TransformersFamilies", "sk.ainet.apps.llm.families.TransformersFamilies"))
public object ModelRegistry {

    /**
     * Detect the model family from a GGUF architecture string.
     *
     * @param architecture The `general.architecture` field from GGUF metadata.
     * @return The detected [ModelFamily], or [ModelFamily.UNKNOWN] if not recognized.
     */
    public fun detect(architecture: String): ModelFamily {
        val arch = architecture.lowercase()
        val family = TransformersFamilies.forArchitecture(arch)?.family ?: when {
            // prefix rules the 0.57 switch accepted for variants no descriptor lists yet
            arch.startsWith("qwen") -> "qwen"
            arch.startsWith("gemma") -> "gemma"
            arch.startsWith("bitnet") -> "bitnet"
            else -> null
        }
        return ModelFamily.entries.firstOrNull { it.id == family } ?: ModelFamily.UNKNOWN
    }

    /**
     * Detect the model family from GGUF metadata fields.
     *
     * @param architecture The `general.architecture` field.
     * @param chatTemplate Optional `tokenizer.chat_template` field for disambiguation.
     * @return The detected [ModelFamily].
     */
    public fun detect(architecture: String, chatTemplate: String?): ModelFamily {
        return detect(architecture)
    }
}

/**
 * Describes a model family and its capabilities.
 *
 * Deprecated view (SKEEP-006, FE-I-01): a family is a `String` on `ModelIdentity` and a
 * descriptor object in [TransformersFamilies], so new families need no enum edit. Removed in 0.60.
 *
 * @property id Unique identifier for the family.
 * @property displayName Human-readable name.
 * @property supportsToolCalling Whether the family supports tool calling via chat templates.
 * @property chatTemplateFamily The chat template family name for [ToolCallingSupportResolver].
 */
@Deprecated("Families are descriptors: TransformersFamilies.forFamily(identity.family) (SKEEP-006); removed in 0.60.")
public enum class ModelFamily(
    public val id: String,
    public val displayName: String,
    public val supportsToolCalling: Boolean,
    public val chatTemplateFamily: String?
) {
    LLAMA("llama", "LLaMA / Mistral", true, "llama3"),
    QWEN("qwen", "Qwen", true, "qwen"),
    GEMMA("gemma", "Gemma", true, "gemma"),
    APERTUS("apertus", "Apertus", true, "apertus"),
    BITNET("bitnet", "BitNet b1.58", false, "bitnet"),
    BERT("bert", "BERT", false, null),
    VOXTRAL("voxtral", "Voxtral TTS", false, null),
    UNKNOWN("unknown", "Unknown", false, null);

    /** GGUF architecture strings that map to this family. */
    public val architectures: Set<String>
        get() = when (this) {
            LLAMA -> setOf("llama", "mistral")
            QWEN -> setOf("qwen2", "qwen3", "qwen35")
            GEMMA -> setOf("gemma", "gemma2", "gemma3", "gemma3n", "gemma4")
            APERTUS -> setOf("apertus")
            // "bitnet" is BitNet.cpp's llama.cpp arch id; "bitnet-25" its 2B4T-era
            // variant; "bitnet-b1.58" appears in community conversions.
            BITNET -> setOf("bitnet", "bitnet-25", "bitnet-b1.58")
            BERT -> setOf("bert")
            VOXTRAL -> setOf("voxtral")
            UNKNOWN -> emptySet()
        }
}
