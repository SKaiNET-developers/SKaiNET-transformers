package sk.ainet.apps.llm

/**
 * Registry of known model architectures and their capabilities.
 *
 * Maps GGUF architecture strings to [ModelFamily] descriptors.
 * Used for auto-detection: given GGUF metadata, determine which
 * network DSL definition, weight loader, and chat template to use.
 *
 * Usage:
 * ```kotlin
 * val family = ModelRegistry.detect("qwen3")  // returns ModelFamily.QWEN
 * val family = ModelRegistry.detect("llama")   // returns ModelFamily.LLAMA
 * ```
 */
public object ModelRegistry {

    /**
     * Detect the model family from a GGUF architecture string.
     *
     * @param architecture The `general.architecture` field from GGUF metadata.
     * @return The detected [ModelFamily], or [ModelFamily.UNKNOWN] if not recognized.
     */
    public fun detect(architecture: String): ModelFamily {
        val arch = architecture.lowercase()
        return when {
            arch == "llama" || arch == "mistral" -> ModelFamily.LLAMA
            arch.startsWith("qwen") -> ModelFamily.QWEN
            arch.startsWith("gemma") -> ModelFamily.GEMMA
            arch == "apertus" -> ModelFamily.APERTUS
            arch.startsWith("bitnet") -> ModelFamily.BITNET
            arch == "bert" -> ModelFamily.BERT
            arch == "voxtral" -> ModelFamily.VOXTRAL
            else -> ModelFamily.UNKNOWN
        }
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
 * @property id Unique identifier for the family.
 * @property displayName Human-readable name.
 * @property supportsToolCalling Whether the family supports tool calling via chat templates.
 * @property chatTemplateFamily The chat template family name for [ToolCallingSupportResolver].
 */
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
    BITNET("bitnet", "BitNet b1.58", false, null),
    BERT("bert", "BERT", false, null),
    VOXTRAL("voxtral", "Voxtral TTS", false, null),
    UNKNOWN("unknown", "Unknown", false, null);

    /** GGUF architecture strings that map to this family. */
    public val architectures: Set<String>
        get() = when (this) {
            LLAMA -> setOf("llama", "mistral")
            QWEN -> setOf("qwen2", "qwen3", "qwen35")
            GEMMA -> setOf("gemma", "gemma2", "gemma3", "gemma3n")
            APERTUS -> setOf("apertus")
            // "bitnet" is BitNet.cpp's llama.cpp arch id; "bitnet-25" its 2B4T-era
            // variant; "bitnet-b1.58" appears in community conversions.
            BITNET -> setOf("bitnet", "bitnet-25", "bitnet-b1.58")
            BERT -> setOf("bert")
            VOXTRAL -> setOf("voxtral")
            UNKNOWN -> emptySet()
        }
}
