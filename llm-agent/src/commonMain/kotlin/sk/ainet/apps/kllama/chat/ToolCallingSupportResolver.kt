package sk.ainet.apps.kllama.chat

/**
 * Resolves the best [ToolCallingSupport] provider for a given model.
 *
 * Resolution order:
 * 1. **Explicit override** — if `explicitFamily` is set, select that provider directly.
 * 2. **Metadata auto-detection** — iterate registered providers and pick the first
 *    whose [ToolCallingSupport.supports] returns `true`.
 * 3. **No match** — returns `null` (caller can decide whether to use a fallback).
 */
public object ToolCallingSupportResolver {

    private val providers: List<ToolCallingSupport> = listOf(
        QwenToolCallingSupport(),
        GemmaToolCallingSupport(),
        Llama3ToolCallingSupport(),
        ChatMLToolCallingSupport()
    )

    /**
     * Resolve a [ToolCallingSupport] provider.
     *
     * @param metadata Model metadata used for auto-detection.
     * @param explicitFamily If non-null, selects the provider with this [ToolCallingSupport.family]
     *   name, bypassing metadata-based detection. Common values: "llama3", "chatml", "qwen", "gemma".
     * @return The matched provider, or `null` if no provider matches.
     */
    public fun resolve(
        metadata: ModelMetadata = ModelMetadata(),
        explicitFamily: String? = null
    ): ToolCallingSupport? {
        // Explicit override takes precedence
        if (explicitFamily != null) {
            val explicit = providers.firstOrNull {
                it.family.equals(explicitFamily, ignoreCase = true)
            }
            if (explicit != null) return explicit
            // Also support legacy aliases
            if (explicitFamily.equals("hermes", ignoreCase = true)) {
                return providers.firstOrNull { it.family == "chatml" }
            }
        }

        // Auto-detection from metadata
        return providers.firstOrNull { it.supports(metadata) }
    }

    /**
     * Like [resolve], but falls back to [GenericToolCallingSupport] instead of
     * returning `null`. Use this when the caller always needs a provider.
     */
    public fun resolveOrFallback(
        metadata: ModelMetadata = ModelMetadata(),
        explicitFamily: String? = null
    ): ToolCallingSupport {
        return resolve(metadata, explicitFamily) ?: GenericToolCallingSupport()
    }
}
