package sk.ainet.apps.kllama.chat

/**
 * Resolves the best [ToolCallingSupport] provider for a given model.
 *
 * Resolution order:
 * 1. **Explicit override** — if `explicitFamily` is set, select that provider directly.
 * 2. **Metadata auto-detection** — iterate registered providers and pick the first
 *    whose [ToolCallingSupport.supports] returns `true`.
 * 3. **No match** — returns `null` (caller can decide whether to use a fallback).
 *
 * New model families can be registered at runtime via [register].
 */
public object ToolCallingSupportResolver {

    private val providers: MutableList<ToolCallingSupport> = mutableListOf(
        QwenToolCallingSupport(),
        // Gemma 4 must come BEFORE the generic Gemma provider because
        // GemmaToolCallingSupport.supports() matches any arch starting
        // with "gemma" and would otherwise claim Gemma 4 checkpoints
        // and hand out the Gemma 2/3 template.
        Gemma4ToolCallingSupport(),
        GemmaToolCallingSupport(),
        ApertusToolCallingSupport(),
        Llama3ToolCallingSupport(),
        ChatMLToolCallingSupport()
    )

    /**
     * Register an additional [ToolCallingSupport] provider.
     *
     * The new provider is prepended so it takes priority over built-in ones.
     * If a provider with the same [ToolCallingSupport.family] already exists,
     * the old one is replaced.
     */
    public fun register(provider: ToolCallingSupport) {
        providers.removeAll { it.family.equals(provider.family, ignoreCase = true) }
        providers.add(0, provider)
    }

    /** Return a snapshot of all currently registered provider family names. */
    public fun registeredFamilies(): List<String> = providers.map { it.family }

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

    /**
     * Resolve a provider and return a diagnostic [ResolutionResult] explaining
     * why that provider was selected.
     */
    public fun resolveWithDiagnostics(
        metadata: ModelMetadata = ModelMetadata(),
        explicitFamily: String? = null
    ): ResolutionResult {
        val provider = resolveOrFallback(metadata, explicitFamily)
        val reason = when {
            explicitFamily != null -> "explicit family override: $explicitFamily"
            provider is GenericToolCallingSupport -> "no native provider matched; using generic fallback"
            else -> "auto-detected from metadata (family=${metadata.family}, arch=${metadata.architecture})"
        }
        return ResolutionResult(provider, provider.toolCallingMode(metadata), reason)
    }
}

/**
 * Result of provider resolution with diagnostics explaining the selection.
 */
public data class ResolutionResult(
    val provider: ToolCallingSupport,
    val mode: ToolCallingMode,
    val reason: String
)
