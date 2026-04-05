package sk.ainet.apps.llm.backend

/**
 * Registry-based implementation for non-JVM targets (native, JS, Wasm).
 *
 * Platform source sets call [register] at startup for each available backend.
 */
public actual object BackendRegistry {
    private val registry: MutableList<BackendProvider> = mutableListOf()

    public actual fun providers(): List<BackendProvider> = registry.toList()

    public actual fun register(provider: BackendProvider) {
        registry.add(provider)
        registry.sortByDescending { it.priority }
    }
}
