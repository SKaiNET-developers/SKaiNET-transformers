package sk.ainet.apps.llm.backend

/**
 * Registry for discovering and selecting compute backends at runtime.
 *
 * Platform implementations provide the discovery mechanism:
 * - JVM: [java.util.ServiceLoader] scans classpath for [BackendProvider] implementations
 * - Native: manual registration via [register]
 *
 * Usage:
 * ```kotlin
 * val provider = BackendRegistry.find("metal") ?: BackendRegistry.bestAvailable()
 * val ctx = provider.createContext()
 * ```
 */
public expect object BackendRegistry {
    /** All registered/discovered providers, sorted by priority descending. */
    public fun providers(): List<BackendProvider>

    /** Register a provider manually (primarily for native targets). */
    public fun register(provider: BackendProvider)
}

/** Find a provider by name (case-insensitive), or null if not found. */
public fun BackendRegistry.find(name: String): BackendProvider? =
    providers().firstOrNull { it.name.equals(name, ignoreCase = true) }

/** Auto-select the best available backend (highest priority that is available). */
public fun BackendRegistry.bestAvailable(): BackendProvider =
    providers().filter { it.isAvailable() }.maxByOrNull { it.priority }
        ?: error("No backends available")

/** List names of all available backends. */
public fun BackendRegistry.availableNames(): List<String> =
    providers().filter { it.isAvailable() }.map { it.name }
