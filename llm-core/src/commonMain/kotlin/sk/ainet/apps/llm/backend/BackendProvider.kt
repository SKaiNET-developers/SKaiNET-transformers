package sk.ainet.apps.llm.backend

import sk.ainet.context.ExecutionContext

/**
 * Service interface for dynamically-discovered compute backends.
 *
 * Each backend (CPU, Metal, MLX, CUDA, etc.) provides one implementation.
 * On JVM, providers are discovered via [java.util.ServiceLoader].
 * On native, providers register with [BackendRegistry] at startup.
 */
public interface BackendProvider {
    /** Unique backend identifier, e.g. "cpu", "metal", "mlx". */
    public val name: String

    /** Human-readable display name, e.g. "CPU (SIMD)", "Metal GPU". */
    public val displayName: String

    /** Priority for auto-selection (higher = preferred). CPU = 0, GPU = 100. */
    public val priority: Int

    /** Returns true if this backend can run on the current platform/hardware. */
    public fun isAvailable(): Boolean

    /** Create an [ExecutionContext] for this backend. */
    public fun createContext(): ExecutionContext
}
