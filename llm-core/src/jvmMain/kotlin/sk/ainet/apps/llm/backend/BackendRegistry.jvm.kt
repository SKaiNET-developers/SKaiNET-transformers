package sk.ainet.apps.llm.backend

import java.util.ServiceLoader

/**
 * JVM implementation using [ServiceLoader] for automatic classpath discovery.
 *
 * Each backend JAR ships a `META-INF/services/sk.ainet.apps.llm.backend.BackendProvider`
 * file listing its implementation class. Adding a backend JAR to the classpath
 * makes it automatically discoverable — no code changes required.
 */
public actual object BackendRegistry {
    private val discovered: MutableList<BackendProvider> = mutableListOf()
    private var loaded = false

    private fun ensureLoaded() {
        if (!loaded) {
            loaded = true
            ServiceLoader.load(BackendProvider::class.java).forEach { discovered.add(it) }
            discovered.sortByDescending { it.priority }
        }
    }

    public actual fun providers(): List<BackendProvider> {
        ensureLoaded()
        return discovered.toList()
    }

    public actual fun register(provider: BackendProvider) {
        ensureLoaded()
        discovered.add(provider)
        discovered.sortByDescending { it.priority }
    }
}
