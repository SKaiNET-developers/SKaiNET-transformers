package sk.ainet.apps.kllama.cli

/**
 * Register platform-specific [BackendProvider]s with [BackendRegistry].
 * Called once at startup before backend selection.
 */
internal expect fun registerPlatformBackends()
