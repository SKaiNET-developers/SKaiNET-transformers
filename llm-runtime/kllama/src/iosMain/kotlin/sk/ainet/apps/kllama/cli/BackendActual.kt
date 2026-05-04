package sk.ainet.apps.kllama.cli

import sk.ainet.apps.kllama.CpuBackendProvider
import sk.ainet.apps.llm.backend.BackendRegistry

internal actual fun registerPlatformBackends() {
    BackendRegistry.register(CpuBackendProvider())
}
