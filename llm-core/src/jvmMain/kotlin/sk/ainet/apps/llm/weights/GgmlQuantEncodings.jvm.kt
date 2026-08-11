package sk.ainet.apps.llm.weights

import sk.ainet.backend.api.kernel.KernelServiceLoader

/**
 * JVM: install every `META-INF/services`-declared
 * [sk.ainet.backend.api.kernel.KernelProvider] into the process registry.
 * Idempotent ([sk.ainet.backend.api.kernel.KernelRegistry.register] no-ops on
 * re-registration), and the same wiring the engine's `DefaultCpuOpsJvm`
 * performs lazily on first matmul — done eagerly here so the
 * [hasPackedMatmulKernel] gate answers correctly at weight-conversion time,
 * which runs before any matmul.
 */
internal actual fun ensurePlatformKernelProviders() {
    KernelServiceLoader.installAll()
}
