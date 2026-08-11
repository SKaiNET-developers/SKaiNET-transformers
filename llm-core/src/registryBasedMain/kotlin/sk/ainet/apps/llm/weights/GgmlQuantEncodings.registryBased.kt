package sk.ainet.apps.llm.weights

/**
 * Registry-based targets (Kotlin/Native, Android, JS/WASM): no ServiceLoader.
 * Kernel providers are registered explicitly by the platform backend factories
 * (e.g. the K/N CPU ops factory pins `ScalarKernelProvider`; Android pins the
 * JNI provider) before any converter runs — [hasPackedMatmulKernel] just reads
 * the registry as-is.
 */
internal actual fun ensurePlatformKernelProviders() {
    // No-op: KernelRegistry is populated by the backend factory on these targets.
}
