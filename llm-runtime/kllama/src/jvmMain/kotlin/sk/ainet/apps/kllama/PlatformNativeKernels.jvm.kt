package sk.ainet.apps.kllama

// No-op: the JVM ops factory discovers the FFM native-cpu provider via
// ServiceLoader on its own (KernelServiceLoaderFactories).
internal actual fun installPlatformNativeKernels() {
}
