package sk.ainet.apps.kllama

// No-op: the Android ops factory discovers the JNI native-cpu provider via
// ServiceLoader on its own (PlatformCpuOpsFactory.android.kt, engine #920 /
// transformers #285/#286).
internal actual fun installPlatformNativeKernels() {
}
