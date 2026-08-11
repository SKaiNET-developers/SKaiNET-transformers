package sk.ainet.apps.kllama

// The engine does not yet publish skainet-backend-native-cpu for macosArm64
// (SKaiNET#959; SKaiNET-transformers#298 tracks wiring it once it does) —
// no-op until then. macosArm64 already gets Accelerate for dense FP32;
// packed-quant matmul stays scalar in the meantime.
internal actual fun installPlatformNativeKernels() {
}
