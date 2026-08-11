package sk.ainet.apps.kllama

// The engine does not yet publish skainet-backend-native-cpu for
// iosArm64/iosSimulatorArm64 (SKaiNET#959; SKaiNET-transformers#298 tracks
// wiring it once it does) — no-op until then; packed-quant matmul stays
// scalar in the meantime.
internal actual fun installPlatformNativeKernels() {
}
