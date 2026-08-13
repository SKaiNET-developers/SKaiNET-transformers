package sk.ainet.apps.kllama

import sk.ainet.exec.kernel.installNativeKernels

// skainet-backend-native-cpu publishes macosArm64 as of SKaiNET#959 (runtime
// FEAT_DotProd dispatch, one archive A12 through M-series) — install the
// native-cinterop packed-quant kernel provider the same way linuxMain does,
// since Kotlin/Native has no ServiceLoader. macosArm64 also gets Accelerate
// for dense FP32 separately (registered elsewhere), unaffected by this call.
internal actual fun installPlatformNativeKernels() {
    installNativeKernels()
}
