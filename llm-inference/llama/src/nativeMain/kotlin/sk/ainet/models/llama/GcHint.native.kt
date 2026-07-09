package sk.ainet.models.llama

@OptIn(kotlin.native.runtime.NativeRuntimeApi::class)
internal actual fun gcCollectHint() {
    kotlin.native.runtime.GC.collect()
}
