package sk.ainet.performance.native

import sk.ainet.apps.kllama.GpuTensorBridge
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.types.DType

internal actual fun createMetalContext(): ExecutionContext? = null

internal actual fun createMlxContext(): ExecutionContext? = null

internal actual fun <T : DType> createGpuBridge(ctx: ExecutionContext): GpuTensorBridge<T>? = null

internal actual fun availableNativeBackends(): List<String> = listOf("CPU")
