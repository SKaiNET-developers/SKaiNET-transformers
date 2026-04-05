package sk.ainet.apps.kllama.cli

import sk.ainet.models.llama.GraphAccelerator
import sk.ainet.apps.kllama.CpuBackendProvider
import sk.ainet.apps.kllama.GpuTensorBridge
import sk.ainet.apps.llm.backend.BackendRegistry
import sk.ainet.context.ExecutionContext
import sk.ainet.models.llama.LlamaRuntimeWeights
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

internal actual fun registerPlatformBackends() {
    BackendRegistry.register(CpuBackendProvider())
    // When Metal backend is released, register it here:
    // BackendRegistry.register(MetalBackendProvider())
}

internal actual fun <T : DType> createGpuTensorBridge(ctx: ExecutionContext, dtype: KClass<T>): GpuTensorBridge<T>? = null

internal actual fun <T : DType> createGraphAccelerator(
    ctx: ExecutionContext,
    weights: LlamaRuntimeWeights<T>,
    dtype: KClass<T>,
    eps: Float
): GraphAccelerator<T>? = null
