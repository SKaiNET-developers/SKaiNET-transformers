package sk.ainet.apps.kllama.cli

import sk.ainet.models.llama.GraphAccelerator
import sk.ainet.apps.kllama.GpuTensorBridge
import sk.ainet.context.ExecutionContext
import sk.ainet.models.llama.LlamaRuntimeWeights
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Register platform-specific [BackendProvider]s with [BackendRegistry].
 * Called once at startup before backend selection.
 */
internal expect fun registerPlatformBackends()

internal expect fun <T : DType> createGpuTensorBridge(ctx: ExecutionContext, dtype: KClass<T>): GpuTensorBridge<T>?
internal expect fun <T : DType> createGraphAccelerator(
    ctx: ExecutionContext,
    weights: LlamaRuntimeWeights<T>,
    dtype: KClass<T>,
    eps: Float
): GraphAccelerator<T>?
