@file:Suppress("UNCHECKED_CAST", "USELESS_CAST")

package sk.ainet.performance.native

import sk.ainet.apps.kllama.GpuTensorBridge
import sk.ainet.context.ExecutionContext
import sk.ainet.context.MetalExecutionContext
import sk.ainet.context.MlxExecutionContext
import sk.ainet.exec.tensor.ops.MetalTensorOps
import sk.ainet.exec.tensor.ops.MlxTensorOps
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

internal actual fun createMetalContext(): ExecutionContext? = MetalExecutionContext()

internal actual fun createMlxContext(): ExecutionContext? = MlxExecutionContext()

internal actual fun <T : DType> createGpuBridge(ctx: ExecutionContext): GpuTensorBridge<T>? {
    val ops = ctx.ops
    return when (ops) {
        is MetalTensorOps -> object : GpuTensorBridge<T> {
            override fun slice(tensor: Tensor<T, Float>, start: IntArray, stop: IntArray, strides: IntArray): Tensor<T, Float> =
                ops.slice(tensor, start, stop, strides) as Tensor<T, Float>
            override fun sliceUpdate(src: Tensor<T, Float>, update: Tensor<T, Float>, start: IntArray, stop: IntArray, strides: IntArray): Tensor<T, Float> =
                ops.sliceUpdate(src, update, start, stop, strides) as Tensor<T, Float>
            override fun concat(tensors: List<Tensor<T, Float>>, axis: Int): Tensor<T, Float> =
                ops.concat(tensors, axis) as Tensor<T, Float>
        }
        is MlxTensorOps -> object : GpuTensorBridge<T> {
            override fun slice(tensor: Tensor<T, Float>, start: IntArray, stop: IntArray, strides: IntArray): Tensor<T, Float> =
                ops.slice(tensor, start, stop, strides) as Tensor<T, Float>
            override fun sliceUpdate(src: Tensor<T, Float>, update: Tensor<T, Float>, start: IntArray, stop: IntArray, strides: IntArray): Tensor<T, Float> =
                ops.sliceUpdate(src, update, start, stop, strides) as Tensor<T, Float>
            override fun concat(tensors: List<Tensor<T, Float>>, axis: Int): Tensor<T, Float> =
                ops.concat(tensors, axis) as Tensor<T, Float>
        }
        else -> null
    }
}

internal actual fun availableNativeBackends(): List<String> = listOf("CPU", "Metal", "MLX")
