package sk.ainet.llm.providers

import sk.ainet.context.ExecutionContext
import sk.ainet.io.ParametersLoader
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.model.DataType
import sk.ainet.io.safetensors.StreamingSafeTensorsReader
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.reflect.KClass

/**
 * [ParametersLoader] over a SafeTensors file that loads *float weights only*,
 * silently skipping integer buffers.
 *
 * Why this exists: HF BERT checkpoints exported by older `transformers`
 * versions persist non-weight buffers — `embeddings.position_ids` (I64
 * `arange`) in BGE among others. The engine's `SafeTensorsParametersLoader`
 * has no tensor filter and fails the whole load on the I64 → FP32 conversion.
 * Our DSL `BertEmbeddings` is index-free, so these buffers are never needed.
 *
 * Interim: remove once the engine's loader accepts a tensor-name filter
 * (SKaiNET-developers/SKaiNET#822) and route BERT loads back through
 * `SafeTensorsParametersLoader`.
 */
internal class FloatSafeTensorsLoader(
    private val sourceProvider: () -> RandomAccessSource,
) : ParametersLoader {

    @Suppress("UNCHECKED_CAST")
    override suspend fun <T : DType, V> load(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (name: String, tensor: Tensor<T, V>) -> Unit,
    ) {
        StreamingSafeTensorsReader.open(sourceProvider()).use { reader ->
            for (info in reader.tensors) {
                val floats = when (info.dataType) {
                    DataType.FLOAT32 -> f32(reader.loadTensorData(info))
                    DataType.FLOAT64 -> f64(reader.loadTensorData(info))
                    DataType.FLOAT16 -> f16(reader.loadTensorData(info))
                    DataType.BFLOAT16 -> bf16(reader.loadTensorData(info))
                    // Non-float buffers (position_ids & friends) are never
                    // BERT weights — skip instead of failing the load.
                    else -> continue
                }
                val shape = Shape(*info.shape.map { it.toInt() }.toIntArray())
                val tensor = ctx.fromFloatArray<T, Float>(shape, dtype, floats) as Tensor<T, V>
                onTensorLoaded(info.name, tensor)
            }
        }
    }

    private fun f32(bytes: ByteArray): FloatArray {
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(buf.remaining()).also { buf.get(it) }
    }

    private fun f64(bytes: ByteArray): FloatArray {
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer()
        return FloatArray(buf.remaining()) { buf.get(it).toFloat() }
    }

    private fun f16(bytes: ByteArray): FloatArray {
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        return FloatArray(buf.remaining()) { java.lang.Float.float16ToFloat(buf.get(it)) }
    }

    private fun bf16(bytes: ByteArray): FloatArray {
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        return FloatArray(buf.remaining()) { Float.fromBits(buf.get(it).toInt() shl 16) }
    }
}
