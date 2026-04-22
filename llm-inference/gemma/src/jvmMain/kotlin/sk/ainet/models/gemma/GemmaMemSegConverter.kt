package sk.ainet.models.gemma

import java.lang.foreign.Arena
import sk.ainet.context.ExecutionContext
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.dequant.DequantOps
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.IntArrayTensorData
import sk.ainet.lang.tensor.data.Q4MemorySegmentTensorData
import sk.ainet.lang.tensor.data.Q8MemorySegmentTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.DType

/**
 * Convert raw-byte quantized tensors in a [Gemma4Weights] map (produced by
 * loading with [sk.ainet.io.model.QuantPolicy.NATIVE_OPTIMIZED]) into
 * MemorySegment-backed Q4 / Q8 tensor data the DSL path can feed to its
 * SIMD matmul kernels.
 *
 * **Different from `MemSegWeightConverter` (Llama)**: no pre-transpose for
 * K-series weights. The Llama-path runtime (`LlamaRuntime.linearProject`)
 * picks direct-matmul-vs-transpose-then-matmul based on a shape check, and
 * pre-transposing FP32 K-series weights lets it take the direct branch. The
 * DSL path's [sk.ainet.lang.nn.transformer.linearProject] always transposes,
 * so pre-transposing here would produce double-transposed weights and the
 * wrong math. Instead, for K-series we dequant to FP32 and keep the
 * canonical `[out, in]` layout — the DSL transposes at runtime like any
 * other FP32 weight. That loses the Q4_K / Q6_K memory savings but keeps
 * numerical correctness until a quant-aware DSL dispatch (recognising
 * `linearProject` on a Q4_K tensor and skipping the transpose) is
 * implemented in the backend.
 *
 * Q4_0 and Q8_0 keep their packed quantized form. The CPU backend's
 * `ops.transpose` does a lazy shape-swap on those MemSeg tensors (no data
 * copy), and the `matmul(FloatArray, Q4/Q8_MemSeg)` SIMD kernels read the
 * packed bytes directly — so the full chain runs without a FP32 round-trip.
 *
 * Token embedding and the output projection are kept as FP32 regardless of
 * their source quant type: embedding needs row-gather access, and output is
 * typically sub-32-aligned past the last block. Both are tiny relative to
 * the attention/FFN weights, so the memory impact is negligible.
 *
 * @param weights Gemma 4 weights produced with `QuantPolicy.NATIVE_OPTIMIZED`
 *   (raw quant bytes in [IntArrayTensorData] + a `quantTypes` map).
 * @param ctx execution context used to wrap new tensor data.
 * @param arena arena that owns every allocated MemorySegment. The caller is
 *   responsible for closing it no sooner than the returned runtime's
 *   lifetime.
 * @return new [Gemma4Weights] with quantized tensors replaced appropriately.
 */
public fun convertGemmaWeightsToMemSeg(
    weights: Gemma4Weights<*, *>,
    ctx: ExecutionContext,
    arena: Arena
): Gemma4Weights<*, *> {
    @Suppress("UNCHECKED_CAST")
    val typedWeights = weights as Gemma4Weights<DType, Any>
    val quantTypes = typedWeights.quantTypes
    if (quantTypes.isEmpty()) return weights

    val dtype = inferDtype(typedWeights) ?: return weights
    val newTensors = linkedMapOf<String, Tensor<DType, Any>>()
    for ((name, tensor) in typedWeights.tensors) {
        val qt = quantTypes[name]
        newTensors[name] = when {
            qt == null -> tensor // not quantized — leave as-is
            name == Gemma4TensorNames.TOKEN_EMBEDDINGS ->
                dequantToFloat(tensor, qt, name, ctx, dtype)
            else -> convertOne(tensor, qt, name, ctx, arena, dtype)
        }
    }
    @Suppress("UNCHECKED_CAST")
    return Gemma4Weights(typedWeights.metadata, newTensors, typedWeights.quantTypes) as Gemma4Weights<*, *>
}

@Suppress("UNCHECKED_CAST")
private fun <T : DType, V> inferDtype(weights: Gemma4Weights<T, V>): kotlin.reflect.KClass<T>? {
    val first = weights.tensors.values.firstOrNull() ?: return null
    return first.dtype as kotlin.reflect.KClass<T>
}

@Suppress("UNCHECKED_CAST")
private fun <T : DType, V> convertOne(
    tensor: Tensor<T, V>,
    qt: GGMLQuantizationType,
    name: String,
    ctx: ExecutionContext,
    arena: Arena,
    dtype: kotlin.reflect.KClass<T>
): Tensor<T, V> {
    val bytes = extractBytes(tensor.data)
    val shape = tensor.shape
    return when (qt) {
        GGMLQuantizationType.Q4_0 -> {
            val data = Q4MemorySegmentTensorData.fromRawBytes(shape, bytes, arena)
            ctx.fromData(data as TensorData<T, V>, dtype)
        }
        GGMLQuantizationType.Q8_0 -> {
            val data = Q8MemorySegmentTensorData.fromRawBytes(shape, bytes, arena)
            ctx.fromData(data as TensorData<T, V>, dtype)
        }
        GGMLQuantizationType.Q4_K,
        GGMLQuantizationType.Q5_K,
        GGMLQuantizationType.Q6_K -> {
            // Dequant to FP32, keep [out, in] layout. DSL linearProject
            // transposes at runtime (free on FP32 MemSeg or the cheap
            // per-element default path).
            val floats = DequantOps.dequantFromBytes(bytes, qt, shape.volume)
            ctx.fromFloatArray<T, V>(shape, dtype, floats)
        }
        else -> {
            println("WARNING: GemmaMemSegConverter: unsupported quant type $qt for '$name'; leaving as-is")
            tensor
        }
    }
}

@Suppress("UNCHECKED_CAST")
private fun <T : DType, V> dequantToFloat(
    tensor: Tensor<T, V>,
    qt: GGMLQuantizationType,
    name: String,
    ctx: ExecutionContext,
    dtype: kotlin.reflect.KClass<T>
): Tensor<T, V> {
    val bytes = extractBytes(tensor.data)
    val volume = tensor.shape.volume
    val floats = when (qt) {
        GGMLQuantizationType.Q4_0,
        GGMLQuantizationType.Q8_0,
        GGMLQuantizationType.Q4_K,
        GGMLQuantizationType.Q5_K,
        GGMLQuantizationType.Q6_K ->
            DequantOps.dequantFromBytes(bytes, qt, volume)
        else -> {
            println("WARNING: GemmaMemSegConverter: cannot dequant $qt for '$name'; leaving as-is")
            return tensor
        }
    }
    return ctx.fromFloatArray<T, V>(tensor.shape, dtype, floats)
}

private fun extractBytes(data: TensorData<*, *>): ByteArray {
    if (data is IntArrayTensorData<*>) {
        val buf = data.buffer
        return ByteArray(buf.size) { buf[it].toByte() }
    }
    val size = data.shape.volume
    return ByteArray(size) {
        @Suppress("UNCHECKED_CAST")
        ((data as TensorData<*, Int>)[it]).toByte()
    }
}
