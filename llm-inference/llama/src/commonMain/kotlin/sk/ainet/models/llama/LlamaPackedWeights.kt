package sk.ainet.models.llama

import sk.ainet.context.ExecutionContext
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.dequant.DequantOps
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.IntArrayTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32

/**
 * commonMain (Kotlin/Native-capable) converter for `NATIVE_OPTIMIZED` Llama weights — the Llama
 * analogue of `convertGemmaWeightsPacked`. Turns the raw-byte quantized tensors a NATIVE_OPTIMIZED
 * load produces into the forms the DSL matmul path consumes:
 *
 * - **Q4_K / Q5_K / Q6_K / Q8_0 matmul weights** → heap-packed `Q*BlockTensorData` (keep the
 *   GGUF footprint; run the in-kernel dequant matmul, NEON on the board).
 * - **token_embd** → FP32 dequant in `[vocab, embed]` order (gathered, not matmul'd; no transpose).
 * - **everything else quantized without a packed kernel** → FP32 dequant transposed to `[out, in]`.
 *
 * No `java.lang.foreign` — runs on the board (Kotlin/Native) and JVM alike.
 */
public fun convertLlamaWeightsPacked(
    weights: DecoderGgufWeights<*, *>,
    ctx: ExecutionContext,
): DecoderGgufWeights<*, *> {
    @Suppress("UNCHECKED_CAST")
    val typed = weights as DecoderGgufWeights<DType, Any>
    val quantTypes = typed.quantTypes
    if (quantTypes.isEmpty()) return weights

    // Memory: drain the SOURCE map as we convert (P1). The streaming loader builds
    // `tensors` as a LinkedHashMap, so we can `remove` each raw tensor right after
    // packing it — its bytes become GC-eligible immediately instead of the whole raw
    // model staying resident alongside the whole packed model. That load-time doubling
    // (~3.2 GB peak for TinyLlama Q4_K) is what OOM-kills the 2 GB board; draining keeps
    // peak ≈ packed-so-far + the one tensor in flight. Falls back to non-destructive if
    // the map is somehow immutable (correctness unchanged, just no memory win).
    @Suppress("UNCHECKED_CAST")
    val src = typed.tensors as? MutableMap<String, Tensor<DType, Any>>
    val names = typed.tensors.keys.toList()

    val newTensors = linkedMapOf<String, Tensor<DType, Any>>()
    for (name in names) {
        val tensor = src?.remove(name) ?: typed.tensors.getValue(name)
        // qt == null (norms/f32) or already-packed (streaming fused path leaves them out of
        // quantTypes) → packLlamaTensor returns the tensor unchanged (idempotent).
        newTensors[name] = packLlamaTensor(name, tensor, quantTypes[name], typed.metadata, ctx)
        // Reclaim this tensor's transient copies (raw source + extractRawBytes + relayout input)
        // before moving on — on Kotlin/Native they otherwise accumulate uncollected and blow up
        // peak RSS. No-op on the JVM. Paired with the source-map drain above.
        gcCollectHint()
    }
    @Suppress("UNCHECKED_CAST")
    return DecoderGgufWeights(typed.metadata, newTensors, typed.quantTypes) as DecoderGgufWeights<*, *>
}

/**
 * Convert ONE raw NATIVE_OPTIMIZED tensor to its DSL-consumed form: Q4_K/Q5_K/Q6_K/Q8_0 matmul
 * weights → heap-packed `Q*BlockTensorData`; `token_embd` → FP32 `[vocab, embed]` (gathered);
 * other quantized-without-a-packed-kernel → FP32 transposed `[out, in]`.
 *
 * Shared by [convertLlamaWeightsPacked] (post-load pass, non-streaming) and the streaming loader's
 * fused load+pack path. Idempotent: `qt == null` or an unknown 2-D layout returns [tensor] as-is,
 * so packing the same map twice (or a map of already-packed tensors) is a no-op.
 */
internal fun packLlamaTensor(
    name: String,
    tensor: Tensor<DType, Any>,
    qt: GGMLQuantizationType?,
    metadata: LlamaModelMetadata,
    ctx: ExecutionContext,
): Tensor<DType, Any> {
    if (qt == null) return tensor // not quantized (norms, f32), or already packed
    val shape = logicalShapeFor(name, metadata) ?: return tensor // unknown 2-D layout — leave as-is
    val bytes = extractRawBytes(tensor.data)
    // token_embd is gathered (row lookup) → must be FP32. Other matrices (incl. output/lm_head)
    // stay packed and run the in-kernel matmul.
    val isEmbed = name == LlamaTensorNames.TOKEN_EMBEDDINGS
    val packed = if (!isEmbed) packLlamaKQuant<FP32>(bytes, qt, shape) else null
    return when {
        packed != null -> {
            @Suppress("UNCHECKED_CAST")
            ctx.fromData(packed as TensorData<FP32, Float>, FP32::class) as Tensor<DType, Any>
        }
        isEmbed -> dequantNoTranspose(bytes, qt, shape, ctx)
        else -> dequantTransposed(bytes, qt, shape, ctx)
    }
}

/** Dequant to FP32 in natural `[rows, cols]` order (embeddings — gathered, not matmul'd). */
@Suppress("UNCHECKED_CAST")
private fun dequantNoTranspose(
    bytes: ByteArray,
    qt: GGMLQuantizationType,
    shape: Shape,
    ctx: ExecutionContext,
): Tensor<DType, Any> {
    val floats = DequantOps.dequantFromBytes(bytes, qt, shape.volume)
    return ctx.fromData(DenseFloatArrayTensorData<FP32>(shape, floats), FP32::class) as Tensor<DType, Any>
}

/** Dequant to canonical FP32 `[out, in]` row-major (GGUF is column-major within a row). */
@Suppress("UNCHECKED_CAST")
private fun dequantTransposed(
    bytes: ByteArray,
    qt: GGMLQuantizationType,
    shape: Shape,
    ctx: ExecutionContext,
): Tensor<DType, Any> {
    val floats = DequantOps.dequantFromBytes(bytes, qt, shape.volume)
    val out = shape[0]
    val inDim = shape[1]
    val rowMajor = DequantOps.transposeColumnMajorToRowMajor(floats, inDim, out)
    return ctx.fromFloatArray<FP32, Float>(shape, FP32::class, rowMajor) as Tensor<DType, Any>
}

/** Read raw packed bytes back from a NATIVE_OPTIMIZED quant tensor (JVM IntArray / Native Byte). */
internal fun extractRawBytes(data: TensorData<*, *>): ByteArray {
    if (data is IntArrayTensorData<*>) {
        val buf = data.buffer
        return ByteArray(buf.size) { buf[it].toByte() }
    }
    val n = data.shape.volume
    @Suppress("UNCHECKED_CAST")
    val d = data as TensorData<*, Any?>
    return ByteArray(n) {
        when (val v = d[it]) {
            is Byte -> v
            is Int -> v.toByte()
            else -> error("convertLlamaWeightsPacked: cannot read bytes from ${data::class.simpleName}")
        }
    }
}
