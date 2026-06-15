package sk.ainet.models.gemma

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
 * commonMain (Kotlin/Native-capable) analogue of the jvmMain
 * `convertGemmaWeightsToMemSeg`. Converts the raw-byte quantized tensors a
 * `NATIVE_OPTIMIZED` load produces into the forms the DSL matmul path consumes:
 *
 * - **Q4_K / Q5_K / Q6_K matmul weights** → heap-packed `Q{4,5,6}_KBlockTensorData`
 *   (via [packGemmaKQuant], with the row-major→block-major relayout). These keep
 *   the GGUF footprint and run the in-kernel dequant matmul (NEON on the board).
 * - **token_embd / output** → FP32 dequant in canonical `[vocab, embed]` order
 *   (the embedding is gathered, not matmul'd, so no transpose).
 * - **everything else quantized** → FP32 dequant transposed to `[out, in]`
 *   row-major so `linearProject` (`x @ W.t()`) is correct.
 *
 * Unlike the MemSeg converter this uses no `java.lang.foreign` — it runs on the
 * SL2610 board binary (Kotlin/Native) as well as the JVM. The JVM still prefers
 * the MemSeg path (lazy transpose + Q4/Q8 MemSeg); this is the board path.
 */
public fun convertGemmaWeightsPacked(
    weights: Gemma4Weights<*, *>,
    ctx: ExecutionContext,
): Gemma4Weights<*, *> {
    @Suppress("UNCHECKED_CAST")
    val typed = weights as Gemma4Weights<DType, Any>
    val quantTypes = typed.quantTypes
    if (quantTypes.isEmpty()) return weights

    val logicalShapes = typed.logicalShapes
    val newTensors = linkedMapOf<String, Tensor<DType, Any>>()
    for ((name, tensor) in typed.tensors) {
        val qt = quantTypes[name]
        newTensors[name] = when {
            qt == null -> tensor // not quantized
            else -> {
                val shape = logicalShapes[name] ?: logicalShapeFor(name, typed.metadata)
                if (shape == null) {
                    tensor // unknown 2-D layout — leave as-is
                } else {
                    val bytes = extractRawBytes(tensor.data)
                    // Only the token-embedding table is gathered (row lookup) and so
                    // must be FP32 here. `output`/lm_head is a real matmul weight —
                    // it stays packed (FunctionGemma's tied output is Q8_0 → NEON
                    // Q8_0 kernel, transposed lazily by ops.transpose) instead of a
                    // second ~0.67 GB FP32 copy that would OOM the 1.9 GB board.
                    val isEmbed = name == Gemma4TensorNames.TOKEN_EMBEDDINGS
                    val packed = if (!isEmbed) packGemmaKQuant<FP32>(bytes, qt, shape) else null
                    when {
                        packed != null -> {
                            @Suppress("UNCHECKED_CAST")
                            ctx.fromData(packed as TensorData<FP32, Float>, FP32::class) as Tensor<DType, Any>
                        }
                        isEmbed -> dequantNoTranspose(bytes, qt, shape, ctx)
                        else -> dequantTransposed(bytes, qt, shape, ctx)
                    }
                }
            }
        }
    }
    @Suppress("UNCHECKED_CAST")
    return Gemma4Weights(typed.metadata, newTensors, typed.quantTypes, typed.logicalShapes) as Gemma4Weights<*, *>
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
    // Wrap the dequant array directly (no-copy) rather than ctx.fromFloatArray,
    // which routes through BufferHandleFactory.owned and allocates a second
    // full-size buffer — for the 262k×640 FP32 token_embd (~0.67 GB) that
    // transient double is itself enough to OOM the 1.9 GB board.
    return ctx.fromData(DenseFloatArrayTensorData<FP32>(shape, floats), FP32::class) as Tensor<DType, Any>
}

/**
 * Dequant to a canonical FP32 `[out, in]` row-major weight. GGUF stores K/legacy
 * blocks column-major within a row, so the dequantized floats are transposed
 * column-major → row-major to match what `linearProject` (`x @ W.t()`) expects.
 */
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

/**
 * Read the raw packed bytes back from a `NATIVE_OPTIMIZED` quant tensor. The
 * backing differs by platform/factory — JVM stores `IntArrayTensorData` (byte
 * values widened to Int); Kotlin/Native stores a Byte-typed tensor — so handle
 * both element types.
 */
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
            else -> error(
                "convertGemmaWeightsPacked: cannot read bytes from ${data::class.simpleName} " +
                    "(element ${v?.let { e -> e::class.simpleName }})",
            )
        }
    }
}
