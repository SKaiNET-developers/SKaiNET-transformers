package sk.ainet.models.gemma

import java.lang.foreign.Arena
import sk.ainet.apps.llm.weights.hasPackedMatmulKernel
import sk.ainet.context.ExecutionContext
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.GGML_QUANT_SIZES
import sk.ainet.io.gguf.dequant.DequantOps
import sk.ainet.lang.nn.quant.BlockQuantPacking
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.IntArrayTensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q5_0BlockTensorData
import sk.ainet.lang.tensor.data.Q5_1BlockTensorData
import sk.ainet.lang.tensor.data.Q5_KBlockTensorData
import sk.ainet.lang.tensor.data.Q6_KBlockTensorData
import sk.ainet.lang.tensor.data.Q4MemorySegmentTensorData
import sk.ainet.lang.tensor.data.Q8MemorySegmentTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32

// logicalShapeFor + relayoutKSeriesRowMajorToBlockMajor moved to commonMain
// (GemmaQuantLayout.kt) so the Kotlin/Native board path shares them. This
// JVM-only file keeps the MemSeg (FFM) conversion + the FP32 dequant fallbacks.

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
 * The token embedding (`token_embd`) is gathered by row, not matmul'd, so
 * when its quant layout is row-sliceable (row width divisible by the quant
 * block size) it stays PACKED, wrapped as a [GemmaPerLayerTokenEmbedTensorData]
 * ([sk.ainet.lang.tensor.data.RowDequantSource]): the shared `Embedding` layer
 * and the engine `ops.gather` dequant only the one row per token actually
 * looked up. For FunctionGemma's Q8_0 `token_embd` that is ~178 MB packed vs
 * ~0.67 GB FP32 — the last big FP32 block on this path (#178 / #184 (1)).
 * Non-row-sliceable layouts fall back to a full FP32 dequant.
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
    val logicalShapes = typedWeights.logicalShapes
    val newTensors = linkedMapOf<String, Tensor<DType, Any>>()
    for ((name, tensor) in typedWeights.tensors) {
        val qt = quantTypes[name]
        newTensors[name] = when {
            qt == null -> tensor // not quantized — leave as-is
            else -> {
                val logicalShape = logicalShapes[name]
                if (logicalShape == null) {
                    println("WARNING: GemmaMemSegConverter: no logical shape for '$name' in weights map; leaving as-is")
                    tensor
                } else if (name == Gemma4TensorNames.TOKEN_EMBEDDINGS) {
                    // Row-sliceable quant layouts stay packed (row-dequant gather);
                    // anything else falls back to the full FP32 dequant. Mirrors
                    // `GemmaPackedWeights.dequantNoTranspose` (the board path).
                    val block = GGML_QUANT_SIZES[qt]
                    if (block != null && logicalShape.rank == 2 && logicalShape[1] % block.first == 0) {
                        tokenEmbedToRowDequant(tensor, qt, ctx, dtype, logicalShape)
                    } else {
                        dequantToFloat(tensor, qt, name, ctx, dtype, logicalShape)
                    }
                } else if (name == Gemma4TensorNames.PER_LAYER_TOKEN_EMBD) {
                    // Can't dequant — per_layer_token_embd on E2B is 9 GB FP32.
                    // The loader already wrapped the raw bytes in a
                    // GemmaPerLayerTokenEmbedTensorData; just pass through.
                    // Fall back to the wrapper path if somehow the data is
                    // still in the generic IntArrayTensorData form.
                    if (tensor.data is GemmaPerLayerTokenEmbedTensorData) tensor
                    else tokenEmbedToRowDequant(tensor, qt, ctx, dtype, logicalShape)
                } else {
                    convertOne(tensor, qt, name, ctx, arena, dtype, logicalShape)
                }
            }
        }
    }
    @Suppress("UNCHECKED_CAST")
    return Gemma4Weights(
        typedWeights.metadata,
        newTensors,
        typedWeights.quantTypes,
        typedWeights.logicalShapes
    ) as Gemma4Weights<*, *>
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
    dtype: kotlin.reflect.KClass<T>,
    shape: Shape
): Tensor<T, V> {
    val bytes = extractBytes(tensor.data)
    // We advertise the logical FP32 dtype to the DSL (since the runtime's
    // matmul kernels consume FP32 activations × quant weights). The tensor's
    // runtime data type is the specific quant class; matmul dispatch
    // inspects tensor.data at runtime to pick a kernel, so the declared
    // DType generic only matters for the type system.
    val advertisedDtype = FP32::class
    return when (qt) {
        GGMLQuantizationType.Q4_0 -> {
            val data = Q4MemorySegmentTensorData.fromRawBytes(shape, bytes, arena)
            ctx.fromData(data as TensorData<FP32, Float>, advertisedDtype) as Tensor<T, V>
        }
        GGMLQuantizationType.Q8_0 -> {
            val data = Q8MemorySegmentTensorData.fromRawBytes(shape, bytes, arena)
            ctx.fromData(data as TensorData<FP32, Float>, advertisedDtype) as Tensor<T, V>
        }
        GGMLQuantizationType.Q4_K -> {
            // Keep Q4_K packed, but re-layout the GGUF-stored bytes from
            // row-major block order `[row, block]` to the input-block-major
            // order `[block, row]` that `JvmQuantizedVectorKernels.matmulQ4_KVec`
            // indexes via `(blockIdx * outputDim + o) * bytesPerBlock`.
            //
            // The lazy Q4_K transpose in `DefaultCpuOpsJvm` expects this
            // layout; combined, a Q4_K_M Gemma 4 E2B checkpoint (3.2 GB on
            // disk) stays near that footprint in RAM instead of inflating
            // to ~18 GB FP32.
            val relaid = BlockQuantPacking.relayoutRowMajorToBlockMajor(bytes, shape, 144, 256)
            val data = Q4_KBlockTensorData.fromRawBytes(shape, relaid)
            ctx.fromData(data as TensorData<FP32, Float>, advertisedDtype) as Tensor<T, V>
        }
        GGMLQuantizationType.Q6_K -> {
            // Same packed-path treatment as Q4_K, enabled by the
            // `matmulQ6_KVec` kernel + lazy transpose in `DefaultCpuOpsJvm`.
            // Gemma 4 E2B Q4_K_M uses Q6_K for ffn_gate/up/down, attn_v,
            // token_embd, and the tied lm_head — keeping these packed saves
            // ~12 GB of FP32 bloat (and the corresponding 7.5 GB per-forward
            // transpose transient). Sanity-checked against FP32 dequant and
            // Q6_K packed produces identical tokens — kernel math is right.
            val relaid = BlockQuantPacking.relayoutRowMajorToBlockMajor(bytes, shape, 210, 256)
            val data = Q6_KBlockTensorData.fromRawBytes(shape, relaid)
            ctx.fromData(data as TensorData<FP32, Float>, advertisedDtype) as Tensor<T, V>
        }
        GGMLQuantizationType.Q5_K -> {
            // Same packed-path treatment as Q4_K/Q6_K, enabled by the Q5_K
            // matmul kernel (scalar/Panama/native) + the lazy Q5_K transpose
            // in DefaultCpuOps. FunctionGemma-270M Q5_K_M ships most attn/FFN
            // weights as Q5_K, so keeping them packed (176 B/block) avoids the
            // FP32 inflation and runs the in-kernel dequant matmul.
            val relaid = BlockQuantPacking.relayoutRowMajorToBlockMajor(bytes, shape, 176, 256)
            val data = Q5_KBlockTensorData.fromRawBytes(shape, relaid)
            ctx.fromData(data as TensorData<FP32, Float>, advertisedDtype) as Tensor<T, V>
        }
        GGMLQuantizationType.Q5_1 -> {
            // Packed-path treatment for the 32-elem/24-byte legacy blocks
            // (#170): FunctionGemma-270M "Q5_K_M" ships attn_q/attn_k and
            // ffn_gate/ffn_up as Q5_1 (81 of 236 tensors), which until #170
            // took the FP32 dequant fallback below. The engine has Q5_1
            // kernels (scalar + Panama since 0.39.0; native FFM/K-N/JNI from
            // SKaiNET#951 / 0.40.0) + the lazy Q5_1 transpose, so the weights
            // stay packed and run the in-kernel dequant matmul.
            //
            // Gated on kernel AVAILABILITY (not engine version): if no
            // registered provider carries a Q5_1 kernel, packing would send
            // the weight down the generic elementwise matmul, which misreads
            // the block-major bytes after the lazy transpose — so fall back
            // to the always-correct #169 FP32 dequant instead.
            if (qt.hasPackedMatmulKernel()) {
                val relaid = BlockQuantPacking.relayoutRowMajorToBlockMajor(bytes, shape, 24, 32)
                val data = Q5_1BlockTensorData.fromRawBytes(shape, relaid)
                ctx.fromData(data as TensorData<FP32, Float>, advertisedDtype) as Tensor<T, V>
            } else {
                dequantPackedToFp32<T, V>(bytes, qt, shape, ctx)
            }
        }
        GGMLQuantizationType.Q5_0 -> {
            // Same as Q5_1, 22-byte blocks (f16 d + qh + qs, symmetric).
            if (qt.hasPackedMatmulKernel()) {
                val relaid = BlockQuantPacking.relayoutRowMajorToBlockMajor(bytes, shape, 22, 32)
                val data = Q5_0BlockTensorData.fromRawBytes(shape, relaid)
                ctx.fromData(data as TensorData<FP32, Float>, advertisedDtype) as Tensor<T, V>
            } else {
                dequantPackedToFp32<T, V>(bytes, qt, shape, ctx)
            }
        }
        else -> {
            // Any other quant type without a packed SIMD kernel (Q4_1/Q2_K/…)
            // would otherwise be left as raw 1-D bytes, which `linearProject` then can't
            // transpose ("Transpose requires at least 2 dimensions"). Dequantize to a
            // correct FP32 `[out, in]` weight so the DSL path runs; the supported packed
            // types above keep their fast SIMD form. This trades those tensors' memory
            // savings for correctness until a packed kernel exists.
            dequantPackedToFp32<T, V>(bytes, qt, shape, ctx)
        }
    }
}

/**
 * Dequantize raw GGUF quant `bytes` of logical shape `[out, in]` to a canonical FP32
 * `[out, in]` row-major weight — the same layout `Gemma4WeightLoader.createTensor` produces
 * on the `DEQUANTIZE_TO_FP32` path. GGUF stores K/legacy-quant blocks column-major within a
 * row, so the dequantized floats are transposed column-major → row-major (rows = `in`,
 * cols = `out`) to match what `linearProject` (`x @ W.t()`) expects.
 */
@Suppress("UNCHECKED_CAST")
private fun <T : DType, V> dequantPackedToFp32(
    bytes: ByteArray,
    qt: GGMLQuantizationType,
    shape: Shape,
    ctx: ExecutionContext,
): Tensor<T, V> {
    val floats = DequantOps.dequantFromBytes(bytes, qt, shape.volume)
    val out = shape[0]
    val inDim = shape[1]
    val rowMajor = DequantOps.transposeColumnMajorToRowMajor(floats, inDim, out)
    return ctx.fromFloatArray<FP32, Float>(shape, FP32::class, rowMajor) as Tensor<T, V>
}

/**
 * Wrap the raw Q-series bytes of a row-gathered embedding table
 * (`per_layer_token_embd.weight` or the main `token_embd.weight`) in a
 * [GemmaPerLayerTokenEmbedTensorData] that dequants one row at a time.
 * Avoids the FP32 blow-up that [dequantToFloat] would produce — 9 GB for
 * Gemma 4 E2B's PLE table, ~0.67 GB for FunctionGemma's `token_embd`.
 * See the class kdoc for the memory math.
 */
@Suppress("UNCHECKED_CAST")
private fun <T : DType, V> tokenEmbedToRowDequant(
    tensor: Tensor<T, V>,
    qt: GGMLQuantizationType,
    ctx: ExecutionContext,
    dtype: kotlin.reflect.KClass<T>,
    logicalShape: Shape
): Tensor<T, V> {
    val bytes = extractBytes(tensor.data)
    val data = GemmaPerLayerTokenEmbedTensorData(logicalShape, qt, bytes)
    return ctx.fromData(data as sk.ainet.lang.tensor.data.TensorData<FP32, Float>, FP32::class) as Tensor<T, V>
}

@Suppress("UNCHECKED_CAST")
private fun <T : DType, V> dequantToFloat(
    tensor: Tensor<T, V>,
    qt: GGMLQuantizationType,
    name: String,
    ctx: ExecutionContext,
    dtype: kotlin.reflect.KClass<T>,
    logicalShape: Shape
): Tensor<T, V> {
    val bytes = extractBytes(tensor.data)
    val volume = logicalShape.volume
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
    // Always produce a real FP32 tensor — the DSL expects FP32 activations.
    return ctx.fromFloatArray<FP32, Float>(logicalShape, FP32::class, floats) as Tensor<T, V>
}

/**
 * Back-compat shim that delegates to the commonMain
 * [relayoutKSeriesRowMajorToBlockMajor] at Q4_K's 144-byte block size. Kept for
 * any callers outside this file pinned to the old name.
 */
@Deprecated(
    "Hoisted to the shared packer (#184): use BlockQuantPacking.relayoutRowMajorToBlockMajor",
    ReplaceWith(
        "BlockQuantPacking.relayoutRowMajorToBlockMajor(bytes, shape, 144, 256)",
        "sk.ainet.lang.nn.quant.BlockQuantPacking",
    ),
)
internal fun relayoutQ4_KRowMajorToBlockMajor(bytes: ByteArray, shape: sk.ainet.lang.tensor.Shape): ByteArray =
    BlockQuantPacking.relayoutRowMajorToBlockMajor(bytes, shape, 144, 256)

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
