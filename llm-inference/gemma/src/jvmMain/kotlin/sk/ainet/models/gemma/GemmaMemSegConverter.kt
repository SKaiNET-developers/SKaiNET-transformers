package sk.ainet.models.gemma

import java.lang.foreign.Arena
import sk.ainet.context.ExecutionContext
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.dequant.DequantOps
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.IntArrayTensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q4MemorySegmentTensorData
import sk.ainet.lang.tensor.data.Q8MemorySegmentTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32

/**
 * Recover the logical 2-D shape of a Gemma 4 weight tensor from its GGUF
 * name and the model metadata. `Gemma4WeightLoader` with
 * `NATIVE_OPTIMIZED` stores quantized tensors as 1-D byte arrays so the
 * tensor-data factory accepts them; the converter needs the original
 * shape to re-layout blocks and construct `Q4_KBlockTensorData` /
 * `Q4/Q8MemorySegmentTensorData`.
 *
 * Returns `null` for tensors that don't have a 2-D matmul layout (norms,
 * embeddings the converter wants to dequant anyway).
 */
internal fun logicalShapeFor(name: String, metadata: Gemma4ModelMetadata): Shape? {
    val embed = metadata.embeddingLength
    val vocab = metadata.vocabSize
    return when {
        name == Gemma4TensorNames.TOKEN_EMBEDDINGS -> Shape(vocab, embed)
        name == Gemma4TensorNames.OUTPUT_WEIGHT -> Shape(vocab, embed)
        name.startsWith("blk.") -> {
            val rest = name.substringAfter("blk.")
            val layer = rest.substringBefore('.').toIntOrNull() ?: return null
            val headDim = metadata.getHeadDim(layer)
            val qDim = metadata.headCount * headDim
            val kvDim = metadata.kvHeadCount * headDim
            val ffn = metadata.intermediateSize
            when {
                name.endsWith(".attn_q.weight") -> Shape(qDim, embed)
                name.endsWith(".attn_k.weight") -> Shape(kvDim, embed)
                name.endsWith(".attn_v.weight") -> Shape(kvDim, embed)
                name.endsWith(".attn_output.weight") -> Shape(embed, qDim)
                name.endsWith(".ffn_gate.weight") -> Shape(ffn, embed)
                name.endsWith(".ffn_up.weight") -> Shape(ffn, embed)
                name.endsWith(".ffn_down.weight") -> Shape(embed, ffn)
                else -> null
            }
        }
        else -> null
    }
}

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
                    dequantToFloat(tensor, qt, name, ctx, dtype, logicalShape)
                } else if (name == Gemma4TensorNames.PER_LAYER_TOKEN_EMBD) {
                    // Can't dequant — per_layer_token_embd on E2B is 9 GB FP32.
                    // The loader already wrapped the raw bytes in a
                    // GemmaPerLayerTokenEmbedTensorData; just pass through.
                    // Fall back to the wrapper path if somehow the data is
                    // still in the generic IntArrayTensorData form.
                    if (tensor.data is GemmaPerLayerTokenEmbedTensorData) tensor
                    else perLayerTokenEmbedToRowDequant(tensor, qt, ctx, dtype, logicalShape)
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
            val relaid = relayoutQ4_KRowMajorToBlockMajor(bytes, shape)
            val data = Q4_KBlockTensorData.fromRawBytes(shape, relaid)
            ctx.fromData(data as TensorData<FP32, Float>, advertisedDtype) as Tensor<T, V>
        }
        GGMLQuantizationType.Q5_K,
        GGMLQuantizationType.Q6_K -> {
            // No native matmul kernel for these yet — fall back to dequant.
            // Kept [out, in] layout (DSL transposes at runtime); pre-transposing
            // here would double-transpose via linearProject and give wrong math.
            val elemCount = shape.volume
            val floats = DequantOps.dequantFromBytes(bytes, qt, elemCount)
            ctx.fromFloatArray<FP32, Float>(shape, advertisedDtype, floats) as Tensor<T, V>
        }
        else -> {
            println("WARNING: GemmaMemSegConverter: unsupported quant type $qt for '$name'; leaving as-is")
            tensor
        }
    }
}

/**
 * Wrap the raw Q-series bytes of `per_layer_token_embd.weight` in a
 * [GemmaPerLayerTokenEmbedTensorData] that dequants one row at a time.
 * Avoids the 9 GB FP32 blow-up that [dequantToFloat] would produce on
 * Gemma 4 E2B. See the class kdoc for the memory math.
 */
@Suppress("UNCHECKED_CAST")
private fun <T : DType, V> perLayerTokenEmbedToRowDequant(
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
 * Re-layout GGUF Q4_K bytes from row-major block order (block at row r,
 * block index b within row → byte offset `(r * blocksPerRow + b) * 144`)
 * to the input-block-major layout that `matmulQ4_KVec` expects (block at
 * blockIdx bI for output row r → byte offset `(bI * outDim + r) * 144`).
 *
 * For a weight of shape `[outDim, inDim]` with inDim divisible by 256
 * (Q4_K block size), this is just a 2D block-level transpose of the
 * `[outDim, inDim/256]` array of 144-byte blocks. The bytes inside a
 * block are untouched.
 */
internal fun relayoutQ4_KRowMajorToBlockMajor(bytes: ByteArray, shape: sk.ainet.lang.tensor.Shape): ByteArray {
    val bytesPerBlock = 144
    val blockSize = 256
    require(shape.rank == 2) { "Q4_K weight must be 2D, got rank ${shape.rank}" }
    val outDim = shape[0]
    val inDim = shape[1]
    require(inDim % blockSize == 0) {
        "Q4_K weight inDim ($inDim) must be a multiple of $blockSize"
    }
    val blocksPerRow = inDim / blockSize
    val expected = outDim.toLong() * blocksPerRow.toLong() * bytesPerBlock.toLong()
    require(bytes.size.toLong() >= expected) {
        "Q4_K byte buffer size ${bytes.size} < expected $expected for shape [$outDim, $inDim]"
    }
    val out = ByteArray(bytes.size)
    for (r in 0 until outDim) {
        for (b in 0 until blocksPerRow) {
            val srcOff = (r * blocksPerRow + b) * bytesPerBlock
            val dstOff = (b * outDim + r) * bytesPerBlock
            System.arraycopy(bytes, srcOff, out, dstOff, bytesPerBlock)
        }
    }
    return out
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
