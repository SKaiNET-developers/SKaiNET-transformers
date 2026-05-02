package sk.ainet.models.apertus

import sk.ainet.context.ExecutionContext
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.dequant.DequantOps
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q4MemorySegmentTensorData
import sk.ainet.lang.tensor.data.Q6_KBlockTensorData
import sk.ainet.lang.tensor.data.Q8MemorySegmentTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import java.lang.foreign.Arena

/**
 * JVM-side post-processor that converts the raw byte-shape quantized tensors
 * returned by [ApertusWeightLoader] under `QuantPolicy.NATIVE_OPTIMIZED` into
 * proper block-major / MemorySegment-backed [TensorData] wrappers that the
 * SIMD / FFM matmul kernels can consume directly.
 *
 * Without this step, every Q/K/V/O attention projection and FFN matmul in
 * the DSL `apertusNetwork()` forward path fails — `linearProject` calls
 * `ops.matmul(input, ops.transpose(weight))`, but `transpose` rejects the
 * rank-1 byte tensors the loader produces under `NATIVE_OPTIMIZED`. With
 * this step, each quantized weight ends up as the right wrapper:
 *
 * - `Q4_K` → [Q4_KBlockTensorData] (relayout from GGUF row-major
 *   `[row, block]` order to the input-block-major `[block, row]` order
 *   `JvmQuantizedVectorKernels.matmulQ4_KVec` expects). 144-byte blocks.
 * - `Q6_K` → [Q6_KBlockTensorData]. 210-byte blocks.
 * - `Q4_0` → [Q4MemorySegmentTensorData] (arena-allocated, 64-byte aligned).
 * - `Q8_0` → [Q8MemorySegmentTensorData].
 * - `Q5_K` → fallback: dequant to FP32. Apertus-8B-Instruct-2509 Q4_K_S has
 *   8 Q5_K tensors out of 324 total — the cost is negligible. `skainet-lang-core`
 *   has no `Q5_KBlockTensorData` yet; once it does, this branch can switch
 *   to the packed path.
 * - Anything else → leave the raw byte tensor in place and warn. The forward
 *   pass will likely fail downstream, but we don't want to silently dequant
 *   tensors we don't recognise.
 *
 * The token embedding (`token_embd.weight`) is **not** in the
 * `quantTypes` map by construction — [ApertusWeightLoader] force-dequants
 * it during loading because `Embedding.gather` requires the logical
 * `[vocab, dim]` shape. Float tensors (norms, output_norm, …) likewise
 * pass through this converter unchanged.
 *
 * Usage — caller manages the [Arena] lifecycle (the MemorySegment-backed
 * tensors are valid only while the Arena is open):
 * ```kotlin
 * val raw = ApertusWeightLoader.fromRandomAccess(...)
 *     .loadToMap<FP32, Float>(ctx)
 * Arena.ofShared().use { arena ->
 *     val converted = convertApertusWeightsToMemSeg(raw, ctx, arena)
 *     val model = ApertusNetworkLoader.fromWeights(ctx, converted)
 *     // ... forward / generate ...
 * }
 * ```
 */
public fun <T : DType, V> convertApertusWeightsToMemSeg(
    weights: ApertusWeights<T, V>,
    ctx: ExecutionContext,
    arena: Arena
): ApertusWeights<T, V> {
    if (weights.quantTypes.isEmpty()) return weights

    // Drain the loader's quant-bytes sidecar as we go so each ~MB byte buffer
    // is eligible for GC the moment the converter has wrapped (and possibly
    // relaid) it. Without this we'd hold ~5 GB of raw bytes plus another
    // ~5 GB of relaid bytes in heap simultaneously on Apertus-8B Q4_K_S.
    val drainable = HashMap(weights.quantBytes)

    val newTensors = LinkedHashMap(weights.tensors)
    for ((name, qt) in weights.quantTypes) {
        val logicalShape = weights.logicalShapes[name] ?: continue
        val bytes = drainable.remove(name) ?: continue
        newTensors[name] = convertOne(qt, name, ctx, arena, logicalShape, bytes)
    }
    // Drop quantBytes / logicalShapes / quantTypes from the result — they
    // were one-shot inputs to this converter; carrying them through wastes
    // ~5 GB of heap on real models.
    return weights.copy(
        tensors = newTensors,
        quantTypes = emptyMap(),
        logicalShapes = emptyMap(),
        quantBytes = emptyMap()
    )
}

@Suppress("UNCHECKED_CAST")
private fun <T : DType, V> convertOne(
    qt: GGMLQuantizationType,
    name: String,
    ctx: ExecutionContext,
    arena: Arena,
    logicalShape: Shape,
    bytes: ByteArray
): Tensor<T, V> {
    val advertisedDtype = FP32::class
    return when (qt) {
        GGMLQuantizationType.Q4_0 -> {
            val data = Q4MemorySegmentTensorData.fromRawBytes(logicalShape, bytes, arena)
            ctx.fromData(data as TensorData<FP32, Float>, advertisedDtype) as Tensor<T, V>
        }

        GGMLQuantizationType.Q8_0 -> {
            val data = Q8MemorySegmentTensorData.fromRawBytes(logicalShape, bytes, arena)
            ctx.fromData(data as TensorData<FP32, Float>, advertisedDtype) as Tensor<T, V>
        }

        GGMLQuantizationType.Q4_K -> {
            val relaid = relayoutKSeriesRowMajorToBlockMajor(bytes, logicalShape, BYTES_PER_Q4_K_BLOCK)
            val data = Q4_KBlockTensorData.fromRawBytes(logicalShape, relaid)
            ctx.fromData(data as TensorData<FP32, Float>, advertisedDtype) as Tensor<T, V>
        }

        GGMLQuantizationType.Q6_K -> {
            val relaid = relayoutKSeriesRowMajorToBlockMajor(bytes, logicalShape, BYTES_PER_Q6_K_BLOCK)
            val data = Q6_KBlockTensorData.fromRawBytes(logicalShape, relaid)
            ctx.fromData(data as TensorData<FP32, Float>, advertisedDtype) as Tensor<T, V>
        }

        GGMLQuantizationType.Q5_K -> {
            // No packed-path kernel yet — dequant. Apertus-8B Q4_K_S has only 8
            // of these (~250 MB total dequantised), so the memory cost is small.
            val floats = DequantOps.dequantFromBytes(bytes, qt, logicalShape.volume)
            ctx.fromFloatArray<FP32, Float>(logicalShape, advertisedDtype, floats) as Tensor<T, V>
        }

        else -> {
            error("ApertusMemSegConverter: unsupported quant type $qt for '$name'. Add a wrapper or fall back to dequant.")
        }
    }
}

private const val BYTES_PER_Q4_K_BLOCK = 144
private const val BYTES_PER_Q6_K_BLOCK = 210
private const val K_SERIES_BLOCK_SIZE = 256

/**
 * Re-layout GGUF K-series bytes from row-major block order
 * (block at row `r`, block index `b` within the row → byte offset
 * `(r * blocksPerRow + b) * bytesPerBlock`) to the input-block-major layout
 * the `matmulQ{K}_Vec` kernels index via `(blockIdx * outDim + r) * bytesPerBlock`.
 *
 * For a weight of shape `[outDim, inDim]` with `inDim % 256 == 0`, this is
 * a 2D block-level transpose of the `[outDim, inDim/256]` block grid. Bytes
 * inside a block are untouched.
 */
internal fun relayoutKSeriesRowMajorToBlockMajor(
    bytes: ByteArray,
    shape: Shape,
    bytesPerBlock: Int
): ByteArray {
    require(shape.rank == 2) { "K-series weight must be 2D, got rank ${shape.rank}" }
    val outDim = shape[0]
    val inDim = shape[1]
    require(inDim % K_SERIES_BLOCK_SIZE == 0) {
        "K-series weight inDim ($inDim) must be a multiple of $K_SERIES_BLOCK_SIZE"
    }
    val blocksPerRow = inDim / K_SERIES_BLOCK_SIZE
    val expected = outDim.toLong() * blocksPerRow.toLong() * bytesPerBlock.toLong()
    require(bytes.size.toLong() >= expected) {
        "K-series byte buffer size ${bytes.size} < expected $expected for shape [$outDim, $inDim] @ ${bytesPerBlock}B/block"
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
