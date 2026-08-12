package sk.ainet.models.apertus

import sk.ainet.context.ExecutionContext
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.dequant.DequantOps
import sk.ainet.lang.nn.quant.BlockQuantPacking
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Q4MemorySegmentTensorData
import sk.ainet.lang.tensor.data.Q8MemorySegmentTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.storage.TensorEncoding
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
 * - `Q4_K` → [BlockQuantPacking.packPreTransposed]: input-block-major relayout
 *   + `[in, out]` shape + `PreTransposedWeight` marker, so `linearProject`
 *   skips `ops.transpose` entirely (aligning apertus with the gemma/llama
 *   converters — the previous inlined relayout under an unmarked `[out, in]`
 *   shape relied on the pre-0.40.1 shape-swap-only packed transpose and is
 *   silently corrupted by the physical block-grid permutation the engine
 *   performs since 0.40.1). 144-byte blocks.
 * - `Q6_K` → same pre-transposed packing. 210-byte blocks.
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
            val data = BlockQuantPacking.packPreTransposed<FP32>(bytes, TensorEncoding.Q4_K, logicalShape)
                ?: error("ApertusMemSegConverter: packPreTransposed returned null for Q4_K ('$name')")
            ctx.fromData(data as TensorData<FP32, Float>, advertisedDtype) as Tensor<T, V>
        }

        GGMLQuantizationType.Q6_K -> {
            val data = BlockQuantPacking.packPreTransposed<FP32>(bytes, TensorEncoding.Q6_K, logicalShape)
                ?: error("ApertusMemSegConverter: packPreTransposed returned null for Q6_K ('$name')")
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

private const val K_SERIES_BLOCK_SIZE = 256

/**
 * Re-layout GGUF K-series bytes from row-major block order to the
 * input-block-major layout the `matmulQ{K}_Vec` kernels index. Delegates to
 * the shared [sk.ainet.lang.nn.quant.BlockQuantPacking] packer (#184 hoist 2);
 * kept as an internal shim for existing call sites and tests.
 */
@Deprecated(
    "Hoisted to the shared packer (#184): use BlockQuantPacking.relayoutRowMajorToBlockMajor",
    ReplaceWith(
        "BlockQuantPacking.relayoutRowMajorToBlockMajor(bytes, shape, bytesPerBlock, 256)",
        "sk.ainet.lang.nn.quant.BlockQuantPacking",
    ),
)
internal fun relayoutKSeriesRowMajorToBlockMajor(
    bytes: ByteArray,
    shape: Shape,
    bytesPerBlock: Int
): ByteArray = BlockQuantPacking.relayoutRowMajorToBlockMajor(
    bytes, shape, bytesPerBlock, K_SERIES_BLOCK_SIZE,
)
