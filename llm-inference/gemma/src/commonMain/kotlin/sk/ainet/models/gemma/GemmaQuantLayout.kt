package sk.ainet.models.gemma

import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q5_KBlockTensorData
import sk.ainet.lang.tensor.data.Q6_KBlockTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.DType

/**
 * Platform-neutral (commonMain) layout helpers for Gemma 4 quantized weights.
 *
 * These were previously JVM-only (inside `GemmaMemSegConverter`), but the
 * Kotlin/Native board path needs the same logic: on K/N there is no
 * `java.lang.foreign` MemSeg conversion, so the eager runtime keeps K-quant
 * weights as heap-packed `Q{4,5,6}_KBlockTensorData` produced here. The JVM
 * MemSeg converter reuses the same relayout + shape recovery.
 */

/**
 * Recover the logical 2-D shape of a Gemma 4 weight tensor from its GGUF name
 * and model metadata. `Gemma4WeightLoader` with `NATIVE_OPTIMIZED` stores
 * quantized tensors as 1-D byte arrays, so converters need the original
 * `[rows, cols]` shape to re-layout blocks. Returns `null` for tensors without
 * a 2-D matmul layout (norms, embeddings the converter dequantizes anyway).
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
 * Re-layout GGUF K-series bytes from row-major block order
 * (`(r * blocksPerRow + b) * bytesPerBlock`) to the input-block-major order the
 * `matmulQ{K}` kernels expect (`(b * outDim + r) * bytesPerBlock`). For a
 * `[outDim, inDim]` weight with `inDim % 256 == 0`, this is a block-level 2-D
 * transpose; bytes inside a block are untouched.
 *
 * @param bytesPerBlock 144 (Q4_K), 176 (Q5_K), 210 (Q6_K).
 */
internal fun relayoutKSeriesRowMajorToBlockMajor(
    bytes: ByteArray,
    shape: Shape,
    bytesPerBlock: Int,
): ByteArray {
    val blockSize = 256
    require(shape.rank == 2) { "K-series weight must be 2D, got rank ${shape.rank}" }
    val outDim = shape[0]
    val inDim = shape[1]
    require(inDim % blockSize == 0) { "K-series weight inDim ($inDim) must be a multiple of $blockSize" }
    val blocksPerRow = inDim / blockSize
    val expected = outDim.toLong() * blocksPerRow.toLong() * bytesPerBlock.toLong()
    require(bytes.size.toLong() >= expected) {
        "K-series byte buffer ${bytes.size} < expected $expected for [$outDim, $inDim] @ ${bytesPerBlock}B/block"
    }
    val out = ByteArray(bytes.size)
    for (r in 0 until outDim) {
        for (b in 0 until blocksPerRow) {
            val srcOff = (r * blocksPerRow + b) * bytesPerBlock
            val dstOff = (b * outDim + r) * bytesPerBlock
            bytes.copyInto(out, dstOff, srcOff, srcOff + bytesPerBlock)
        }
    }
    return out
}

/** Bytes per ggml block for the K-quant types this packer handles. */
private fun kQuantBytesPerBlock(qt: GGMLQuantizationType): Int? = when (qt) {
    GGMLQuantizationType.Q4_K -> 144
    GGMLQuantizationType.Q5_K -> 176
    GGMLQuantizationType.Q6_K -> 210
    else -> null
}

/**
 * Pack raw GGUF K-quant `bytes` of logical `[out, in]` shape into the
 * heap-packed block tensor data the matmul kernels read directly (Q4_K / Q5_K /
 * Q6_K). Performs the row-major → block-major relayout. Returns `null` for
 * non-K-quant types (caller dequantizes those to FP32).
 *
 * commonMain → works on JVM and Kotlin/Native alike (no MemSeg / Arena).
 */
internal fun <T : DType> packGemmaKQuant(
    bytes: ByteArray,
    qt: GGMLQuantizationType,
    shape: Shape,
): TensorData<T, *>? {
    val bpb = kQuantBytesPerBlock(qt) ?: return null
    val relaid = relayoutKSeriesRowMajorToBlockMajor(bytes, shape, bpb)
    @Suppress("UNCHECKED_CAST")
    return when (qt) {
        GGMLQuantizationType.Q4_K -> Q4_KBlockTensorData(shape, relaid) as TensorData<T, *>
        GGMLQuantizationType.Q5_K -> Q5_KBlockTensorData(shape, relaid) as TensorData<T, *>
        GGMLQuantizationType.Q6_K -> Q6_KBlockTensorData(shape, relaid) as TensorData<T, *>
        else -> null
    }
}
