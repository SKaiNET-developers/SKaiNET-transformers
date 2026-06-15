package sk.ainet.models.gemma

import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q5_KBlockTensorData
import sk.ainet.lang.tensor.data.Q6_KBlockTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
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
    blockSize: Int = 256,
): ByteArray {
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

/**
 * Block geometry `(blockElems, bytesPerBlock)` for the quant types this packer
 * handles. The K-series are 256-element super-blocks; Q8_0 is a 32-element block
 * (f16 scale + 32 int8). All four have a first-class CPU matmul kernel + a lazy
 * transpose in `ops.transpose`, so all four can stay packed instead of FP32.
 */
private fun quantBlockLayout(qt: GGMLQuantizationType): Pair<Int, Int>? = when (qt) {
    GGMLQuantizationType.Q4_K -> 256 to 144
    GGMLQuantizationType.Q5_K -> 256 to 176
    GGMLQuantizationType.Q6_K -> 256 to 210
    GGMLQuantizationType.Q8_0 -> 32 to 34
    else -> null
}

/**
 * Pack raw GGUF `bytes` of logical `[out, in]` shape into the heap-packed block
 * tensor data the matmul kernels read directly (Q4_K / Q5_K / Q6_K / Q8_0).
 * Performs the row-major → block-major relayout. Returns `null` for types
 * without a packed kernel (caller dequantizes those to FP32).
 *
 * Q8_0 matters for gemma's tied `output`/lm_head: FunctionGemma's token_embd is
 * Q8_0, so keeping the lm_head packed (vs ~0.67 GB FP32) is what lets the eager
 * decode fit the 1.9 GB board, and it runs on the NEON Q8_0 kernel. (Requires
 * the Q8_0 case in `ops.transpose` — engine — so `linearProject` can transpose
 * the packed weight; see transformers #178.)
 *
 * commonMain → works on JVM and Kotlin/Native alike (no MemSeg / Arena).
 */
internal fun <T : DType> packGemmaKQuant(
    bytes: ByteArray,
    qt: GGMLQuantizationType,
    shape: Shape,
): TensorData<T, *>? {
    val (blockElems, bpb) = quantBlockLayout(qt) ?: return null
    val relaid = relayoutKSeriesRowMajorToBlockMajor(bytes, shape, bpb, blockElems)
    @Suppress("UNCHECKED_CAST")
    return when (qt) {
        GGMLQuantizationType.Q4_K -> Q4_KBlockTensorData(shape, relaid) as TensorData<T, *>
        GGMLQuantizationType.Q5_K -> Q5_KBlockTensorData(shape, relaid) as TensorData<T, *>
        GGMLQuantizationType.Q6_K -> Q6_KBlockTensorData(shape, relaid) as TensorData<T, *>
        GGMLQuantizationType.Q8_0 -> Q8_0BlockTensorData(shape, relaid) as TensorData<T, *>
        else -> null
    }
}
