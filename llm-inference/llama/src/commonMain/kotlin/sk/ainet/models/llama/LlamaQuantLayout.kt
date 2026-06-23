package sk.ainet.models.llama

import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q5_KBlockTensorData
import sk.ainet.lang.tensor.data.Q6_KBlockTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.DType

/**
 * Platform-neutral (commonMain) layout helpers for Llama quantized weights — the Llama analogue
 * of `GemmaQuantLayout`. A `NATIVE_OPTIMIZED` load stores quantized tensors as 1-D byte arrays,
 * so the converter needs the original `[out, in]` shape (from metadata) to relayout blocks.
 */

/**
 * Recover the logical 2-D `[out, in]` shape of a Llama weight from its GGUF name + metadata.
 * Null for tensors without a 2-D matmul layout (norms etc.). Llama has uniform per-layer dims,
 * so metadata is authoritative.
 */
internal fun logicalShapeFor(name: String, metadata: LlamaModelMetadata): Shape? {
    val embed = metadata.embeddingLength
    val vocab = metadata.vocabSize
    val headDim = if (metadata.headCount > 0) embed / metadata.headCount else 0
    val qDim = metadata.headCount * headDim
    val kvDim = metadata.kvHeadCount * headDim
    val ffn = metadata.feedForwardLength
    return when {
        name == LlamaTensorNames.TOKEN_EMBEDDINGS -> Shape(vocab, embed)
        name == LlamaTensorNames.OUTPUT_WEIGHT -> Shape(vocab, embed)
        name.startsWith("blk.") -> when {
            name.endsWith(".attn_q.weight") -> Shape(qDim, embed)
            name.endsWith(".attn_k.weight") -> Shape(kvDim, embed)
            name.endsWith(".attn_v.weight") -> Shape(kvDim, embed)
            name.endsWith(".attn_output.weight") -> Shape(embed, qDim)
            name.endsWith(".ffn_gate.weight") -> Shape(ffn, embed)
            name.endsWith(".ffn_up.weight") -> Shape(ffn, embed)
            name.endsWith(".ffn_down.weight") -> Shape(embed, ffn)
            else -> null
        }
        else -> null
    }
}

/**
 * Re-layout GGUF K-series bytes from row-major block order to the input-block-major order the
 * `matmulQ{K}` kernels expect. For a `[outDim, inDim]` weight with `inDim % 256 == 0` this is a
 * block-level 2-D transpose; bytes inside a block are untouched. (Mirror of GemmaQuantLayout.)
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

private fun quantBlockLayout(qt: GGMLQuantizationType): Pair<Int, Int>? = when (qt) {
    GGMLQuantizationType.Q4_K -> 256 to 144
    GGMLQuantizationType.Q5_K -> 256 to 176
    GGMLQuantizationType.Q6_K -> 256 to 210
    GGMLQuantizationType.Q8_0 -> 32 to 34
    else -> null
}

/**
 * Pack raw GGUF `bytes` of logical `[out, in]` shape into heap-packed block tensor data the
 * matmul kernels read directly (Q4_K / Q5_K / Q6_K / Q8_0), with the row-major → block-major
 * relayout. Null for types without a packed kernel (caller dequantizes those to FP32).
 */
internal fun <T : DType> packLlamaKQuant(
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
