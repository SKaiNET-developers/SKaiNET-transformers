package sk.ainet.models.llama

import sk.ainet.apps.llm.weights.hasPackedMatmulKernel
import sk.ainet.apps.llm.weights.toBlockEncoding
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.lang.nn.quant.BlockQuantPacking
import sk.ainet.lang.tensor.Shape
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
 * `matmulQ{K}` kernels expect. Delegates to the shared
 * [BlockQuantPacking.relayoutRowMajorToBlockMajor] (#184 hoist 2); kept as an internal shim
 * for existing call sites and tests.
 */
@Deprecated(
    "Hoisted to the shared packer (#184): use BlockQuantPacking.relayoutRowMajorToBlockMajor",
    ReplaceWith(
        "BlockQuantPacking.relayoutRowMajorToBlockMajor(bytes, shape, bytesPerBlock, blockSize)",
        "sk.ainet.lang.nn.quant.BlockQuantPacking",
    ),
)
internal fun relayoutKSeriesRowMajorToBlockMajor(
    bytes: ByteArray,
    shape: Shape,
    bytesPerBlock: Int,
    blockSize: Int = 256,
): ByteArray = BlockQuantPacking.relayoutRowMajorToBlockMajor(bytes, shape, bytesPerBlock, blockSize)

/**
 * Pack raw GGUF `bytes` of logical `[out, in]` shape into heap-packed block tensor data the
 * matmul kernels read directly, with the row-major → block-major relayout. Null for types
 * without a packed kernel (caller dequantizes those to FP32). Delegates to the shared
 * [BlockQuantPacking] packer (#184 hoist 2), which covers Q4_K / Q5_K / Q6_K / Q8_0 plus
 * Q4_0 / Q5_0 / Q5_1 (#170).
 *
 * @param preTransposed when `true` (the default as of the engine 0.40.0 native-kernel
 *   closure train, #184 (3)/#170), packs via [BlockQuantPacking.packPreTransposed] —
 *   logical `[in, out]`, marked [sk.ainet.lang.nn.quant.PreTransposedWeight] — so
 *   [sk.ainet.lang.nn.transformer.linearProject] skips its per-forward `ops.transpose`
 *   for this weight. Pass `false` for the classic [BlockQuantPacking.pack] `[out, in]`
 *   result (kept reachable, deprecate-don't-delete, for fallback / parity comparison).
 *   Mirrors `packGemmaKQuant`.
 */
internal fun <T : DType> packLlamaKQuant(
    bytes: ByteArray,
    qt: GGMLQuantizationType,
    shape: Shape,
    preTransposed: Boolean = true,
): TensorData<T, *>? {
    val encoding = qt.toBlockEncoding() ?: return null
    // Legacy 32-elem formats (Q4_0/Q5_0/Q5_1) are new to this packed path
    // (#170): gate on an actually-registered matmul kernel — without one, a
    // packed weight would fall through to the generic elementwise matmul,
    // which misreads block-major bytes after the lazy transpose. `null` →
    // the caller's FP32 dequant fallback. Mirrors `packGemmaKQuant`.
    if (qt in legacyPackedQuantTypes && !qt.hasPackedMatmulKernel()) return null
    return if (preTransposed) {
        BlockQuantPacking.packPreTransposed(bytes, encoding, shape)
    } else {
        BlockQuantPacking.pack(bytes, encoding, shape)
    }
}

/** See `GemmaQuantLayout.legacyPackedQuantTypes` — kernel-gated new formats. */
private val legacyPackedQuantTypes: Set<GGMLQuantizationType> = setOf(
    GGMLQuantizationType.Q4_0,
    GGMLQuantizationType.Q5_0,
    GGMLQuantizationType.Q5_1,
)
