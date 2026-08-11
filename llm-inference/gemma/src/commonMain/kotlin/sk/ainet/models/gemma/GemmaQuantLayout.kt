package sk.ainet.models.gemma

import sk.ainet.apps.llm.weights.hasPackedMatmulKernel
import sk.ainet.apps.llm.weights.toBlockEncoding
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.lang.nn.quant.BlockQuantPacking
import sk.ainet.lang.tensor.Shape
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
 * Re-layout GGUF K-series bytes from row-major block order to the
 * input-block-major order the `matmulQ{K}` kernels expect.
 *
 * Delegates to the shared [BlockQuantPacking.relayoutRowMajorToBlockMajor]
 * (#184 hoist 2); kept as an internal shim because the JVM MemSeg converter
 * and the gemma tests call it by this name.
 *
 * @param bytesPerBlock 144 (Q4_K), 176 (Q5_K), 210 (Q6_K).
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
 * Pack raw GGUF `bytes` of logical `[out, in]` shape into the heap-packed block
 * tensor data the matmul kernels read directly. Performs the row-major →
 * block-major relayout. Returns `null` for types without a packed kernel
 * (caller dequantizes those to FP32).
 *
 * Delegates to the shared [BlockQuantPacking] packer (#184 hoist 2), which
 * covers all seven packed-kernel formats — Q4_K / Q5_K / Q6_K / Q8_0 plus
 * Q4_0 / Q5_0 / Q5_1 (#170: FunctionGemma "Q5_K_M" checkpoints carry their
 * attention q/k and ffn_gate weights as Q5_1, previously dequantized here).
 *
 * Q8_0 matters for gemma's tied `output`/lm_head: FunctionGemma's token_embd is
 * Q8_0, so keeping the lm_head packed (vs ~0.67 GB FP32) is what lets the eager
 * decode fit the 1.9 GB board, and it runs on the NEON Q8_0 kernel.
 *
 * commonMain → works on JVM and Kotlin/Native alike (no MemSeg / Arena).
 */
internal fun <T : DType> packGemmaKQuant(
    bytes: ByteArray,
    qt: GGMLQuantizationType,
    shape: Shape,
): TensorData<T, *>? {
    val encoding = qt.toBlockEncoding() ?: return null
    // The legacy 32-elem formats (Q4_0/Q5_0/Q5_1) are NEW to this packed path
    // (#170) — gate them on an actually-registered matmul kernel rather than
    // a hard engine version: without a kernel, a packed weight would fall
    // through to the generic elementwise matmul, which reads the block-major
    // bytes with row-major strides after the lazy transpose (garbage), so the
    // correct degradation is `null` → the caller's FP32 dequant fallback
    // (#169 behavior). Q4_K/Q5_K/Q6_K/Q8_0 keep their long-standing
    // unconditional packing. Under engine 0.39.0 the scalar/Panama Q5_x
    // kernels already satisfy the gate; SKaiNET#951 (0.40.0) adds the native
    // FFM/K-N/JNI tiers behind the same check.
    if (qt in legacyPackedQuantTypes && !qt.hasPackedMatmulKernel()) return null
    return BlockQuantPacking.pack(bytes, encoding, shape)
}

/**
 * GGUF legacy block formats whose packed-matmul support arrived with the
 * 0.39.0 loading + scalar/Panama kernels and completes with the native-tier
 * kernels of SKaiNET#951 (0.40.0). Packing these is gated on
 * [hasPackedMatmulKernel]; everything else packs as before.
 */
private val legacyPackedQuantTypes: Set<GGMLQuantizationType> = setOf(
    GGMLQuantizationType.Q4_0,
    GGMLQuantizationType.Q5_0,
    GGMLQuantizationType.Q5_1,
)
