package sk.ainet.lang.nn.quant

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.nn.transformer.linearProject
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32

/**
 * Layout-parity matrix for **all seven** packed matmul formats: for each
 * encoding, the classic path ([BlockQuantPacking.pack] → `linearProject`'s
 * `ops.matmul(x, ops.transpose(W))`) and the pre-transposed path
 * ([BlockQuantPacking.packPreTransposed] → transpose-skipping branch) must
 * produce bit-identical output, and both must match an FP32 reference matmul
 * over the canonical dequant of the same bytes.
 *
 * This is the cross-check the 0.40.0→0.40.1 engine regression slipped
 * through: the engine's packed `ops.transpose` changed from a shape-swap
 * (which required kernel-order input bytes) to a physical block-grid
 * permutation (which requires canonical input bytes), and only Q5_1 had a
 * test on the classic path. Multi-block in both grid dimensions, so any
 * block-order convention mismatch between packer, transpose, and kernel
 * shows up as a value error rather than coincidentally passing.
 *
 * The synthetic blocks carry pseudorandom quant payloads with all FP16
 * scale/min fields pinned to small exact halves, so every format dequantizes
 * to finite, well-conditioned values without this test needing to reimplement
 * per-format dequant math (the engine's own `toFloatArray()` provides the
 * reference weight matrix — it reads canonical block order, matching
 * [BlockQuantPacking.pack]'s verbatim bytes).
 *
 * jvmTest: needs real packed-kernel dispatch (ServiceLoader scalar/Panama
 * providers), same as [LinearProjectionPreTransposedTest].
 */
class LinearProjectionPackedParityMatrixTest {

    private data class Fmt(
        val encoding: TensorEncoding,
        val blockElems: Int,
        val bytesPerBlock: Int,
        /** Byte offsets of FP16 fields inside a block to pin to an exact small half. */
        val f16Offsets: List<Int>,
    )

    // FP16 field positions per ggml block layout, as mirrored by the engine's
    // *TensorData constants (OFFSET_D = 208 for Q6_K; d/dmin at 0/2 for K-quants;
    // d (+ m for Q5_1) at 0 (/2) for the 32-element legacy formats).
    private val formats = listOf(
        Fmt(TensorEncoding.Q4_0, 32, 18, listOf(0)),
        Fmt(TensorEncoding.Q8_0, 32, 34, listOf(0)),
        Fmt(TensorEncoding.Q5_0, 32, 22, listOf(0)),
        Fmt(TensorEncoding.Q5_1, 32, 24, listOf(0, 2)),
        Fmt(TensorEncoding.Q4_K, 256, 144, listOf(0, 2)),
        Fmt(TensorEncoding.Q5_K, 256, 176, listOf(0, 2)),
        Fmt(TensorEncoding.Q6_K, 256, 210, listOf(208)),
    )

    /** 0.25f and 0.125f as FP16 little-endian byte pairs. */
    private val halfQuarter = byteArrayOf(0x00, 0x34)
    private val halfEighth = byteArrayOf(0x00, 0x30)

    private fun buildBlocks(fmt: Fmt, blockCount: Int): ByteArray {
        val out = ByteArray(blockCount * fmt.bytesPerBlock)
        for (b in 0 until blockCount) {
            val base = b * fmt.bytesPerBlock
            for (j in 0 until fmt.bytesPerBlock) {
                // Deterministic pseudorandom payload, distinct per block so a
                // misplaced block always changes dequant values.
                out[base + j] = ((b * 31 + j * 7 + 13) % 251).toByte()
            }
            fmt.f16Offsets.forEachIndexed { i, off ->
                val half = if (i == 0) halfQuarter else halfEighth
                out[base + off] = half[0]
                out[base + off + 1] = half[1]
            }
        }
        return out
    }

    @Test
    fun classic_and_preTransposed_paths_match_fp32_reference_for_all_seven_formats() {
        val ctx = DirectCpuExecutionContext.create()
        for (fmt in formats) {
            val outDim = 4
            val blocksPerRow = 2 // multi-block along the input dim
            val inDim = blocksPerRow * fmt.blockElems // and outDim (4) > blocksPerRow (2): non-square block grid
            val shape = Shape(outDim, inDim)
            val ggufBytes = buildBlocks(fmt, outDim * blocksPerRow)

            val x = ctx.fromFloatArray<FP32, Float>(
                Shape(2, inDim), FP32::class,
                FloatArray(2 * inDim) { i -> ((i * 31 + 7) % 17 - 8) / 8.0f },
            )

            // Canonical packed tensor ([out,in], checkpoint bytes verbatim).
            @Suppress("UNCHECKED_CAST")
            val packed = BlockQuantPacking.pack<FP32>(ggufBytes, fmt.encoding, shape)
                as TensorData<FP32, Float>

            // FP32 reference: engine's canonical dequant of the same bytes.
            val wFlat = (packed as PackedBlockStorage).toFloatArray()
            assertEquals(outDim * inDim, wFlat.size, "${fmt.encoding.name}: dequant size")
            for (v in wFlat) assertTrue(v.isFinite(), "${fmt.encoding.name}: non-finite dequant value $v")
            val wFp32 = ctx.fromFloatArray<FP32, Float>(shape, FP32::class, wFlat)
            val ref = linearProject(ctx.ops, x, wFp32).data.copyToFloatArray()

            // Classic path: canonical [out,in] + engine ops.transpose every forward.
            val yClassic = linearProject(ctx.ops, x, ctx.fromData(packed, FP32::class)).data.copyToFloatArray()

            // Pre-transposed path: [in,out] + marker, transpose skipped.
            val pre = BlockQuantPacking.packPreTransposed<FP32>(ggufBytes, fmt.encoding, shape)
                ?: error("${fmt.encoding.name}: packPreTransposed returned null")
            assertTrue(pre is PreTransposedWeight, "${fmt.encoding.name}: missing marker")
            assertEquals(Shape(inDim, outDim), pre.shape, "${fmt.encoding.name}: pre-transposed shape")
            @Suppress("UNCHECKED_CAST")
            val yPre = linearProject(ctx.ops, x, ctx.fromData(pre as TensorData<FP32, Float>, FP32::class))
                .data.copyToFloatArray()

            // Same kernel, same (post-transpose vs pre-relaid) bytes -> bit-identical.
            assertTrue(
                yClassic.contentEquals(yPre),
                "${fmt.encoding.name}: classic (transpose) path diverged from pre-transposed path",
            )
            // Both agree with the FP32 reference (accumulation-order tolerance).
            for (i in ref.indices) {
                assertTrue(
                    abs(ref[i] - yClassic[i]) <= 1e-3f * maxOf(1.0f, abs(ref[i])),
                    "${fmt.encoding.name}[$i]: packed ${yClassic[i]} vs FP32 ref ${ref[i]}",
                )
            }
        }
    }
}
