package sk.ainet.lang.nn.quant

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.nn.transformer.linearProject
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q4_0BlockTensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q5_0BlockTensorData
import sk.ainet.lang.tensor.data.Q5_1BlockTensorData
import sk.ainet.lang.tensor.data.Q5_KBlockTensorData
import sk.ainet.lang.tensor.data.Q6_KBlockTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32

/**
 * Packed-weight parity matrix for **all seven** GGML block formats the engine
 * loader can deliver: a canonical `[out, in]` engine block tensor
 * (`Q*BlockTensorData`, checkpoint bytes verbatim, ROW_MAJOR block order —
 * exactly the form `StreamingGgufParametersLoader` produces on the heap)
 * pushed through [linearProject] must match the same projection over the
 * engine's own canonical dequant of the same bytes.
 *
 * [linearProject] is the single seam every transformer DSL module projects
 * through; since the #338 migration arc it is one expression —
 * `ops.matmulWeightTransposed(input, weight)` — so this test pins the whole
 * packed dispatch chain behind that primitive (relayout-once, packed-kernel,
 * or dequant-view fallback) against the FP32 reference, for every encoding.
 *
 * The synthetic blocks carry pseudorandom quant payloads with all FP16
 * scale/min fields pinned to small exact halves, so every format dequantizes
 * to finite, well-conditioned values without this test reimplementing
 * per-format dequant math (the engine's `PackedBlockStorage.toFloatArray()`
 * provides the reference weight matrix).
 *
 * Tolerance: packed matmul kernels may quantize activations to int8 per
 * block (W4A8), so near-zero outputs carry absolute error — parity is gated
 * on an abs-floor OR relative band, not bitwise equality (#944).
 *
 * jvmTest: needs real packed-kernel dispatch (ServiceLoader scalar/Panama
 * providers).
 */
class LinearProjectionPackedParityMatrixTest {

    private data class Fmt(
        val encoding: TensorEncoding,
        val blockElems: Int,
        val bytesPerBlock: Int,
        /** Byte offsets of FP16 fields inside a block to pin to an exact small half. */
        val f16Offsets: List<Int>,
        val pack: (Shape, ByteArray) -> TensorData<*, *>,
    )

    // FP16 field positions per ggml block layout, as mirrored by the engine's
    // *TensorData constants (OFFSET_D = 208 for Q6_K; d/dmin at 0/2 for K-quants;
    // d (+ m for Q5_1) at 0 (/2) for the 32-element legacy formats).
    private val formats = listOf(
        Fmt(TensorEncoding.Q4_0, 32, 18, listOf(0)) { s, b -> Q4_0BlockTensorData(s, b) },
        Fmt(TensorEncoding.Q8_0, 32, 34, listOf(0)) { s, b -> Q8_0BlockTensorData(s, b) },
        Fmt(TensorEncoding.Q5_0, 32, 22, listOf(0)) { s, b -> Q5_0BlockTensorData(s, b) },
        Fmt(TensorEncoding.Q5_1, 32, 24, listOf(0, 2)) { s, b -> Q5_1BlockTensorData(s, b) },
        Fmt(TensorEncoding.Q4_K, 256, 144, listOf(0, 2)) { s, b -> Q4_KBlockTensorData(s, b) },
        Fmt(TensorEncoding.Q5_K, 256, 176, listOf(0, 2)) { s, b -> Q5_KBlockTensorData(s, b) },
        Fmt(TensorEncoding.Q6_K, 256, 210, listOf(208)) { s, b -> Q6_KBlockTensorData(s, b) },
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
    fun packed_linearProject_matches_fp32_reference_for_all_seven_formats() {
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

            // Canonical packed tensor ([out,in], checkpoint bytes verbatim) —
            // the heap form the engine loader delivers.
            @Suppress("UNCHECKED_CAST")
            val packed = fmt.pack(shape, ggufBytes) as TensorData<FP32, Float>

            // FP32 reference: engine's canonical dequant of the same bytes.
            val wFlat = (packed as PackedBlockStorage).toFloatArray()
            assertEquals(outDim * inDim, wFlat.size, "${fmt.encoding.name}: dequant size")
            for (v in wFlat) assertTrue(v.isFinite(), "${fmt.encoding.name}: non-finite dequant value $v")
            val wFp32 = ctx.fromFloatArray<FP32, Float>(shape, FP32::class, wFlat)
            val ref = linearProject(ctx.ops, x, wFp32).data.copyToFloatArray()

            // Packed path: same projection over the packed weight.
            val yPacked = linearProject(ctx.ops, x, ctx.fromData(packed, FP32::class)).data.copyToFloatArray()

            assertEquals(ref.size, yPacked.size, "${fmt.encoding.name}: output size")
            // W4A8 tolerance: abs-floor OR relative band (#944).
            for (i in ref.indices) {
                val err = abs(ref[i] - yPacked[i])
                assertTrue(
                    err <= maxOf(2e-3f, 0.05f * abs(ref[i])),
                    "${fmt.encoding.name}[$i]: packed ${yPacked[i]} vs FP32 ref ${ref[i]} (err=$err)",
                )
            }
        }
    }
}
