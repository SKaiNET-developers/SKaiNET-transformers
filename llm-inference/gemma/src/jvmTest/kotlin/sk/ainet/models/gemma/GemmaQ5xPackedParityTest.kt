package sk.ainet.models.gemma

import java.io.File
import java.lang.foreign.Arena
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Assumptions
import org.junit.jupiter.api.Tag
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.dequant.DequantOps
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.nn.quant.BlockQuantPacking
import sk.ainet.lang.nn.transformer.linearProject
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q5_0TensorData
import sk.ainet.lang.tensor.data.Q5_1TensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32

/**
 * #170: Q5_1 / Q5_0 weights must stay PACKED through the NATIVE_OPTIMIZED
 * converter (instead of the #169 dequant-to-FP32 fallback) and their packed
 * `linearProject` results must match the FP32-dequant reference.
 *
 * Two layers of evidence:
 * - [synthetic tests] byte-level parity per format: the converter's exact
 *   packed pipeline (canonical bytes via `BlockQuantPacking.pack` + the
 *   engine's physical packed transpose matmul) vs the converter's exact FP32 fallback pipeline
 *   (`DequantOps.dequantFromBytes` + identity col→row transpose). Q5_0 is
 *   covered here only — the FunctionGemma checkpoint carries no Q5_0 tensor.
 * - [real checkpoint] FunctionGemma-270M "Q5_K_M" ships 81 of 236 tensors as
 *   Q5_1 (attn_q / attn_k / ffn_gate / ffn_up): after conversion all of them
 *   must be `Q5_1TensorData` (packed), none FP32-inflated. Token-for-token
 *   decode parity of the full model incl. these tensors is asserted by
 *   [GemmaQ5KPackedParityTest], which decodes NATIVE_OPTIMIZED (now packing
 *   Q5_1 too) against the DEQUANTIZE_TO_FP32 baseline.
 *
 * The packed path is availability-gated (`hasPackedMatmulKernel`), so on a
 * JVM with scalar/Panama providers (engine >= 0.39.0) these tests exercise
 * the packed branch; the FFM native tier from SKaiNET#951 (0.40.0) slots into
 * the same dispatch without any change here.
 */
class GemmaQ5xPackedParityTest {

    // --- synthetic byte-level parity -------------------------------------

    private fun halfBits(f: Float): Int {
        val bits = f.toRawBits()
        val sign = (bits ushr 16) and 0x8000
        var exp = ((bits ushr 23) and 0xFF) - 127 + 15
        var mant = bits and 0x7FFFFF
        if (exp <= 0) return sign
        if (exp >= 31) return sign or 0x7C00
        mant += 0x1000
        if (mant and 0x800000 != 0) {
            mant = 0
            exp += 1
            if (exp >= 31) return sign or 0x7C00
        }
        return sign or (exp shl 10) or (mant ushr 13)
    }

    /** One GGUF Q5_1 block (24 B): d f16, m f16, qh[4], qs[16]. */
    private fun q5_1Block(d: Float, m: Float, codes: IntArray): ByteArray {
        val out = ByteArray(24)
        val db = halfBits(d); val mb = halfBits(m)
        out[0] = (db and 0xFF).toByte(); out[1] = ((db ushr 8) and 0xFF).toByte()
        out[2] = (mb and 0xFF).toByte(); out[3] = ((mb ushr 8) and 0xFF).toByte()
        var qh = 0
        for (j in 0 until 32) if ((codes[j] ushr 4) and 1 == 1) qh = qh or (1 shl j)
        for (b in 0 until 4) out[4 + b] = ((qh ushr (8 * b)) and 0xFF).toByte()
        for (j in 0 until 16) out[8 + j] = ((codes[j] and 0xF) or ((codes[j + 16] and 0xF) shl 4)).toByte()
        return out
    }

    /** One GGUF Q5_0 block (22 B): d f16, qh[4], qs[16]. */
    private fun q5_0Block(d: Float, codes: IntArray): ByteArray {
        val out = ByteArray(22)
        val db = halfBits(d)
        out[0] = (db and 0xFF).toByte(); out[1] = ((db ushr 8) and 0xFF).toByte()
        var qh = 0
        for (j in 0 until 32) if ((codes[j] ushr 4) and 1 == 1) qh = qh or (1 shl j)
        for (b in 0 until 4) out[2 + b] = ((qh ushr (8 * b)) and 0xFF).toByte()
        for (j in 0 until 16) out[6 + j] = ((codes[j] and 0xF) or ((codes[j + 16] and 0xF) shl 4)).toByte()
        return out
    }

    /**
     * Packed-vs-dequant parity for one legacy Q5 format over the converter's
     * two pipelines, on a multi-block `[out, in]` weight (multi-block in both
     * dimensions so a relayout indexing bug cannot cancel out).
     */
    private fun assertPackedMatchesDequant(qt: GGMLQuantizationType) {
        val outDim = 3
        val inDim = 64
        val blocksPerRow = inDim / 32
        val shape = Shape(outDim, inDim)
        val bpb = if (qt == GGMLQuantizationType.Q5_1) 24 else 22

        val gguf = ByteArray(outDim * blocksPerRow * bpb)
        for (r in 0 until outDim) {
            for (b in 0 until blocksPerRow) {
                val blockIdx = r * blocksPerRow + b
                val d = 0.25f * ((blockIdx % 4) + 1)
                val codes = IntArray(32) { j -> (j * 11 + r * 7 + b * 3) % 32 }
                val block =
                    if (qt == GGMLQuantizationType.Q5_1) q5_1Block(d, -1.5f + 0.5f * (blockIdx % 3), codes)
                    else q5_0Block(d, codes)
                block.copyInto(gguf, blockIdx * bpb)
            }
        }

        val ctx = DirectCpuExecutionContext.create()
        val x = ctx.fromFloatArray<FP32, Float>(
            Shape(2, inDim), FP32::class,
            FloatArray(2 * inDim) { i -> ((i * 13 + 5) % 19 - 9) / 9.0f },
        )

        // Reference: the converter's FP32 fallback pipeline, verbatim.
        val floats = DequantOps.dequantFromBytes(gguf, qt, shape.volume)
        val rowMajor = DequantOps.transposeColumnMajorToRowMajor(floats, inDim, outDim)
        val wRef = ctx.fromFloatArray<FP32, Float>(shape, FP32::class, rowMajor)
        val ref = linearProject(ctx.ops, x, wRef).data.copyToFloatArray()

        // Packed: the converter's classic packed pipeline, verbatim — canonical
        // checkpoint bytes, [out, in] shape; the engine's physical packed
        // ops.transpose (>= 0.40.1) inside linearProject produces kernel order.
        val encoding = if (qt == GGMLQuantizationType.Q5_1) TensorEncoding.Q5_1 else TensorEncoding.Q5_0
        @Suppress("UNCHECKED_CAST")
        val data = BlockQuantPacking.pack<FP32>(gguf, encoding, shape) as TensorData<FP32, Float>
        val y = linearProject(ctx.ops, x, ctx.fromData(data, FP32::class)).data.copyToFloatArray()

        assertEquals(ref.size, y.size)
        for (i in ref.indices) {
            assertTrue(
                abs(ref[i] - y[i]) <= 1e-3f * maxOf(1.0f, abs(ref[i])),
                "$qt packed[$i]=${y[i]} vs FP32-dequant ref ${ref[i]}",
            )
        }
    }

    @Test
    fun q5_1_packed_linearProject_matches_fp32_dequant_reference() =
        assertPackedMatchesDequant(GGMLQuantizationType.Q5_1)

    @Test
    fun q5_0_packed_linearProject_matches_fp32_dequant_reference() =
        assertPackedMatchesDequant(GGMLQuantizationType.Q5_0)

    // --- real checkpoint: Q5_1 tensors stay packed -----------------------

    @Test
    @Tag("integration")
    fun functionGemma_q5_1_tensors_stay_packed_after_conversion() = runBlocking {
        val gguf = FunctionGemmaFixture.gguf
        Assumptions.assumeTrue(File(gguf).exists(), "FunctionGemma GGUF not present — skipping")

        val ctx = DirectCpuExecutionContext.create()
        Arena.ofConfined().use { arena ->
            val weights = Gemma4WeightLoader(
                randomAccessProvider = { JvmRandomAccessSource.open(gguf) },
                quantPolicy = QuantPolicy.NATIVE_OPTIMIZED,
            ).loadToMapStreaming<FP32, Float>(ctx, FP32::class)
            val converted = convertGemmaWeightsToMemSeg(weights, ctx, arena)

            val q51 = converted.quantTypes.filterValues { it == GGMLQuantizationType.Q5_1 }.keys
            // functiongemma-physical-ai-v10-Q5_K_M carries 81 Q5_1 tensors
            // (attn_q/attn_k/ffn_gate/ffn_up); a rename in a future fixture
            // still must leave SOME Q5_1 for this test to be meaningful.
            assertTrue(q51.isNotEmpty(), "fixture has no Q5_1 tensors — parity claim would be vacuous")

            val notPacked = q51.filter { name -> converted.tensors[name]?.data !is Q5_1TensorData }
            assertTrue(
                notPacked.isEmpty(),
                "Q5_1 tensors not packed after conversion (dequant fallback taken?): $notPacked",
            )
            println("Q5_1 packed after conversion: ${q51.size} tensors (e.g. ${q51.take(3)})")

            // No Q5_0 in this checkpoint — the synthetic test above carries
            // Q5_0 parity; assert the premise so a fixture change surfaces here.
            val q50 = converted.quantTypes.filterValues { it == GGMLQuantizationType.Q5_0 }
            if (q50.isNotEmpty()) {
                val notPacked50 = q50.keys.filter { converted.tensors[it]?.data !is Q5_0TensorData }
                assertTrue(notPacked50.isEmpty(), "Q5_0 tensors not packed: $notPacked50")
            }
        }
    }
}
