package sk.ainet.models.apertus

import java.lang.foreign.Arena
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.lang.nn.quant.PreTransposedWeight
import sk.ainet.lang.nn.transformer.linearProject
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q6_KBlockTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.types.FP32

/**
 * Synthetic (no real checkpoint) coverage for the Apertus K-quant converter
 * path — the gap that let the SKaiNET 0.40.1 layout-contract regression go
 * unobserved here: [ApertusRealGgufLoadingTest] needs a real GGUF and skips
 * by default, and the previous converter emitted relaid bytes under an
 * unmarked `[out, in]` shape, which `linearProject`'s per-forward
 * `ops.transpose` silently corrupts on engines >= 0.40.1.
 *
 * Asserts the converter now emits [PreTransposedWeight]-marked `[in, out]`
 * tensors (mirroring gemma/llama) whose `linearProject` output matches an
 * FP32 reference built from the canonical dequant of the same bytes.
 */
class ApertusMemSegConverterSyntheticTest {

    /** Deterministic block bytes with FP16 scale fields pinned to 0.25/0.125. */
    private fun buildBlocks(blockCount: Int, bytesPerBlock: Int, f16Offsets: List<Int>): ByteArray {
        val out = ByteArray(blockCount * bytesPerBlock)
        for (b in 0 until blockCount) {
            val base = b * bytesPerBlock
            for (j in 0 until bytesPerBlock) out[base + j] = ((b * 37 + j * 11 + 5) % 251).toByte()
            f16Offsets.forEachIndexed { i, off ->
                out[base + off] = 0x00
                out[base + off + 1] = if (i == 0) 0x34 else 0x30 // 0.25f / 0.125f
            }
        }
        return out
    }

    private fun metadata() = ApertusModelMetadata(
        architecture = "apertus",
        embeddingLength = 512,
        contextLength = 128,
        blockCount = 1,
        headCount = 2,
        kvHeadCount = 1,
        feedForwardLength = 512,
        ropeDimensionCount = null,
        vocabSize = 100,
    )

    private fun assertConvertedParity(qt: GGMLQuantizationType, bytesPerBlock: Int, f16Offsets: List<Int>) {
        val outDim = 4
        val inDim = 512 // blocksPerRow = 2: multi-block both grid dimensions
        val shape = Shape(outDim, inDim)
        val name = "blk.0.attn_q.weight"
        val bytes = buildBlocks(outDim * (inDim / 256), bytesPerBlock, f16Offsets)

        val ctx = DirectCpuExecutionContext.create()
        // Placeholder for the loader's rank-1 byte tensor; the converter
        // replaces it by key from the quantBytes sidecar.
        val placeholder = ctx.fromFloatArray<FP32, Float>(Shape(1), FP32::class, floatArrayOf(0f))
        val weights = ApertusWeights<FP32, Float>(
            metadata = metadata(),
            tensors = mapOf(name to placeholder),
            quantTypes = mapOf(name to qt),
            logicalShapes = mapOf(name to shape),
            quantBytes = mapOf(name to bytes),
        )

        Arena.ofConfined().use { arena ->
            val converted = convertApertusWeightsToMemSeg(weights, ctx, arena)
            val w = converted.tensors[name] ?: error("converted weight missing")

            assertTrue(
                w.data is PreTransposedWeight,
                "$qt: converter must emit a PreTransposedWeight-marked tensor (unmarked [out,in] + " +
                    "relaid bytes is corrupted by the >= 0.40.1 physical packed transpose)",
            )
            assertEquals(Shape(inDim, outDim), w.shape, "$qt: pre-transposed [in, out] shape")

            // FP32 reference: canonical dequant of the same bytes.
            @Suppress("UNCHECKED_CAST")
            val canonical = when (qt) {
                GGMLQuantizationType.Q4_K -> Q4_KBlockTensorData(shape, bytes)
                else -> Q6_KBlockTensorData(shape, bytes)
            } as TensorData<FP32, Float>
            val wFlat = (canonical as PackedBlockStorage).toFloatArray()
            for (v in wFlat) assertTrue(v.isFinite(), "$qt: non-finite dequant value $v")
            val wRef = ctx.fromFloatArray<FP32, Float>(shape, FP32::class, wFlat)

            val x = ctx.fromFloatArray<FP32, Float>(
                Shape(2, inDim), FP32::class,
                FloatArray(2 * inDim) { i -> ((i * 29 + 3) % 23 - 11) / 11.0f },
            )
            val ref = linearProject(ctx.ops, x, wRef).data.copyToFloatArray()
            val y = linearProject(ctx.ops, x, w).data.copyToFloatArray()

            assertEquals(ref.size, y.size)
            for (i in ref.indices) {
                assertTrue(
                    abs(ref[i] - y[i]) <= 1e-3f * maxOf(1.0f, abs(ref[i])),
                    "$qt converted[$i]=${y[i]} vs FP32-dequant ref ${ref[i]}",
                )
            }
        }
    }

    @Test
    fun q4k_converted_weight_is_marked_and_matches_fp32_reference() =
        assertConvertedParity(GGMLQuantizationType.Q4_K, bytesPerBlock = 144, f16Offsets = listOf(0, 2))

    @Test
    fun q6k_converted_weight_is_marked_and_matches_fp32_reference() =
        assertConvertedParity(GGMLQuantizationType.Q6_K, bytesPerBlock = 210, f16Offsets = listOf(208))
}
