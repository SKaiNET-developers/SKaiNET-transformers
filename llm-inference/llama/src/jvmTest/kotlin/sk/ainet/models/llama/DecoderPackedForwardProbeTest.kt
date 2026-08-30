package sk.ainet.models.llama

import java.io.File
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlinx.coroutines.runBlocking
import org.junit.Assume.assumeTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.nn.transformer.linearProject
import sk.ainet.lang.tensor.data.MemorySegmentTensorDataFactory
import sk.ainet.lang.types.FP32

/**
 * Migration probe (#340/#338 arc): a real Q4_K projection through linearProject must match the
 * dequantized-FP32 oracle from the same file — for heap AND MemorySegment activations.
 */
class DecoderPackedForwardProbeTest {

    private val model = "/Users/A9973957/projects/neuroSKai/neuroSKai-research/1bitgpu/models/qwen2.5-1.5b-q4km.gguf"

    @Test
    fun packed_projection_matches_dequantized_oracle() {
        assumeTrue(File(model).canRead())
        val name = "blk.0.attn_q.weight"

        // ctx variants: default heap factory, and the CLI's MemSegment factory.
        val heapCtx = DirectCpuExecutionContext()
        val memSegCtx = DirectCpuExecutionContext(tensorDataFactory = MemorySegmentTensorDataFactory())

        val packedW = runBlocking {
            DecoderGgufWeightLoader(
                randomAccessProvider = { JvmRandomAccessSource.open(model) },
                acceptedArchitectures = setOf("qwen2"),
            ).loadToMapStreaming<FP32, Float>(heapCtx)
        }.tensors.getValue(name)

        println("packed: ${packedW.data::class.simpleName} ${packedW.shape}")
        // Oracle: the packed data's own block decode — forward matmul must equal
        // the decoded-weights matmul (the engine's core invariant).
        val pb = packedW.data as sk.ainet.lang.tensor.storage.PackedBlockStorage
        val decoded = pb.toFloatArray()

        val k = packedW.shape.dimensions.last()
        val xArr = FloatArray(k) { Random(42 + it).nextFloat() - 0.5f }

        for ((label, ctx) in listOf("heap" to heapCtx, "memseg" to memSegCtx)) {
            val x = ctx.fromFloatArray<FP32, Float>(sk.ainet.lang.tensor.Shape(1, k), FP32::class, xArr)
            val yPacked = linearProject(ctx.ops, x, packedW)
            var maxRel = 0.0f
            var worst = ""
            for (o in 0 until 64) {
                var acc = 0f
                for (c in 0 until k) acc += decoded[o * k + c] * xArr[c]
                val a = yPacked.data.get(0, o)
                // W4A8 tolerance: the packed kernels quantize the activation to int8
                // per block (#944 in the engine repo) — near-zero outputs carry the
                // whole absolute error, so gate on abs OR rel, not rel alone.
                val err = abs(a - acc)
                val rel = err / maxOf(2e-3f, abs(acc))
                if (rel > maxRel) { maxRel = rel; worst = "o=$o forward=$a decoded=$acc" }
            }
            println("[$label] maxRel=$maxRel  worst: $worst")
            assertTrue(maxRel < 1.0f, "[$label] packed projection diverges from its own decode: $worst (maxRel=$maxRel)")
        }
    }
}
