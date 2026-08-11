package sk.ainet.models.gemma

import org.junit.jupiter.api.Tag
import kotlinx.coroutines.runBlocking
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.types.FP32
import java.io.File
import kotlin.test.Test

/** Dumps SKaiNET's dequantized values for representative tensors (one per quant
 *  type: Q5_1 attn_q, Q6_K ffn_down, Q8_0 token_embd) so they can be diffed
 *  against gguf's reference dequantize. Pins whether the parity residual is a
 *  SKaiNET dequant bug. */
@Tag("integration")
class RealGemmaDequantDumpTest {
    @Test
    fun dumpDequant() = runBlocking {
        val path = FunctionGemmaFixture.gguf
        val ctx = DirectCpuExecutionContext.create()
        val weights = Gemma4WeightLoader(
            randomAccessProvider = { JvmRandomAccessSource.open(path) },
            quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
        ).loadToMapStreaming<FP32, Float>(ctx, FP32::class)
        val m = weights.metadata
        println("ROPE full.base=${m.ropeParametersFull.base} sliding.base=${m.ropeParametersSliding.base}")
        println("ROPE full.partial=${m.ropeParametersFull.partialRotaryFactor} full.type=${m.ropeParametersFull.ropeType} full.factor=${m.ropeParametersFull.factor}")
        for (l in 0 until m.blockCount) {
            print("L$l=${m.getLayerType(l)}:${m.getRopeBase(l)} "); if (l % 6 == 5) println()
        }
        println()
        val outDir = File("/home/miso/projects/coral/build-mlir/out").apply { mkdirs() }
        for (nm in listOf("blk.0.attn_q.weight", "blk.0.ffn_down.weight", "token_embd.weight")) {
            val t = weights.tensors[nm] ?: run { println("MISSING $nm"); continue }
            val a = t.data.copyToFloatArray()
            println("DUMP $nm shape=${t.shape} n=${a.size}")
            val bb = java.nio.ByteBuffer.allocate(a.size * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN)
            for (f in a) bb.putFloat(f)
            File(outDir, "dq_${nm.replace('.', '_')}.bin").writeBytes(bb.array())
        }
    }
}
