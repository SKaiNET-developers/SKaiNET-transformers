package sk.ainet.models.gemma

import org.junit.jupiter.api.Tag
import kotlinx.coroutines.runBlocking
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.transformer.MultiHeadAttention
import sk.ainet.lang.types.FP32
import java.io.File
import kotlin.test.Test

/**
 * Eager-vs-llama.cpp split. Runs the REAL FunctionGemma DSL network EAGERLY
 * (DirectCpuExecutionContext, real ops) on the SAME 4 tokens fed to llama.cpp
 * and the baked vmfb. Writes logits to a raw .bin for comparison:
 *   - eager vs llama.cpp  -> isolates DSL/weights correctness (RoPE, dequant, norms)
 *   - vmfb  vs eager      -> isolates the StableHLO/IREE lowering
 * KV cache stripped so the config matches the vmfb trace exactly.
 */
@Tag("integration")
class RealGemmaEagerAbTest {
    @Test
    fun eagerLogits() = runBlocking {
        val path = "/home/miso/projects/coral/SKaiNET-embedded/sl2610-function-calling/models/functiongemma-physical-ai-v10-Q5_K_M.gguf"
        val ctx = DirectCpuExecutionContext.create()
        val weights = Gemma4WeightLoader(
            randomAccessProvider = { JvmRandomAccessSource.open(path) },
            quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
        ).loadToMapStreaming<FP32, Float>(ctx, FP32::class)
        val partial = (System.getProperty("partialRotary") ?: "1.0").toFloat()
        val patched = weights.copy(
            metadata = weights.metadata.copy(
                ropeParametersFull = weights.metadata.ropeParametersFull.copy(partialRotaryFactor = partial),
            ),
        )
        val model = GemmaNetworkLoader.fromWeights(ctx, patched, FP32::class)
        fun stripKvCache(m: Module<*, *>) {
            if (m is MultiHeadAttention<*, *>) m.kvCache = null
            m.modules.forEach { stripKvCache(it) }
        }
        stripKvCache(model)

        // Read the exact token ids llama.cpp used (ab_llama.py writes tokens.json)
        // so both sides see an identical prefix; fall back to a default.
        val tokFile = File("/home/miso/projects/coral/build-mlir/out/tokens.json")
        val tokens = if (tokFile.exists()) {
            tokFile.readText().trim().removeSurrounding("[", "]")
                .split(",").map { it.trim().toFloat() }.toFloatArray()
        } else floatArrayOf(2f, 887f, 506f, 2214f)
        println("TOKENS_USED ${tokens.toList()}")
        // Eager Embedding indexes a 1-D [seq] token tensor (not [1,seq]).
        val input = ctx.fromFloatArray<FP32, Float>(sk.ainet.lang.tensor.Shape(tokens.size), FP32::class, tokens)
        val out = model.forward(input, ctx as ExecutionContext)
        val logits = out.data.copyToFloatArray()
        println("EAGER out.shape=${out.shape} logits.size=${logits.size}")

        val vocab = 262153
        val seq = logits.size / vocab
        for (s in 0 until seq) {
            var best = 0; var bv = Float.NEGATIVE_INFINITY
            for (j in 0 until vocab) { val v = logits[s * vocab + j]; if (v > bv) { bv = v; best = j } }
            println("EAGER pos$s argmax=$best maxlogit=$bv")
        }
        val bb = java.nio.ByteBuffer.allocate(logits.size * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        for (f in logits) bb.putFloat(f)
        val o = File("/home/miso/projects/coral/build-mlir/out/eager_logits.bin")
        o.parentFile?.mkdirs(); o.writeBytes(bb.array())
        println("WROTE_EAGER ${o.absolutePath} (${seq}x$vocab)")
    }
}
