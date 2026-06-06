package sk.ainet.models.gemma

import org.junit.jupiter.api.Tag
import kotlinx.coroutines.runBlocking
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.types.FP32
import kotlin.test.Test

/**
 * Loads the REAL FunctionGemma-270M gguf via the DSL-path loader
 * (Gemma4WeightLoader + DequantOps, Q5_K -> FP32) — NOT the eager runtime.
 * Validates: the loader handles this gemma3 gguf, and reports the real config
 * + weight tensors for the upcoming real-config trace + arg mapping.
 *
 * FP32 dequant here is a correctness-first choice (clean FP32 A/B); production
 * would keep weights packed (QuantPolicy.NATIVE_OPTIMIZED) for memory/NPU.
 */
@Tag("integration")
class RealGemmaLoadTest {
    @Test
    fun loadFunctionGemmaWeights() = runBlocking {
        val path = "/home/miso/projects/coral/sl2610-voice-cc-kt/models/functiongemma-physical-ai-v10-Q5_K_M.gguf"
        val ctx = DirectCpuExecutionContext.create()
        val loader = Gemma4WeightLoader(
            randomAccessProvider = { JvmRandomAccessSource.open(path) },
            quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
        )
        val w = loader.loadToMapStreaming<FP32, Float>(ctx, FP32::class)
        val m = w.metadata
        println("REALCFG emb=${m.embeddingLength} layers=${m.blockCount} heads=${m.headCount} kv=${m.kvHeadCount} headDim=${m.headDim} ffn=${m.intermediateSize} vocab=${m.vocabSize} sliding=${m.slidingWindow}")
        println("TENSORS ${w.tensors.size}")
        w.tensors.entries.sortedBy { it.key }.take(6).forEach { println("  ${it.key} ${it.value.shape}") }
    }
}
