package sk.ainet.apps.kgemma

import kotlinx.coroutines.runBlocking
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.types.FP32
import sk.ainet.models.gemma.GEMMA_DEQUANTIZE_ALL
import sk.ainet.models.gemma.GemmaModelMetadata
import sk.ainet.models.gemma.GemmaWeightLoader
import sk.ainet.models.gemma.GemmaNetworkLoader
import kotlin.test.Test

/**
 * Parameter sweep over the architecture values our loader has to *guess* because the
 * FunctionGemma / gemma3 GGUF omits them, checked against llama.cpp's greedy argmax.
 *
 * Why a sweep: with byte-identical prompt ids we match llama.cpp at prefill n=3 and diverge by
 * n=8 — far below the 512 sliding window, so the error is position-driven and shows up almost
 * immediately. The values the checkpoint does not declare are exactly the position-related ones:
 *
 *  - `rope.dimension_count` absent  → partialRotaryFactor guessed (0.25 for gemma, forced 1.0 by
 *    `FunctionGemma.fromGguf`)
 *  - `rope.freq_base_swa` absent    → sliding-layer RoPE base defaults to 10000
 *  - no sliding-window pattern key  → `buildDefaultLayerTypes` assumes 5 sliding : 1 full
 *
 * llama.cpp reference (greedy, same ids): n=8 → 5192 `' language'` at logprob −0.02;
 * n=16 → 6436 `' tools'` at −1.0; n=3 → 107 `'\n'` (which we already match).
 */
class FunctionGemmaRopeSweepProbe {

    private val ids = intArrayOf(
        2, 105, 55060, 107, 3048, 659, 496, 2028, 600, 740, 776, 1292, 11687, 607, 506, 2269,
        5151, 46, 163688, 236787, 828, 236779, 19323, 236782, 7777, 236787, 52, 3407, 1873, 7606,
    )

    @Test
    fun sweep_rope_and_layer_pattern() {
        val gguf = System.getenv("FUNCTIONGEMMA_GGUF")
        if (gguf.isNullOrBlank() || System.getenv("FUNCTIONGEMMA_PROBE") != "1") {
            println("[skip] set FUNCTIONGEMMA_GGUF and FUNCTIONGEMMA_PROBE=1"); return
        }
        val ctx = DirectCpuExecutionContext.create()
        val weights = runBlocking {
            GemmaWeightLoader(
                randomAccessProvider = { JvmRandomAccessSource.open(gguf) },
                weightForm = GEMMA_DEQUANTIZE_ALL,
            ).loadToMapStreaming<FP32, Float>(ctx, FP32::class)
        }
        val base = weights.metadata
        println(
            "SWEEP base: layerTypes=${base.layerTypes} slidingWindow=${base.slidingWindow} " +
                "ropeFull=${base.ropeParametersFull} ropeSliding=${base.ropeParametersSliding}"
        )

        val allFull = List(base.blockCount) { "full_attention" }
        val allSliding = List(base.blockCount) { "sliding_attention" }

        val configs: List<Pair<String, GemmaModelMetadata>> = listOf(
            "A baseline(partial=1.0)" to base.copy(
                ropeParametersFull = base.ropeParametersFull.copy(partialRotaryFactor = 1.0f),
            ),
            "B partial=0.25" to base.copy(
                ropeParametersFull = base.ropeParametersFull.copy(partialRotaryFactor = 0.25f),
            ),
            "C slidingBase=1e6" to base.copy(
                ropeParametersFull = base.ropeParametersFull.copy(partialRotaryFactor = 1.0f),
                ropeParametersSliding = base.ropeParametersSliding.copy(base = 1_000_000f),
            ),
            "D allFullLayers" to base.copy(
                layerTypes = allFull,
                ropeParametersFull = base.ropeParametersFull.copy(partialRotaryFactor = 1.0f),
            ),
            "E allSlidingLayers" to base.copy(
                layerTypes = allSliding,
                ropeParametersFull = base.ropeParametersFull.copy(partialRotaryFactor = 1.0f),
            ),
            "F ropeTypeDefault" to base.copy(
                ropeParametersFull = base.ropeParametersFull.copy(
                    partialRotaryFactor = 1.0f, ropeType = "default",
                ),
            ),
        )

        for ((label, md) in configs) {
            val model = GemmaNetworkLoader.fromWeights(ctx, weights.copy(metadata = md), FP32::class)
            val runtime = sk.ainet.apps.llm.OptimizedLLMRuntime(
                model, ctx, sk.ainet.apps.llm.OptimizedLLMMode.DIRECT, FP32::class,
                random = kotlin.random.Random.Default,
            )
            val out = StringBuilder("SWEEP ").append(label.padEnd(26))
            for (n in intArrayOf(3, 8, 16)) {
                runtime.reset()
                var l = runtime.forward(ids[0])
                for (i in 1 until n) l = runtime.forward(ids[i])
                val b = l.data.copyToFloatArray()
                var best = 0
                for (i in b.indices) if (b[i] > b[best]) best = i
                out.append(" n=").append(n).append(":").append(best)
            }
            out.append("   (reference n=3:107 n=8:5192 n=16:6436)")
            println(out)
        }
    }
}
