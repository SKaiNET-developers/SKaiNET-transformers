package sk.ainet.apps.kgemma

import kotlinx.coroutines.runBlocking
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.nn.hooks.ForwardHooks
import sk.ainet.lang.nn.topology.ModuleNode
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import sk.ainet.models.gemma.GEMMA_DEQUANTIZE_ALL
import sk.ainet.models.gemma.GemmaWeightLoader
import sk.ainet.models.gemma.GemmaNetworkLoader
import kotlin.test.Test

/**
 * Captures every module output during one forward pass and prints the per-block hidden state, so
 * it can be lined up against a `llama-eval-callback` trace to find the FIRST diverging layer.
 *
 * Context: with byte-identical prompt ids we match llama.cpp's greedy argmax at prefill n=3 and
 * diverge by n=8, and a sweep over the architecture values the gemma3 GGUF omits (RoPE bases,
 * partial-rotary factor, ropeType, sliding/full layer pattern) does not fix it — while changing
 * the sliding RoPE base *breaks* the previously-correct n=3, confirming those defaults are right.
 * So the fault is inside a block, not in the guessed metadata.
 *
 * Reference for prompt `[2, 105, 2364]` at token index 1, first three components of `l_out-N`:
 * ```
 * l_out-0  -0.4209   0.6501  -0.2097   …   9.7131
 * l_out-1  -0.6006  -0.0620   2.2189   … 110.7476
 * l_out-2  -0.3748  -0.0865   2.0292   … 109.4381
 * l_out-3   1.1400  -1.9443   3.2290   … 258.8997
 * l_out-4   2.6605  -0.4618   5.2972   … 261.9620
 * l_out-5  -2.8948  -1.3700  -0.1114   … 274.5901
 * ```
 * (last value shown is component 639; note llama.cpp's huge final component, a known Gemma
 * activation outlier — a useful fingerprint for alignment.)
 */
class FunctionGemmaBlockTraceProbe {

    private class Capture : ForwardHooks {
        val rows = mutableListOf<Pair<String, FloatArray>>()
        override fun onForwardBegin(module: ModuleNode, input: Any) {}
        override fun onForwardEnd(module: ModuleNode, input: Any, output: Any) {
            val name = module.id.ifBlank { module.name }
            val t = output as? Tensor<*, *> ?: return
            val data = runCatching { t.data.copyToFloatArray() }.getOrNull() ?: return
            rows += name to data
        }
    }

    @Test
    fun per_block_trace() {
        val gguf = System.getenv("FUNCTIONGEMMA_GGUF")
        if (gguf.isNullOrBlank() || System.getenv("FUNCTIONGEMMA_PROBE") != "1") {
            println("[skip] set FUNCTIONGEMMA_GGUF and FUNCTIONGEMMA_PROBE=1"); return
        }
        val capture = Capture()
        val ctx = DirectCpuExecutionContext(_hooks = capture)
        val weights = runBlocking {
            GemmaWeightLoader(
                randomAccessProvider = { JvmRandomAccessSource.open(gguf) },
                weightForm = GEMMA_DEQUANTIZE_ALL,
            ).loadToMapStreaming<FP32, Float>(ctx, FP32::class)
        }
        val patched = weights.copy(
            metadata = weights.metadata.copy(
                ropeParametersFull = weights.metadata.ropeParametersFull.copy(partialRotaryFactor = 1.0f),
            ),
        )
        val model = GemmaNetworkLoader.fromWeights(ctx, patched, FP32::class)
        val runtime = sk.ainet.apps.llm.OptimizedLLMRuntime(
            model, ctx, sk.ainet.apps.llm.OptimizedLLMMode.DIRECT, FP32::class,
            random = kotlin.random.Random.Default,
        )

        // Prefill [2, 105, 2364]; keep only the LAST token's trace, matching the reference's
        // token index 1... note llama.cpp evaluates all 3 at once, so its row 1 is <start_of_turn>.
        // Here we capture the pass for token 105 (the second forward).
        runtime.forward(2)
        capture.rows.clear()
        runtime.forward(105)
        val afterControl = capture.rows.toList()
        capture.rows.clear()
        runtime.forward(2364)

        fun fmt(v: FloatArray) =
            "[%+.4f, %+.4f, %+.4f … %+.4f]".format(v[0], v[1], v[2], v[v.size - 1])

        println("TRACE modules captured for token 105: ${afterControl.size}")
        // Layer 0 only, every module regardless of width, to line up against the
        // reference's Qcur/Kcur/Vcur/attn internals.
        var printed = 0
        for ((name, v) in afterControl) {
            if (printed++ > 24) break
            println("TRACE %-22s n=%-5d %s".format(name, v.size, fmt(v)))
        }
        println("TRACE reference l_out-0 [-0.4209, +0.6501, -0.2097 … +9.7131]")
        println("TRACE reference l_out-1 [-0.6006, -0.0620, +2.2189 … +110.7476]")
    }
}
