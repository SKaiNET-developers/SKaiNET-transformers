package sk.ainet.apps.kgemma

import kotlinx.coroutines.runBlocking
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.types.FP32
import sk.ainet.models.gemma.GEMMA_DEQUANTIZE_ALL
import sk.ainet.models.gemma.GemmaWeightLoader
import sk.ainet.models.gemma.GemmaNetworkLoader
import kotlin.test.Test

/**
 * Layer-parity probe against a llama.cpp `llama-eval-callback` trace, using FunctionGemma 270M
 * (18 layers, 640-wide, no PLE) as the small reproducer for the control-token forward-pass
 * divergence that also affects Gemma 4 E2B.
 *
 * Reference, `llama-eval-callback -p "<start_of_turn>user"` → tokens `[2, 105, 2364]`, values
 * shown for token index 1 (`<start_of_turn>`, id 105), first three and last three components:
 *
 * ```
 * embd        -0.0099   0.0332  -0.0184  …   0.0023   0.0085  -0.0356
 * inp_scaled  -0.2517   0.8400  -0.4663  …   0.0594   0.2162  -0.9017   (embd * sqrt(640))
 * l_out-0     -0.4209   0.6501  -0.2097  …  -0.0440  -0.3090   9.7131
 * l_out-1     -0.6006  -0.0620   2.2189  …  -0.1328  -0.1503 110.7476
 * l_out-17    -0.1422  -0.1694  -0.1601  …  -3.6288  -1.5057   0.1818
 * result_norm -30.1625 -23.3340 -15.1081  … -33.5116 -33.4075 -33.5124
 * ```
 *
 * Run with `GEMMA4_DUMP_HIDDEN=1` to get our per-block stats alongside; this probe prints the
 * embedding row itself, which is the first place the two traces can disagree.
 */
class FunctionGemmaLayerParityProbe {

    @Test
    fun embedding_and_layer_trace_for_control_token() {
        val gguf = System.getenv("FUNCTIONGEMMA_GGUF")
        if (gguf.isNullOrBlank() || System.getenv("FUNCTIONGEMMA_PROBE") != "1") {
            println("[skip] set FUNCTIONGEMMA_GGUF and FUNCTIONGEMMA_PROBE=1"); return
        }
        val referenceIds = intArrayOf(2, 105, 2364)

        val tokenizer2 = sk.ainet.apps.kllama.GGUFTokenizer.fromRandomAccessSource(
            JvmRandomAccessSource.open(gguf)
        )
        val ctx = DirectCpuExecutionContext.create()
        val weights = runBlocking {
            GemmaWeightLoader(
                randomAccessProvider = { JvmRandomAccessSource.open(gguf) },
                weightForm = GEMMA_DEQUANTIZE_ALL,
            ).loadToMapStreaming<FP32, Float>(ctx, FP32::class)
        }

        // 1) The raw embedding row for the control token, straight off the loaded weight.
        val embd = weights.tensors.getValue(sk.ainet.models.gemma.GemmaTensorNames.TOKEN_EMBEDDINGS)
        val dim = embd.shape[embd.shape.rank - 1]
        println("EMB tensor shape=${embd.shape} data=${embd.data::class.simpleName}")
        fun row(tokenId: Int): FloatArray =
            FloatArray(dim) { j -> embd.data.get(tokenId, j) as Float }
        fun norm(r: FloatArray): Double =
            kotlin.math.sqrt(r.fold(0.0) { a, v -> a + v.toDouble() * v.toDouble() })
        for (id in referenceIds) {
            val r = row(id)
            println(
                "EMB id=" + id + " first=[" + r[0] + ", " + r[1] + ", " + r[2] + "]" +
                    " last=[" + r[dim - 3] + ", " + r[dim - 2] + ", " + r[dim - 1] + "]" +
                    " norm=" + norm(r)
            )
        }
        println("EMB reference id=105 first=[-0.0099, +0.0332, -0.0184] last=[+0.0023, +0.0085, -0.0356]")

        // Norms of a few ordinary rows, for scale comparison against the control tokens.
        for (id in intArrayOf(105, 106, 107, 2364, 10000, 100000)) {
            println("EMB norm id=" + id + " = " + norm(row(id)))
        }

        // 2) Full forward with GEMMA4_DUMP_HIDDEN=1 to print per-block stats.
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
        var logits = runtime.forward(referenceIds[0])
        for (i in 1 until referenceIds.size) logits = runtime.forward(referenceIds[i])
        val buf = logits.data.copyToFloatArray()
        val top5 = buf.toList().mapIndexed { i, v -> i to v }.sortedByDescending { it.second }.take(5)
        println("LOGITS top-5 after [2,105,2364]: " + top5.joinToString { it.first.toString() + "=" + it.second })

        // Bisect over prefill length against the llama.cpp reference (see companion test doc).
        val full = intArrayOf(
            2,105,55060,107,3048,659,496,2028,600,740,776,1292,11687,607,506,2269,5151,46,163688,
            236787,828,236779,19323,236782,7777,236787,52,3407,1873,7606,573,496,4563,52,236764,
            19031,29616,15921,29616,7125,29616,7777,236787,52,17698,1463,52,236764,2084,236787,52,
            35410,52,5237,15979,24845,52,7125,52,1604,2084,236787,52,60688,52,1807,47,106,107,105,
            2364,107,3689,236789,236751,506,7606,528,9079,236881,106,107,105,4368,107,
        )
        for (n in intArrayOf(3, 8, 16, 24, 32, 48, 64, 72, 80, 85)) {
            runtime.reset()
            var l = runtime.forward(full[0])
            for (i in 1 until n) l = runtime.forward(full[i])
            val b = l.data.copyToFloatArray()
            var best = 0
            for (i in b.indices) if (b[i] > b[best]) best = i
            println("OURS n=" + n + " -> id=" + best + " logit=" + b[best] + " piece='" +
                runCatching { tokenizer2.decode(best) }.getOrElse { "?" }.replace("\n", "\\n") + "'")
        }
    }
}
