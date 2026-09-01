package sk.ainet.apps.kgemma

import kotlinx.coroutines.runBlocking
import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import sk.ainet.models.gemma.GEMMA_DEQUANTIZE_ALL
import sk.ainet.models.gemma.GemmaWeightLoader
import sk.ainet.models.gemma.GemmaModel
import sk.ainet.models.gemma.GemmaNetworkLoader
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Phase-1 make-or-break gate for the KV-cache with_past Gemma decoder: drive the two-graph
 * KV loop (`forwardPrefill` → loop `forwardWithPast`) EAGERLY on the CPU and assert it produces
 * the board/llama.cpp oracle token-for-token. This validates that qkNorm-before-RoPE, the two-base
 * SPLIT_HALF runtime cos/sin, GQA expansion, the sandwich norms, layer_output_scale and the final
 * logit softcap all compose correctly — before any export/board work.
 *
 * Skips without the GGUF or with too small a heap. 12g is the module default heap — override
 * with -PkgemmaTestMaxHeap. Run with:
 *   ./gradlew -PuseLocalSkainet=true \
 *     :llm-runtime:kgemma:jvmTest --tests "*FunctionGemmaWithPastCpuTest*"
 */
class FunctionGemmaWithPastCpuTest {
    private val gguf = FunctionGemmaFixture.gguf

    // "turn the light on" -> <tool_0>(state="on")<end> (token-for-token vs llama.cpp / the board).
    private val oracle = listOf(262146, 236769, 3255, 718, 498, 1373, 262152, 106)

    @Test
    fun with_past_two_graph_loop_matches_oracle() {
        // Gated OUTSIDE runBlocking: an aborted assumption propagates out of the coroutine
        // builder unchanged, but keeping the gate here also frees the block's last expression
        // from having to stay Unit for Jupiter discovery.
        FunctionGemmaFixture.assumeRealCheckpointRunnable()
        runBlocking {
            val ctx = DirectCpuExecutionContext.create()
            val tok = GGUFTokenizer.fromRandomAccessSource(JvmRandomAccessSource.open(gguf))
            val weights = GemmaWeightLoader(
                randomAccessProvider = { JvmRandomAccessSource.open(gguf) },
                weightForm = GEMMA_DEQUANTIZE_ALL,
            ).loadToMapStreaming<FP32, Float>(ctx, FP32::class)
            // gemma3 uses FULL rotary; force it (the gguf omits the factor — see FunctionGemmaExport).
            val patched = weights.copy(
                metadata = weights.metadata.copy(
                    ropeParametersFull = weights.metadata.ropeParametersFull.copy(partialRotaryFactor = 1.0f),
                ),
            )
            @Suppress("UNCHECKED_CAST")
            val model = GemmaNetworkLoader.fromWeights(ctx, patched, FP32::class) as GemmaModel<FP32, Float>

            val eot = tok.encode("<end_of_turn>").single()
            val eos = tok.eosTokenId
            val promptText = "<start_of_turn>user\nturn the light on<end_of_turn>\n<start_of_turn>model\n"
            val ptoks: List<Int> = listOf(tok.bosTokenId) + tok.encode(promptText).toList()

            // --- Prefill: process the whole prompt, seed the per-layer self K/V, sample token 0 ---
            val prefIn = ctx.fromFloatArray<FP32, Float>(Shape(ptoks.size), FP32::class, FloatArray(ptoks.size) { ptoks[it].toFloat() })
            val pre = model.forwardPrefill(prefIn, ctx)
            var selfK = pre.selfK
            var selfV = pre.selfV
            val vocab = pre.logits.shape[pre.logits.rank - 1]
            var next = argmaxRow(pre.logits, ptoks.size - 1, vocab)

            // --- Decode: one token per step over the growing cache (mirrors the board contract) ---
            val gen = ArrayList<Int>()
            var pos = ptoks.size
            var steps = 0
            while (steps < 24) {
                if (next == eos) break
                gen.add(next)
                if (next == eot) break
                val tokIn = ctx.fromFloatArray<FP32, Float>(Shape(1), FP32::class, floatArrayOf(next.toFloat()))
                val rope = model.buildRopeCosSin(pos, ctx)
                val out = model.forwardWithPast(tokIn, rope, selfK, selfV, ctx)
                selfK = out.selfK
                selfV = out.selfV
                next = argmaxRow(out.logits, 0, vocab)
                pos++
                steps++
            }
            println("WITH_PAST gen=$gen (oracle=$oracle)")
            assertTrue(gen.isNotEmpty(), "no tokens generated")
            assertEquals(oracle, gen, "with_past 2-graph KV loop must match the oracle token-for-token")
        }
    }

    /** argmax over row [row] of a `[rows, vocab]` logits tensor. */
    private fun argmaxRow(logits: Tensor<FP32, Float>, row: Int, vocab: Int): Int {
        val data = logits.data.copyToFloatArray()
        val base = row * vocab
        var best = 0
        var bestV = Float.NEGATIVE_INFINITY
        for (i in 0 until vocab) {
            val v = data[base + i]
            if (v > bestV) { bestV = v; best = i }
        }
        return best
    }
}
