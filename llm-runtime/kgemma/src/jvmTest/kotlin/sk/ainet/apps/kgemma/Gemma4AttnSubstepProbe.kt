package sk.ainet.apps.kgemma

import kotlinx.coroutines.runBlocking
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.nn.transformer.MultiHeadAttentionDiag
import sk.ainet.lang.nn.transformer.mhaStatSink
import sk.ainet.lang.types.FP32
import sk.ainet.models.gemma.GemmaWeightLoader
import sk.ainet.models.gemma.GemmaNetworkLoader
import kotlin.test.Test

/**
 * Dumps Gemma 4 attention substeps for layer 0 and lines them up against llama.cpp.
 *
 * Established so far for E2B, prompt `[2, 105, 2364]`, token index 1 (`<|turn>`, id 105):
 * `token_embd` and `attn_norm-0` are exact; `q_norm`, `k_norm` and both RoPE outputs match within
 * Q4_K noise; the attention **output** (`node_32 = MUL_MAT(attn_output.weight, kqv_out-0)`) does
 * not. V is the one attention input not yet compared, and E2B is exactly where V-norm is enabled.
 *
 * llama.cpp reference (token 1, head 0), from `llama-eval-callback`:
 * ```
 * Vcur_normed-0  RMS_NORM   -0.7105  -1.4492  -0.9298
 * Qcur_normed-0  MUL         0.3417   0.3402   0.3911
 * Kcur_normed-0  MUL         0.4673  -0.2566  -0.0489
 * kqv_out-0      RESHAPE     0.0565   0.1085   0.0255
 * node_32        MUL_MAT    -0.7368  -0.5313  -0.8144   <- attention output
 * ```
 * Note the shapes: 8 query heads, **1 KV head** (MQA), head_dim 256 on this sliding layer.
 */
class Gemma4AttnSubstepProbe {

    @Test
    fun layer0_attention_substeps() {
        val gguf = System.getenv("GEMMA4_E2B_GGUF_PATH")
        if (gguf.isNullOrBlank() || System.getenv("GEMMA4_PROBE") != "1") {
            println("[skip] set GEMMA4_E2B_GGUF_PATH and GEMMA4_PROBE=1"); return
        }
        val ctx = DirectCpuExecutionContext.create()
        val weights = runBlocking {
            GemmaWeightLoader(
                randomAccessProvider = { JvmRandomAccessSource.open(gguf) },
            ).loadToMapStreaming<FP32, Float>(ctx, FP32::class)
        }
        val model = GemmaNetworkLoader.fromWeights(ctx, weights, FP32::class)
        val runtime = sk.ainet.apps.llm.OptimizedLLMRuntime(
            model, ctx, sk.ainet.apps.llm.OptimizedLLMMode.DIRECT, FP32::class,
            random = kotlin.random.Random.Default,
        )

        mhaStatSink = { label, t ->
            val v = runCatching { t.data.copyToFloatArray() }.getOrNull()
            if (v != null && v.isNotEmpty()) {
                println(
                    "MHA %-30s shape=%-18s [%+.4f, %+.4f, %+.4f]".format(
                        label.trim(), t.shape.toString().substringAfter("= ").take(16),
                        v[0], v.getOrElse(1) { 0f }, v.getOrElse(2) { 0f },
                    )
                )
            }
        }

        // Dump BOS's own pass too: its post-RoPE-K is exactly what the cache
        // should hold at position 0 for the next step.
        MultiHeadAttentionDiag.shouldDumpThisCall = true
        println("MHA ---- token 2 (BOS, position 0) ----")
        runtime.forward(2)
        println("MHA ---- token 105 (position 1) ----")
        runtime.forward(105)                     // the control token, position 1
        MultiHeadAttentionDiag.shouldDumpThisCall = false
        mhaStatSink = null

        println("MHA reference Vn  [-0.7105, -1.4492, -0.9298]")
        println("MHA reference Qn  [+0.3417, +0.3402, +0.3911]")
        println("MHA reference Kn  [+0.4673, -0.2566, -0.0489]")
        println("MHA reference out [-0.7368, -0.5313, -0.8144]")
    }
}
