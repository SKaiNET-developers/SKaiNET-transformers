package sk.ainet.apps.kllama

import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlinx.coroutines.runBlocking
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.MemorySegmentTensorData
import sk.ainet.lang.types.FP32
import sk.ainet.models.llama.DECODER_DEQUANTIZE_ALL
import sk.ainet.models.llama.LlamaNetworkLoader

/**
 * Verifies that `forwardBatched(IntArray)` produces the same last-position
 * logits as the equivalent autoregressive `forward(t)` per token. This is
 * the regression test the `bd3eb9c` revert was missing — without it,
 * batched prefill quietly diverged from the autoregressive baseline.
 *
 * Uses TinyLlama 1.1B Q8_0 (DEQUANTIZE_TO_FP32 policy → pure FP32 forward
 * pass). This sidesteps the Gemma 4 forward-pass correctness issues
 * tracked separately on develop, so this test is a clean check on the
 * batched-vs-autoregressive plumbing only.
 *
 * Skipped if the model is not present.
 */
class BatchedPrefillEquivalenceTest {

    companion object {
        private val MODEL_PATH = Path.of(
            System.getProperty("user.home"),
            ".lmstudio/models/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
        )
    }

    @Test
    fun `forwardBatched matches autoregressive at N=1`() {
        runEquivalence(intArrayOf(450)) // first prompt token only — should be trivial
    }

    @Test
    fun `forwardBatched matches autoregressive at N=2`() {
        runEquivalence(intArrayOf(450, 7483))
    }

    @Test
    fun `forwardBatched matches autoregressive prefill at last position`() {
        if (!MODEL_PATH.exists()) {
            println("[skip] Model not at $MODEL_PATH")
            return
        }
        runBlocking {
            // Fixed prompt — encode once, replay through both paths.
            // Tokenizer is loaded but the integer prompt is what we feed.
            val ctx = DirectCpuExecutionContext()
            val tokenizer = JvmRandomAccessSource.open(MODEL_PATH.toString()).use { source ->
                GGUFTokenizer.fromRandomAccessSource(source)
            }
            val prompt = "The capital of France is"
            val promptTokens = tokenizer.encode(prompt)
            require(promptTokens.size >= 2) { "Need ≥2 tokens to exercise the loop" }
            println("[diag] prompt tokens: ${promptTokens.toList()}")

            // --- Autoregressive baseline ---
            val autoLogits = run {
                val model = LlamaNetworkLoader.fromGguf(
                    randomAccessProvider = { JvmRandomAccessSource.open(MODEL_PATH.toString()) },
                    weightForm = DECODER_DEQUANTIZE_ALL
                ).load<FP32, Float>(ctx)
                val runtime = OptimizedLLMRuntime(
                    model = model,
                    ctx = ctx,
                    mode = OptimizedLLMMode.DIRECT,
                    dtype = FP32::class
                )
                var l: Tensor<FP32, Float> = runtime.forward(promptTokens[0])
                for (i in 1 until promptTokens.size) {
                    l = runtime.forward(promptTokens[i])
                }
                extractLogits(l)
            }

            // --- Batched ---
            val batchLogits = run {
                val model = LlamaNetworkLoader.fromGguf(
                    randomAccessProvider = { JvmRandomAccessSource.open(MODEL_PATH.toString()) },
                    weightForm = DECODER_DEQUANTIZE_ALL
                ).load<FP32, Float>(ctx)
                val runtime = OptimizedLLMRuntime(
                    model = model,
                    ctx = ctx,
                    mode = OptimizedLLMMode.DIRECT,
                    dtype = FP32::class
                )
                extractLogits(runtime.forwardBatched(promptTokens))
            }

            // --- Compare ---
            assertEquals(autoLogits.size, batchLogits.size,
                "logit vector length mismatch")
            val tol = 1e-3f
            var maxAbsDiff = 0f
            var maxRelDiff = 0f
            var argmaxAuto = 0
            var argmaxBatch = 0
            for (i in autoLogits.indices) {
                val a = autoLogits[i]
                val b = batchLogits[i]
                val d = kotlin.math.abs(a - b)
                if (d > maxAbsDiff) maxAbsDiff = d
                val r = if (kotlin.math.abs(a) > 1e-6f) d / kotlin.math.abs(a) else 0f
                if (r > maxRelDiff) maxRelDiff = r
                if (a > autoLogits[argmaxAuto]) argmaxAuto = i
                if (b > batchLogits[argmaxBatch]) argmaxBatch = i
            }
            println("[diag] max_abs_diff=$maxAbsDiff max_rel_diff=$maxRelDiff " +
                "argmax_auto=$argmaxAuto argmax_batch=$argmaxBatch " +
                "auto[argmax]=${autoLogits[argmaxAuto]} " +
                "batch[argmax]=${batchLogits[argmaxBatch]}")
            assertEquals(argmaxAuto, argmaxBatch,
                "argmax token differs: auto=$argmaxAuto batch=$argmaxBatch")
            assertTrue(maxAbsDiff < tol,
                "max_abs_diff=$maxAbsDiff exceeds tolerance $tol; " +
                    "batched prefill diverges from autoregressive")
        }
    }

    private fun runEquivalence(promptTokens: IntArray) {
        if (!MODEL_PATH.exists()) {
            println("[skip] Model not at $MODEL_PATH")
            return
        }
        runBlocking {
            val ctx = DirectCpuExecutionContext()
            println("[diag] N=${promptTokens.size} prompt tokens: ${promptTokens.toList()}")

            val autoLogits = run {
                val model = LlamaNetworkLoader.fromGguf(
                    randomAccessProvider = { JvmRandomAccessSource.open(MODEL_PATH.toString()) },
                    weightForm = DECODER_DEQUANTIZE_ALL
                ).load<FP32, Float>(ctx)
                val runtime = OptimizedLLMRuntime(
                    model = model, ctx = ctx,
                    mode = OptimizedLLMMode.DIRECT, dtype = FP32::class
                )
                var l: Tensor<FP32, Float> = runtime.forward(promptTokens[0])
                for (i in 1 until promptTokens.size) l = runtime.forward(promptTokens[i])
                extractLogits(l)
            }
            val batchLogits = run {
                val model = LlamaNetworkLoader.fromGguf(
                    randomAccessProvider = { JvmRandomAccessSource.open(MODEL_PATH.toString()) },
                    weightForm = DECODER_DEQUANTIZE_ALL
                ).load<FP32, Float>(ctx)
                val runtime = OptimizedLLMRuntime(
                    model = model, ctx = ctx,
                    mode = OptimizedLLMMode.DIRECT, dtype = FP32::class
                )
                extractLogits(runtime.forwardBatched(promptTokens))
            }
            assertEquals(autoLogits.size, batchLogits.size)
            var maxAbsDiff = 0f
            var argmaxAuto = 0
            var argmaxBatch = 0
            for (i in autoLogits.indices) {
                val d = kotlin.math.abs(autoLogits[i] - batchLogits[i])
                if (d > maxAbsDiff) maxAbsDiff = d
                if (autoLogits[i] > autoLogits[argmaxAuto]) argmaxAuto = i
                if (batchLogits[i] > batchLogits[argmaxBatch]) argmaxBatch = i
            }
            println("[diag] N=${promptTokens.size} max_abs_diff=$maxAbsDiff " +
                "argmax_auto=$argmaxAuto argmax_batch=$argmaxBatch " +
                "auto_top=${autoLogits[argmaxAuto]} batch_top=${batchLogits[argmaxBatch]}")
            assertEquals(argmaxAuto, argmaxBatch,
                "argmax differs at N=${promptTokens.size}")
            assertTrue(maxAbsDiff < 1e-3f,
                "max_abs_diff=$maxAbsDiff exceeds 1e-3 at N=${promptTokens.size}")
        }
    }

    private fun extractLogits(t: Tensor<FP32, Float>): FloatArray {
        val data = t.data
        return when (data) {
            is DenseFloatArrayTensorData<*> -> {
                val n = t.shape.volume
                if (data.buffer.size == n) data.buffer.copyOf()
                else data.buffer.copyOf(n)
            }
            is MemorySegmentTensorData<*> -> {
                val n = t.shape.volume
                val out = FloatArray(n)
                java.lang.foreign.MemorySegment.copy(
                    data.segment,
                    java.lang.foreign.ValueLayout.JAVA_FLOAT,
                    data.segmentByteOffset,
                    out, 0, n
                )
                out
            }
            else -> error("Unsupported tensor data type: ${data::class}")
        }
    }
}
