package sk.ainet.apps.kgemma.cli

import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.io.path.isDirectory
import kotlin.test.Test
import kotlinx.coroutines.runBlocking
import sk.ainet.apps.kgemma.GemmaIngestion
import sk.ainet.apps.kgemma.Gemma4LoadConfig
import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.tensor.data.MemorySegmentTensorDataFactory
import sk.ainet.lang.types.FP32

/**
 * Diagnostic for the "real Gemma 4 produces gibberish even though Phase 5f is
 * marked DONE" problem.
 *
 * Loads `gemma-4-E2B-it-Q4_K_M.gguf` via NATIVE_OPTIMIZED, greedy-decodes 8
 * tokens for the prompt `"Hi"` (BOS + 1 user token), and prints both the
 * generated text and the top-5 tokens at the very first decode step.
 *
 * Reference (llama-cpp-python 0.3.20 against the same Q4_K_M file, from
 * `gemma4-research/findings/reference_outputs.md`):
 * ```
 * PROMPT: 'Hi'        input tokens: [2, 10979]
 * generated:          ' = 100\n$1 = 10'
 * ```
 *
 * If our DSL produces a substring like ` =`, `100`, etc. anywhere in the
 * generated text, we're at structural parity with llama.cpp's Q4_K_M output.
 * If we produce `???? ?????` (the current symptom from the smoke test), there
 * is a real bug somewhere in the assembled forward pass that the synthetic
 * Phase 5f tests (GemmaDslPleTest, GemmaDslQ4KTest) failed to catch.
 *
 * Gated on `GEMMA4_E2B_MODEL_PATH` so CI stays green.
 */
class Gemma4ReferenceParityDiagnostic {

    @Test
    fun `greedy decode Hi first eight tokens prints top-5 first-step logits`() {
        val modelPath = System.getenv("GEMMA4_E2B_MODEL_PATH")?.trim().orEmpty()
        if (modelPath.isEmpty()) {
            println("[skip] GEMMA4_E2B_MODEL_PATH not set.")
            return
        }
        val path = Path.of(modelPath)
        if (!path.exists() || path.isDirectory() || !path.toString().endsWith(".gguf")) {
            println("[skip] Need a .gguf file at $modelPath")
            return
        }

        // Same bootstrap every production entry point performs. Without it the
        // keep-packed weights — including the tied lm_head, which rides
        // token_embd's PackedBlockStorage — fall to KernelDispatch's decoding
        // reference kernel, and a "diagnostic" run takes tens of minutes for a
        // handful of tokens (measured) instead of seconds.
        sk.ainet.apps.kgemma.KgemmaKernels.ensureInstalled()

        runBlocking {
            val memSegFactory = MemorySegmentTensorDataFactory()
            val ctx = DirectCpuExecutionContext(tensorDataFactory = memSegFactory)
            try {
                val ingestion = GemmaIngestion<FP32>(
                    ctx = ctx,
                    dtype = FP32::class,
                    config = Gemma4LoadConfig()
                )
                println("Loading $path via DSL engine loader (keep-packed)...")
                val runtime = ingestion.loadDslRuntimeStreaming(
                    randomAccessProvider = { JvmRandomAccessSource.open(path.toString()) }
                )
                val tokenizer = JvmRandomAccessSource.open(path.toString()).use { source ->
                    GGUFTokenizer.fromRandomAccessSource(source)
                }

                // Use llama.cpp's exact reference token sequence so any
                // forward-pass divergence is isolated from tokenizer differences.
                // GEMMA4_PROMPT_TOKENS=comma-separated overrides; default = BOS+Hi.
                val promptTokens = System.getenv("GEMMA4_PROMPT_TOKENS")?.let { csv ->
                    csv.split(',').map { it.trim().toInt() }.toIntArray()
                } ?: intArrayOf(2, 10979)
                println("Prompt tokens (REFERENCE): ${promptTokens.toList()}")
                println("Reference output (llama.cpp Q4_K_M, greedy): ' = 100\\n\$1 = 10'")
                println("BOS token id: ${tokenizer.bosTokenId}")
                println("Decode of token 10979: '${runCatching { tokenizer.decode(10979) }.getOrElse { "<err>" }}'")
                println("Decode of token 18428 (what our tokenizer produces for 'Hi'): '${runCatching { tokenizer.decode(18428) }.getOrElse { "<err>" }}'")
                println()

                // Prefill: by default, forward each prompt token autoregressively
                // (equivalent to decode loop). When GEMMA4_BATCHED_PREFILL=1, run
                // a single batched forward over the entire prompt — the LAST row
                // of the [N, vocab] output is the same prediction-for-next-token
                // logits the autoregressive path would produce. Different
                // numerically only if the autoregressive single-token prefill
                // has step-to-step accumulation drift.
                val batchedPrefill = System.getenv("GEMMA4_BATCHED_PREFILL") == "1"
                val lastLogits: sk.ainet.lang.tensor.Tensor<FP32, Float> = if (batchedPrefill) {
                    println("[diag] GEMMA4_BATCHED_PREFILL=1 → single batched forward over ${promptTokens.size} tokens")
                    // forwardBatched now returns the last-position logits
                    // directly (shape [vocab]) — no caller-side narrow.
                    runtime.forwardBatched(promptTokens)
                } else {
                    var l = runtime.forward(promptTokens[0])
                    for (i in 1 until promptTokens.size) {
                        l = runtime.forward(promptTokens[i])
                    }
                    l
                }

                // Top-5 of the very first decode step (after the last prompt token).
                val firstLogits = extractLogits(lastLogits)
                val top5 = firstLogits.toList()
                    .mapIndexed { idx, v -> idx to v }
                    .sortedByDescending { it.second }
                    .take(5)
                println("Top-5 logits after prompt:")
                for ((id, score) in top5) {
                    val piece = runCatching { tokenizer.decode(id) }.getOrElse { "<err>" }
                    println("  id=%6d score=%+.4f piece=%s".format(id, score, piece.replace("\n", "\\n")))
                }
                println()

                // Greedy 8 tokens from the argmax of lastLogits.
                val generated = mutableListOf<Int>()
                generated += top5.first().first
                for (step in 1 until 8) {
                    val logits = runtime.forward(generated.last())
                    val ids = extractLogits(logits)
                    var bestIdx = 0
                    var bestVal = ids[0]
                    for (i in 1 until ids.size) {
                        if (ids[i] > bestVal) { bestVal = ids[i]; bestIdx = i }
                    }
                    generated += bestIdx
                }
                val text = tokenizer.decode(generated.toIntArray())
                println("Generated tokens: $generated")
                println("Generated text:   '${text.replace("\n", "\\n")}'")
                println()
                println("Match against reference? Look for substrings: ' =', '100', '\$1'")
            } finally {
                memSegFactory.close()
            }
        }
    }

    private fun extractLogits(t: Any): FloatArray {
        // The runtime returns a Tensor<FP32, Float> with a 1-D vocab logits
        // payload. Pull a FloatArray view via the well-known data accessors.
        val tensor = t as sk.ainet.lang.tensor.Tensor<FP32, Float>
        val data = tensor.data
        return when (data) {
            is sk.ainet.lang.tensor.data.DenseFloatArrayTensorData<*> -> data.buffer.copyOf()
            is sk.ainet.lang.tensor.data.MemorySegmentTensorData<*> -> {
                val n = tensor.shape.volume
                val out = FloatArray(n)
                java.lang.foreign.MemorySegment.copy(
                    data.segment,
                    java.lang.foreign.ValueLayout.JAVA_FLOAT,
                    data.segmentByteOffset,
                    out, 0, n
                )
                out
            }
            // Any other TensorData (e.g. StorageFloatTensorData, which the 0.51
            // memory model produces for logits) reads back through the generic
            // façade rather than failing the diagnostic on a container type.
            else -> data.copyToFloatArray()
        }
    }
}
