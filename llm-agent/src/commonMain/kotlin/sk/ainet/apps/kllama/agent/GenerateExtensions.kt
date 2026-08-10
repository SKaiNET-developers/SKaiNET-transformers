package sk.ainet.apps.kllama.agent

import kotlin.random.Random
import sk.ainet.apps.llm.InferenceRuntime
import sk.ainet.apps.llm.PrefillStrategy
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.types.DType

/**
 * Generate tokens until an EOS token is produced or [maxTokens] is reached.
 *
 * Unlike batch generation, this function:
 * - Stops when the model emits [eosTokenId]
 * - Does NOT prepend BOS automatically (the caller is responsible for encoding the
 *   full prompt including special tokens via the chat template)
 * - Returns a [GenerateResult] with all generated tokens and decoded text
 *
 * @param prompt Encoded prompt token IDs (should include BOS if needed).
 * @param maxTokens Maximum number of tokens to generate.
 * @param eosTokenId The EOS token ID to stop on.
 * @param temperature Sampling temperature (0 = greedy).
 * @param random Random generator for sampling.
 * @param onToken Optional callback invoked for each generated token.
 * @param decode Optional function to decode a token ID to a string.
 * @param onPrefill Optional callback invoked as the prompt is ingested, with
 *   `(done, total)` where `done` counts the tokens already processed and
 *   `total` is `prompt.size`. Under [PrefillStrategy.Autoregressive] it fires
 *   once per token; under [PrefillStrategy.Batched] once per chunk. Useful
 *   for showing progress: on a CPU-only runtime prefill can dominate wall
 *   time before the first generated token appears.
 * @param prefillStrategy How to ingest the prompt. The default
 *   [PrefillStrategy.Autoregressive] preserves the historical one-token-per-
 *   forward behavior. [PrefillStrategy.Batched] feeds the prompt through
 *   [InferenceRuntime.forwardBatched] in chunks — typically 3–10× faster on
 *   long prompts. Equivalence with the autoregressive path is pinned by
 *   `BatchedPrefillEquivalenceTest` / `GenerateUntilStopPrefillEquivalenceTest`
 *   (the historical position-collapse and head-reshape divergences are fixed;
 *   see `swapSeqHeadDims` in MultiHeadAttention and the batch-aware RoPE).
 */
public fun <T : DType> InferenceRuntime<T>.generateUntilStop(
    prompt: IntArray,
    maxTokens: Int,
    eosTokenId: Int,
    temperature: Float = 0.8f,
    random: Random = Random.Default,
    onToken: ((Int) -> Unit)? = null,
    decode: ((Int) -> String)? = null,
    onPrefill: ((Int, Int) -> Unit)? = null,
    prefillStrategy: PrefillStrategy = PrefillStrategy.Autoregressive
): GenerateResult {
    if (prompt.isEmpty()) {
        return GenerateResult(emptyList(), "", false)
    }
    val total = prompt.size
    var lastLogits: Tensor<T, Float>? = null
    when (prefillStrategy) {
        is PrefillStrategy.Autoregressive -> {
            for ((i, tokenId) in prompt.withIndex()) {
                lastLogits = forward(tokenId)
                onPrefill?.invoke(i + 1, total)
            }
        }
        is PrefillStrategy.Batched -> {
            var i = 0
            while (i < total) {
                val end = minOf(i + prefillStrategy.maxBatch, total)
                lastLogits = forwardBatched(prompt.copyOfRange(i, end))
                i = end
                onPrefill?.invoke(i, total)
            }
        }
    }
    lastLogits ?: return GenerateResult(emptyList(), "", false)

    val generated = mutableListOf<Int>()
    val textBuilder = StringBuilder()
    var stoppedByEos = false

    var logits: Tensor<T, Float> = lastLogits
    for (step in 0 until maxTokens) {
        val nextToken = sampleFromLogits<T>(logits, temperature, random)

        if (nextToken == eosTokenId) {
            stoppedByEos = true
            break
        }

        generated.add(nextToken)
        onToken?.invoke(nextToken)
        decode?.let { textBuilder.append(it(nextToken)) }

        logits = forward(nextToken)
    }

    return GenerateResult(generated, textBuilder.toString(), stoppedByEos)
}

/**
 * Sample a token ID from a logits tensor.
 *
 * Delegates to the shared [sk.ainet.apps.llm.sampleFromLogits] after extracting the float array.
 *
 * @param logits The logits tensor (1D, vocabSize).
 * @param temperature Sampling temperature. Values <= 1e-6 use greedy (argmax).
 * @param random Random generator.
 * @return The sampled token ID.
 */
public fun <T : DType> sampleFromLogits(
    logits: Tensor<T, Float>,
    temperature: Float,
    random: Random = Random.Default
): Int {
    val buf = logits.toFloatArray()
    return sk.ainet.apps.llm.sampleFromLogits(buf, temperature, random)
}

/**
 * Extract a FloatArray from a tensor, creating a copy for safe mutation.
 */
private fun <T : DType> Tensor<T, Float>.toFloatArray(): FloatArray {
    val data = this.data
    if (data is FloatArrayTensorData<*>) return data.buffer.copyOf()
    return data.copyToFloatArray()
}
